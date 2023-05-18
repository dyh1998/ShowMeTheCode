from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filter_total = num_filters * len(filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, n_class, bias=False)
        self.bias = nn.Parameter(torch.ones([n_class]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, hidden_size)) for size in filter_sizes
        ])  # nn.Conv2d(in_channel, out_channels, kernel_size=(size, hidden_size))，
        # 其中size不大于句子中隐藏层的层数，hidden_size表示句子的嵌入维度

    def forward(self, x):
        # x:[batch_size, encoder_layer, embed_size]
        x = x.unsqueeze(1)  # [bs, channel=1, seq, hidden]  [8, 1, 24, 1024]
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))  # [bs, channel=1, seq-kernel_size+1, 1]
            mp = nn.MaxPool2d(
                kernel_size=(encode_layer - filter_sizes[i] + 1, 1)
            )
            # mp: [bs, channel=3, w, h]
            pooled = mp(h).permute(0, 3, 2, 1)  # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)
        h_pool = torch.cat(pooled_outputs, 3)  # [bs, h=1, w=1, channel=896]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])

        output = self.Weight(h_pool_flat) + self.bias  # [bs, n_class]

        return output


class Bert_Blend_CNN(nn.Module):
    def __init__(self):
        super(Bert_Blend_CNN, self).__init__()
        self.bert = AutoModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.textcnn = TextCNN()

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 取每一层encode出来的向量
        # outputs.pooler_output: [bs, hidden_size]
        hidden_states = outputs.hidden_states  # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        # cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [bs, 1, hidden]
        mask_embeddings = hidden_states[1][:, 1, :].unsqueeze(1)
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
        # [batch_size, 1, 1024]:批次cls向量
        for i in range(2, len(hidden_states)):
            # cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
            mask_embeddings = torch.cat((mask_embeddings, hidden_states[i][:, 1, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=24, hidden]   [8, 24, 1024]
        # logits = self.textcnn(cls_embeddings)
        logits = self.textcnn(mask_embeddings)
        return logits
