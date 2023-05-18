import torch
from torch import nn
from d2l import torch as d2l
import sys

batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)

for t in train_iter:
    print(len(t))



# for train in train_iter:
#     print(train)
#     print(type(train), len(train))
#     print(train[0], train[0].shape)
#     print(train[1], train[1].shape)
#     print(train[2], train[2].shape)
#     print(train[3], train[3].shape)
#     print(train[4], train[4].shape)
#     print(train[5], train[5].shape)
#     print(train[6], train[6].shape)
#     break
# break


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs.

    Defined in :numref:`sec_bert`"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class MaskLM(nn.Module):
    """The masked language model task of BERT.

    Defined in :numref:`subsec_bert_input_rep`"""

    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)

        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)

        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class BERTEncoder(nn.Module):
    """BERT encoder.

    Defined in :numref:`subsec_bert_input_rep`"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class BERTModel(nn.Module):
    """The BERT model.

    Defined in :numref:`subsec_nsp`"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                                   dropout, max_len=max_len, key_size=key_size,
                                   query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        print("encode_x:", encoded_X)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
# print(net)
devices = d2l.try_gpu()
print(devices)
loss = nn.CrossEntropyLoss()


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    print(mlm_Y_hat)
    print(mlm_Y_hat.shape)
    print("mlm_y:", mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * \
            mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    #     net = net.to(devices)
    #     trainer = torch.optim.Adam(net.parameters(), lr=1e-3)
    #     step, timer = 0, d2l.Timer()
    #     # animator = d2l.Animator(xlabel='step', ylabel='loss',
    #     #                         xlim=[1, num_steps], legend=['mlm', 'nsp'])
    #     # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    #     metric = d2l.Accumulator(4)
    #     num_steps_reached = False
    #     while step < num_steps and not num_steps_reached:
    for tokens_X, segments_X, valid_lens_x, pred_positions_X, \
        mlm_weights_X, mlm_Y, nsp_y in train_iter:
        tokens_X = tokens_X
        segments_X = segments_X
        valid_lens_x = valid_lens_x
        pred_positions_X = pred_positions_X
        mlm_weights_X = mlm_weights_X
        mlm_Y, nsp_y = mlm_Y, nsp_y
        #         trainer.zero_grad()
        #         timer.start()
        mlm_l, nsp_l, l = _get_batch_loss_bert(
            net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
            pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
        break
    #         l.backward()
    #         trainer.step()
    #         metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
    #         timer.stop()
    #         animator.add(step + 1,
    #                      (metric[0] / metric[3], metric[1] / metric[3]))
    #         step += 1
    #         if step == num_steps:
    #             num_steps_reached = True
    #             break
    #
    # print(f'MLM loss {metric[0] / metric[3]:.3f}, '
    #       f'NSP loss {metric[1] / metric[3]:.3f}')
    # print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
    #       f'{str(devices)}')


train_bert(train_iter, net, loss, len(vocab), devices, 50)
