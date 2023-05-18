from torch import nn
from TorchCRF import CRF
from torch.nn.functional import cross_entropy, relu
import torch.nn.functional as F

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)
}

LOCAL_PATH = {
    'bert': '../predata/bert-base-uncased',
    'biobert': '../predata',
    'roberta': '../predata/roberta-base',
}
LOSS_FUNC = {
    'cross_entropy': nn.CrossEntropyLoss(),
    'rdrop': RDrop(),
    'focal_loss': FocalLoss()
}
class BertCRF(nn.Module):
    """
    几种不同的标注方式：
    1、BIO: num_labels = 3  # BIO会减少目标领域中少样本的实例数量
    2、IO: num_labels = 2  # IO无法很好的分清边界，但是在少样本简单的NER任务中还是有优势的
    3、首尾标记  # 这个待考虑
    """

    def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
                 local_path: bool = True, args=None) -> None:
        super(BertCRF, self).__init__()
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name
        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]
        self.model = model_class.from_pretrained(self.encoder_name)
        self.config = self.model.config
        self.num_labels = num_labels

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.crf = CRF(num_labels=num_labels)
        self.position_wise_ffn = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, x, tags=None):
        outputs = self.model(**x)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ffn(last_encoder_layer)
        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, tags, x['attention_mask']), self.crf.viterbi_decode(
                emissions, x['attention_mask'])
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.viterbi_decode(emissions, x['attention_mask'])
            return sequence_of_tags


class BertBiLSTMCRF(nn.Module):
    """
    BertBiLSTMCRF
    """

    def __init__(self, encoder_type: str = None, encoder_name: str = None, num_labels: int = 0,
                 local_path: bool = True, args=None) -> None:
        super(BertBiLSTMCRF, self).__init__()
        if args is not None:
            self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        self.encoder_name = encoder_name
        if local_path:
            self.encoder_name = LOCAL_PATH[self.encoder_name]
        self.model = model_class.from_pretrained(self.encoder_name)
        self.config = self.model.config
        self.num_labels = num_labels
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.crf = CRF(num_labels=num_labels)
        self.bilstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=(self.config.hidden_size) // 2,
                              num_layers=2, dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)

        self.position_wise_ffn = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, x, tags=None):
        """
        :param x:
        :param tags: torch.tensor(batch,length_attention_mask)
                if tags==0:training else:evaluating
        :return:
        """

        outputs = self.model(**x)
        last_encoder_layer = outputs[0]
        last_encoder_layer = self.dropout(last_encoder_layer)
        outputs, hc = self.bilstm(last_encoder_layer)
        emissions = self.position_wise_ffn(outputs)

        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions, tags, x['attention_mask']), self.crf.viterbi_decode(
                emissions, x['attention_mask'])
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.viterbi_decode(emissions, x['attention_mask'])
            return sequence_of_tags
