
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.bart_model import BartModel, PretrainedBartModel, BartQuestionAnsweringHead
from model.bart_model import _make_linear_from_emb


"""需要逐层训练"""
# 需要修改bartmodel的输出的格式
class EncoderDecoderQuestionAnsweringBart(PretrainedBartModel):
    def __init__(self, config):
        super(EncoderDecoderQuestionAnsweringBart, self).__init__(config)
        base_model = BartModel(config)
        self.model = base_model
        self.qa_head = BartQuestionAnsweringHead(inner_dim=config.d_model)
        self.label_smoothing = config.label_smoothing
        self.predict_dense = nn.Linear(config.d_model * 2, config.d_model)   # 将拼接的表征转换成d_model维度
        self.init_linear_weight(self.predict_dense)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """扩充词表的大小"""
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens, old_num_tokens):
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def init_linear_weight(self, l):
        assert isinstance(l, nn.Linear), "don't init linear weights for not Linear Module"
        nn.init.xavier_uniform_(l.weight)
        nn.init.constant_(l.bias, 0)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            segment_ids=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_cached_states=None,
            summary_labels=None,
            start_positions=None,
            end_positions=None,
            sentence_start_positions=None,
            sentence_end_positions=None,
            use_cache=False,
            **unused
    ):
        """一共包含三个损失 encoder端的问答损失 decoder端的摘要损失 以及en-decoder端的互信息损失(这一部分的损失暂时不计
        将encoder的hiddens和decoder进行拼接计算decoder端的输出)"""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            token_type_ids=segment_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

        encoder_hiddens = outputs["encoder_outputs"][0]
        decoder_hiddens = outputs["decoder_outputs"][0]
        loss = 0
        if start_positions is not None and end_positions is not None:
            start_logits, end_logits = self.qa_head(encoder_hiddens)
            qa_loss_fun = nn.BCEWithLogitsLoss()
            start_positions = start_positions.float()
            end_positions = end_positions.float()
            start_loss = qa_loss_fun(start_logits, start_positions)
            end_loss = qa_loss_fun(end_logits, end_positions)
            qa_loss = start_loss + end_loss
            loss += qa_loss
            outputs["qa_loss"] = qa_loss
            # 计算encoder和decoder的互信息损失
            # 得到encoder端抽取摘要的平均表征 然后拼接到decoder端进行一起进行预测
            encoder_extract_sumamry = get_encoder_summary_representation(
                hiddens=encoder_hiddens,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=3,
                sentence_start_positions=sentence_start_positions,
                sentence_end_positions=sentence_end_positions
            )

            # 将得到的encoder的摘要表示拼接到decoder上进行预测

            batch_size, seq_len, hidden_size = encoder_hiddens.shape
            encoder_extract_sumamry = encoder_extract_sumamry.unsqueeze(1).expand(batch_size, seq_len, hidden_size)

            # 得到拼接后的decoder的表征
            decoder_hiddens = torch.cat([decoder_hiddens, encoder_extract_sumamry], dim=-1)
            decoder_hiddens = self.predict_dense(decoder_hiddens)

        summary_logits = F.linear(decoder_hiddens, self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (summary_logits, ) + outputs[2:]
        if summary_labels is not None:
            sum_loss_fun = nn.CrossEntropyLoss()
            sum_loss = sum_loss_fun(summary_logits, summary_labels)
            outputs["summary_loss"] = sum_loss
            loss += sum_loss
            outputs["loss"] = loss

        return outputs    # 返回值是一个字典

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)


def _get_best_indexes(start_logits, end_logits, n_best_size, sentence_start_positions, sentence_end_positions):
    """Given sentence_start_positions or sentence_end_positions to get n-best logits from a logits list
    找到当前feature 抽取到的摘要的开始位置和结束位置"""

    assert len(sentence_start_positions) == len(sentence_end_positions)
    n_best_size = n_best_size if n_best_size < len(sentence_start_positions) else len(sentence_end_positions)
    sentence_start_positions_logits = start_logits[sentence_start_positions]
    sentence_end_positions_logtis = end_logits[sentence_end_positions]
    logits = sentence_start_positions_logits + sentence_end_positions_logtis  # 因为我们选择的是一个句子 因此start_logtis和end_logits是成对出现的

    # 对当前的logits进行排序
    index_and_scores = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True
    )

    index_and_scores = index_and_scores[: n_best_size]

    # 抽取到的句子的开始和结束位置
    extract_start_positions = []
    extract_end_positions = []
    _start_logits = []
    _end_logits = []
    for index_and_score in index_and_scores:
        pos_index = index_and_score[0]
        extract_start_positions.append(sentence_start_positions[pos_index])
        extract_end_positions.append(sentence_end_positions[pos_index])
        _start_logits.append(start_logits[sentence_start_positions[pos_index]])
        _end_logits.append(end_logits[sentence_end_positions[pos_index]])   # 找到对应位置的logits

    return extract_start_positions, extract_end_positions, _start_logits, _end_logits


def get_encoder_summary_representation(
        hiddens, start_logits, end_logits, n_best_size,
        sentence_start_positions, sentence_end_positions
):
    """

    :param hiddens: batch_size, seq_len, hidden_dim
    :param start_logits:
    :param end_logits:
    :param n_best_size:
    :return: 针对整个batch输入返回抽取到的摘要的平均表征 1, som_len, hidden_dim->1, 1, hidden_dim
    """

    batch_size, seq_len, hidden_size = hiddens.shape
    batch_extract_hiddens = []   # 保存当前batch下抽取到的摘要的信息平均表征
    for i in range(batch_size):
        cur_hiddens = hiddens[i]   # 取当前sequence的hiddens 进行抽取
        cur_start_logits = start_logits[i]
        cur_end_logits = end_logits[i]
        cur_sentence_start_positions = sentence_start_positions[i]
        cur_sentence_end_positions = sentence_end_positions
        extract_start_positions, extract_end_positions, _, _ = _get_best_indexes(
            cur_start_logits, cur_end_logits, n_best_size,
            cur_sentence_start_positions, cur_sentence_end_positions
        )

        extract_poses = []
        for start_pos, end_pos in zip(extract_start_positions, extract_end_positions):
            extract_poses.extend(range(start_pos, end_pos + 1))
        extract_poses = torch.tensor(extract_poses)
        batch_extract_hiddens.append(torch.mean(cur_hiddens[extract_poses, :], dim=0))

    # 将列表的tensor张量转化为batch_size, hiddens_dim
    batch_average_hiddens = torch.zeros(batch_size, hidden_size)
    for i in range(batch_size):
        batch_average_hiddens.data[i] = batch_extract_hiddens[i].data

    assert batch_average_hiddens.shape == (batch_size, hidden_size)

    return batch_average_hiddens





        


