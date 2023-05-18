import torch
from transformers import BartTokenizer, BartForQuestionAnswering, BartForConditionalGeneration, BartModel
from d2l import torch as d2l
import os

tokenizer = BartTokenizer.from_pretrained('../../shared/local_models/bart-base')
model1 = BartForQuestionAnswering.from_pretrained('../../shared/local_models/bart-base')
model2 = BartForConditionalGeneration.from_pretrained('../../shared/local_models/bart-base')
for name, params in model1.named_parameters():
    print(name, params.shape)
print("-----------------------------------------------")
for name, params in model2.named_parameters():
    print(name,params.shape)
os._exit(0)
# model = BartForConditionalGeneration.from_pretrained('../../shared/local_models/bart-base')
# model = BartModel.
question, text = "EU rejects German callto boycott British lamb .", 'EU is an organization entity'

inputs = tokenizer(question, return_tensors='pt')
labels = tokenizer(text, return_tensors='pt')
# inputs = tokenizer(question, text, return_tensors='pt')
print("inputs:", inputs)
print("labels:", labels)

# inputs = {
#     'input_ids': inputs['input_ids'],
#     # 'token_type_ids': inputs['token_type_ids'],
#     "attention_mask": inputs['attention_mask'],
#     "labels": inputs['labels']
# }

# device = d2l.try_gpu()
# print(device)
# model.to(device)
# outputs = model()
# with torch.no_grad():
output = model(labels=labels, inputs=inputs)
# import torch
# import torch.nn.functional as F
# from torch import Tensor, nn
# from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
# from transformers.modeling_bart import shift_tokens_right

# class MyBart(BartForConditionalGeneration):
#     def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
#                 decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
#                 use_cache=False, is_training=False):
#
#         if is_training:
#             _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
#         else:
#             _decoder_input_ids = decoder_input_ids
#
#         outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             encoder_outputs=encoder_outputs,
#             decoder_input_ids=_decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             decoder_cached_states=decoder_cached_states,
#             use_cache=use_cache,
#         )
#         lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
#         if is_training:
#             loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.config.pad_token_id)
#             loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
#                             decoder_input_ids.view(-1))
#             return loss
#         return (lm_logits,) + outputs[1:]


# print(outputs)
# answer_start_index = outputs.start_logits.argmax()
# answer_end_index = outputs.end_logits.argmax()

# print(answer_start_index, answer_end_index)
# predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
# print(tokenizer.decode(predict_answer_tokens))

# print(output)
print(output[0].shape, "start_logits")
print(output[0], "start_logits")
print(output[1].shape, "end_logits")
print(output[1], "end_logits")

# print(output[2], "past_key_values")
# print(output[3].shape, "encoder_last_hidden_states")
# print(output)
# print(len(output))
