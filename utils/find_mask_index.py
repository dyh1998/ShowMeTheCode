# [MASK]位置发现
mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
int_pred_id = int(predicted_token_id[0])
int_label_id = int(labels[:, mask_token_index][:, 0][0])