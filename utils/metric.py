def cal_correct_gold_pred(attention_mask, labels, pred):
    correct, total_gold, total_pred = 0, 0, 0
    for i in range(len(attention_mask)):
        # print(i)
        p = pred[i]
        label = labels[i]
        length = sum(attention_mask[i])
        pred_span = result2span_list(p[:length])
        label_span = result2span_list(label[:length])
        total_pred += len(pred_span)
        total_gold += len(label_span)
        for m in pred_span:
            for n in label_span:
                if m[0] == n[0] and m[1] == n[1]:
                    correct += 1
                    break
    return correct, total_gold, total_pred


def cal_prf(correct, total_gold, total_pred):
    p = correct / total_pred if correct > 0 else 0.0
    r = correct / total_gold if correct > 0 else 0.0
    f1 = 2 * p * r / (p + r) * 100 if correct else 0.0
    return f1, p * 100, r * 100


def result2span_list(label_list):
    """
    将标签组织成span
    """
    span_list = []
    start_index = 0
    end_index = 0
    if self.args.model == 'tempalte':
        for l in range(1, len(label_list)):
            if label_list[l] == tokenizer.convert_token_to_ids('entity'):
                if label_list[l - 1] != tokenizer.convert_token_to_ids('entity'):
                    start_index = l
                else:
                    end_index = l

            if label_list[l] != tokenizer.convert_token_to_ids('entity'):
                if label_list[l - 1] == tokenizer.convert_token_to_ids('entity'):
                    end_index = l
                    if end_index > start_index:
                        span_list.append((start_index, end_index))

    if self.args.num_labels == 3:
        for l in range(1, len(label_list)):
            if label_list[l] == 1:
                if label_list[l - 1] == 0:
                    start_index = l
                elif label_list[l - 1] == 2:
                    end_index = l
                    span_list.append((start_index, end_index))
                    start_index = l
                else:
                    pass
            if label_list[l] == 2:
                pass
            if label_list[l] == 0:
                if label_list[l - 1] != 0:
                    end_index = l
                    span_list.append((start_index, end_index))
                else:
                    pass
    elif self.args.num_labels == 2:
        for l in range(1, len(label_list)):
            if label_list[l] == 1:
                if label_list[l - 1] == 0:
                    start_index = l
                else:
                    end_index = l
            if label_list[l] == 0:
                if label_list[l - 1] == 1:
                    end_index = l
                    if end_index > start_index:
                        span_list.append((start_index, end_index))

    return span_list
