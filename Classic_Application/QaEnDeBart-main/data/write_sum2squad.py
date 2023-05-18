import re
import os
import hashlib
from tqdm import tqdm
import json
from nltk.tokenize import sent_tokenize
import argparse

"""将dm处理成squad的形式"""


def hashhex(s):
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    """
    根据golden summary的列表选择summary size个数的原文中的句子作为候选摘要
    选择当前doc_sent_list中的三个句子作为候选摘要
    :param doc_sent_list:  包含多个自然句子list的原文档的列表
    :param abstract_sent_list:
    :param summary_size: 选择候选摘要的个数
    :return:
    """
    def _rouge_clean(s):
         return re.sub(r'[^a-zA-Z0-9 ]', '', s)       # 对单个token进行匹配 将不是数字或者字母或者空格的字符替换成空

    max_rouge = 0.0
    abstract_words = sum(abstract_sent_list, [])     # 获取gold摘要中的所有token 返回一个列表
    abstract_words = _rouge_clean(' '.join(abstract_words)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]

    evaluated_one_grams = [get_word_ngrams(1, [sent]) for sent in sents]
    evaluated_two_grams = [get_word_ngrams(2, [sent]) for sent in sents]

    reference_one_grams = get_word_ngrams(1, [abstract_words])
    reference_two_grams = get_word_ngrams(2, [abstract_words])

    selected = []   # 选择句子的rouge值高的前几个句子作为候选摘要 得到抽取式摘要的标签
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_one_grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_two_grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))

            rouge_1 = cal_rouge(candidates_1, reference_one_grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_two_grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def cal_rouge(evaluate_ngrams, reference_ngrams):
    """reference 表示的是gold summary"""
    reference_count = len(reference_ngrams)
    evaluate_count = len(evaluate_ngrams)

    overlapping_ngrams = evaluate_ngrams.intersection(reference_ngrams)   # 获得交集
    overlapping_count = len(overlapping_ngrams)

    if evaluate_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluate_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 *((precision * recall) / (precision + recall + 1e-8))

    score_dict = {
        "p": precision,
        "r": recall,
        "f": f1_score
    }

    return score_dict


def get_word_ngrams(n, sentences):
    """
    calculate word n-grams for multiple sentences
    :param n:   n-gram
    :param sentences:  list： including some sentences
    :return:
    """

    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])    # 所有单词的列表 包含重复的元素

    return get_ngram(n, words)


def get_ngram(n, words):
    """
    calculate n-grams
    :param n: n-gram
    :param words:  list of tokens
    :return: a set of n-grams   set中的每一个ngram都是一个元组
    """

    ngram_set = set()
    text_length = len(words)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start):
        ngram_set.add(tuple(words[i: i + n]))

    return ngram_set


def format_to_json(args):
    """使用多线程处理原数据集到模型指定的输入形式 读取整个数据集 处理成 bart类型，"""
    if args.dataset_type != '':
        dataset_type = [args.dataset_type]
    else:
        dataset_type = ['train', 'val']      # 分别处理训练集 验证集 和 测试集

    for corpus_type in dataset_type:
        corpus_source_path = os.path.join(args.data_path, corpus_type + ".source")
        corpus_target_path = os.path.join(args.data_path, corpus_type + ".target")
        dct = {'data': [], 'version': args.data_name + "-" + corpus_type + "-squad1.1"}
        items = 0
        ### 与squad1.1数据集有一些不同， 每一个paragraphs包含一篇文档，每篇文档一个问题 但是包含有多答案
        with open(corpus_source_path, 'r', encoding="utf-8") as source, \
                open(corpus_target_path, 'r', encoding="utf-8") as target:
            source_lines = source.readlines()
            target_lines = target.readlines()
            for index, (source_line, target_line) in tqdm(enumerate(zip(source_lines, target_lines))):
                items += 1
                cur_data = {}
                paragraphs = []
                paragraph = {}

                # 处理每一个句子到自然句,并且去除掉短句子
                source_sents, source_start_pos, source_end_pos, source_line = doc_sentences_to_list(
                    source_line, args, name="source"
                )
                target_sents = doc_sentences_to_list(target_line, args, name="target")
                if len(source_sents) < args.min_src_nsents:
                    continue
                paragraph['context'] = source_line
                paragraph['sentence_start_positions'] = source_start_pos
                paragraph['sentence_end_positions'] = source_end_pos
                paragraph["target"] = target_line

                cur_data['title'] = ' '.join(source_sents[0][:4])
                qas = []
                sent_labels = greedy_selection(source_sents[: args.max_src_nsents], target_sents, 3)   # 最多三个句子

                # 获取每一个标签句子在原文档中地位置
                answers = []
                for label in sent_labels:
                    ans = {}
                    label_sent = source_sents[label]
                    text = ' '.join(label_sent)
                    answer_start = source_line.find(text)
                    ans['answer_start'] = answer_start
                    ans['text'] = text
                    answers.append(ans)
                cur_qas = {}
                cur_qas['answers'] = answers
                cur_qas['question'] = 'what is the summary of the article ?'
                cur_qas_str = ' '.join(source_sents[0])     # 取第一句话最为id生成
                cur_qas['id'] = hashhex(cur_qas_str)
                qas.append(cur_qas)
                paragraph['qas'] = qas
                paragraphs.append(paragraph)
                cur_data['paragraphs'] = paragraphs

                dct['data'].append(cur_data)

            write_path = os.path.join(args.save_path, dct["version"] + ".json")

            with open(write_path, 'w', encoding="utf-8") as f:
                json.dump(dct, f, indent=4)


def doc_sentences_to_list(document, args, name):
    sents = []
    sent_start_pos = []
    sent_end_pos = []
    nature_sents = sent_tokenize(document)
    document_temp = []
    if name == "source":
        for i, sent in enumerate(nature_sents):
            if len(sent.split()) < args.min_src_ntokens_per_sent:
                continue
            cur_sent_split = sent.split()[:args.max_src_ntokens_per_sent]    # 获取当前句子的
            cur_sent = ' '.join(cur_sent_split)
            sents.append(cur_sent_split)
            document_temp.append(cur_sent)
            cur_document = ' '.join(document_temp)
            start_pos = cur_document.rfind(cur_sent)
            end_pos = start_pos + len(cur_sent) - 1
            sent_start_pos.append(start_pos)
            sent_end_pos.append(end_pos)

        document = ' '.join([' '.join(s) for s in sents])

        assert len(sent_start_pos) == len(sent_end_pos), "make sure the length of start pos is equal to that of end pos"
        return sents, sent_start_pos, sent_end_pos, document
    else:
        for sent in nature_sents:
            sents.append(sent.split())
        return sents


def get_rouge_extract_summary(args):
    test_source_file = os.path.join(args.data_path, "test.source")
    test_target_file = os.path.join(args.data_path, 'test.target')
    summary = []
    with open(test_source_file, 'r', encoding="utf-8") as source, \
            open(test_target_file, 'r', encoding="utf-8") as target:
        source_lines = source.readlines()
        target_lines = target.readlines()
        for index, (source_line, target_line) in tqdm(enumerate(zip(source_lines, target_lines))):
            cur_summary = []

            # 处理每一个句子到自然句,并且去除掉短句子
            source_sents, source_start_pos, source_end_pos, source_line = doc_sentences_to_list(
                source_line, args, name="source"
            )
            target_sents = doc_sentences_to_list(target_line, args, name="target")
            if len(source_sents) < args.min_src_nsents:
                continue

            sent_labels = greedy_selection(source_sents[: args.max_src_nsents], target_sents, 3)  # 最多三个句子
            for label in sent_labels:
                cur_summary.append(" ".join(source_sents[label]).strip())
            summary.append(" ".join(cur_summary)+'\n')

        write_path = os.path.join(args.data_path, "rouge_test_extract_summary.txt")

        with open(write_path, 'w', encoding="utf-8") as f:
            f.writelines(summary)



if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', default="E:\\data\\owndeal_cnndm\\dm\\", type=str
    )
    parser.add_argument(
        '--dataset_type', default="train", type=str
    )
    parser.add_argument(
        '--max_src_nsents', type=int, default=100
    )
    parser.add_argument(
        '--min_src_nsents', type=int, default=3
    )
    parser.add_argument(
        '--max_src_ntokens_per_sent', type=int, default=200
    )
    parser.add_argument(
        '--min_src_ntokens_per_sent', type=int, default=5
    )
    parser.add_argument(
        '--data_name', type=str, help="cnn dm xsum or nyt"
    )
    parser.add_argument(
        '--save_path', type=str, help="json data saved path"
    )

    args = parser.parse_args()
    format_to_json(args)
    # get_rouge_extract_summary(args)

