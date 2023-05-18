import os
import logging
import json
from tqdm import tqdm
import collections
import numpy as np

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""将得到的摘要问答json文件处理成features cache, 以便dataset的加载"""

"""args.sum  args.squad, args.sum_doc_stride, args.extract_answer_num=3"""

BOS = "<s>"
CLS = "<s>"
PAD = "<pad>"
SEP = "</s>"
UNK = "<unk>"
MASK = "<mask>"


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def get_sum_squad_examples_and_features(tokenizer, file, args, output_examples=False, is_training=True):
    """将json文件处理成examples和features"""
    logger.info("Creating example and features from data file at %s", args.data_dir)
    examples = read_sum_squad_examples(file, is_training=is_training, debug=args.debug)

    features = convert_sum_examples_to_features(
                        examples,
                        tokenizer,
                        args.max_seq_length,
                        args.max_target_length,
                        args.sum_doc_stride,
                        args.max_query_length,
                        is_training
             )
    # cached-train-cnn-512
    cached_features_file = os.path.join(
        args.data_save,
        "cached-{}-{}-{}-{}".format(
            "dev" if not is_training else "train",
            args.sum_data_type,
            "features",
            str(args.max_seq_length),
        )
    )

    logger.info("Saving features into cached file to  %s", cached_features_file)
    torch.save(features, cached_features_file)

    if output_examples:
        cached_examples_file = os.path.join(
            args.data_save,
            "cached-{}-{}-{}-{}".format(
                "dev" if not is_training else "train",
                args.sum_data_type,
                "examples",
                str(args.max_seq_length)
            )
        )
        if not os.path.exists(cached_examples_file):
            logger.info(f"Creating cached_example_file -{cached_examples_file}")
            logger.info("Saving example into cached file to  %s", cached_examples_file)
            torch.save(examples, cached_examples_file)


def read_sum_squad_examples(
        input_file, is_training, version_2_with_negative=False,
        debug=False):
    """Read a Summary-SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    if debug:
        input_data = input_data[:5]

    for entry in input_data:
        paragraphs = entry["paragraphs"]
        for paragraph in paragraphs:
            paragraph_text = paragraph["context"]
            target = paragraph["target"]
            sentence_start_positions = paragraph['sentence_start_positions']    # 句子的个数不一样是否进行补全  这个在训练的时候是不需要进行补全的
            sentence_end_positions = paragraph['sentence_end_positions']

            doc_tokens = []   # 按照空格分词得到的doc_tokens
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            sentence_start_positions = [char_to_word_offset[pos] for pos in sentence_start_positions]
            sentence_end_positions = [char_to_word_offset[pos] for pos in sentence_end_positions]
            assert len(sentence_start_positions) == len(sentence_end_positions), \
                "sentence_start_positions is not eq sentence_end_positions"

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]

                # orig_answer_text = None   # 如果是验证集的话 orig_answer_text是None 不能用于评估
                is_impossible = False
                answers = None
                start_positions = []    # 一个问题对于摘要任务来说包含多个答案 最多3个  label
                end_positions = []
                orig_answer_texts = []   # 每一个抽取到的候选摘要的列表集合
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if not is_impossible:
                        answers = qa["answers"]        # 在summary qa中answer最多包含三个句子 并且通过最长的单词数进行限制
                        for answer in answers:
                            orig_answer_text = answer["text"]
                            orig_answer_texts.append(orig_answer_text)
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset +
                                                               answer_length - 1]
                            start_positions.append(start_position)
                            end_positions.append(end_position)
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(
                                doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                continue
                        assert len(start_positions) == len(end_positions), \
                            "Make sure the length of summary-squad for start positions is eq to end positions "
                        if len(start_positions) == 0:
                            continue
                        # if len(start_positions) < 3:   # 确保抽取到的句子数是3
                        #     start_positions.extend([-100] * (3 - len(start_positions)))  # 计算损失的时候会忽略-100
                        #     end_positions.extend([-100] * (3 - len(end_positions)))

                    else:
                        start_positions = [-1, -1, -1]
                        end_positions = [-1, -1, -1]
                        orig_answer_texts = ["", "", ""]

                else:
                    evaluate_answers = []
                    for answer in qa["answers"]:
                        evaluate_answers.append(answer["text"])

                    answers = ' '.join(evaluate_answers)     # 增加answers的部分用于评估


                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    target=target,
                    orig_answer_texts=orig_answer_texts,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    sentence_start_positions=sentence_start_positions,
                    sentence_end_positions=sentence_end_positions,
                    is_impossible=is_impossible,
                    answers=answers)
                examples.append(example)

    return examples


class SquadExample(object):
    """
       A single training/test example for the Summary Squad dataset.
       For examples without an answer, the start and end position are -1.
       """
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 target,
                 orig_answer_texts=None,
                 start_positions=None,
                 end_positions=None,
                 sentence_start_positions=None,
                 sentence_end_positions=None,
                 is_impossible=None,
                 answers=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_texts = orig_answer_texts
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.sentence_start_positions = sentence_start_positions
        self.sentence_end_positions = sentence_end_positions
        self.is_impossible = is_impossible
        self.answers = answers
        self.target = target


def convert_sum_examples_to_features(
        examples, tokenizer, max_seq_length, max_target_length,
        doc_stride, max_query_length, is_training
        ):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in tqdm(
            enumerate(examples), total=len(examples), desc="convert summary squad examples to features"
    ):

        query_tokens = tokenizer.tokenize(example.question_text)
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        target = tokenizer.tokenize(example.target)
        if len(target) < max_target_length:
            diff = max_target_length - len(target)
            target.extend([PAD] * diff)
        target_ids = tokenizer.convert_tokens_to_ids(target)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []   # tok表示BPE之后的tok
        orig_to_tok_index = []
        all_doc_tokens = []
        # 获取BPE之后的tok和未分词之前的token位置对应关系
        for (i, token) in enumerate(example.doc_tokens):

            orig_to_tok_index.append(len(all_doc_tokens))   # 只是针对文档进行处理的 没有加入query
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # 将句子的边界换成BPE分词之后的边界

        _sentence_start_positions = [orig_to_tok_index[pos] for pos in example.sentence_start_positions]   # 分词之后每一个句子的界限
        _sentence_end_positions = [orig_to_tok_index[pos] for pos in example.sentence_end_positions]


        tok_start_positions = []   # 表示答案在all_doc_tokens得到的下标(即分词之后的文章中对应的下标)
        tok_end_positions = []
        if is_training and example.is_impossible:
            tok_start_positions = [-1, -1, -1]
            tok_end_positions = [-1, -1, -1]
        if is_training and not example.is_impossible:
            for index, start_position in enumerate(example.start_positions):
                tok_start_position = orig_to_tok_index[start_position]
                if example.end_positions[index] < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_positions[index] + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_texts[index])  # 目前得到的是基于doc的开始和结束的位置
                tok_start_positions.append(tok_start_position)
                tok_end_positions.append(tok_end_position)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        # 将原文按照doc_stride确定span的开始位置 span的长度由max_tokens_for_doc决定的

        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset   # 文档中剩余的单词数
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))  # 以max_tokens_for_doc长度作为一个span进行划分
            if start_offset + length == len(all_doc_tokens):   # 对文档中所有的token都进行了划分
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):    # 在每一个span中进行input_ids的构建
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append(BOS)
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append(SEP)
            segment_ids.append(0)    # segment_ids为0的部分包括[CLS] + query + [SEP]三部分  但是query_ids的部分不包含[CLS][SEP]

            # 检查当前span下的句子的分界是否仍然存在
            doc_offset = len(query_tokens) + 2

            # 确定当前span的句子边界
            sentence_start_positions, sentence_end_positions = check_sentence_boundary(
                cur_span=doc_span,
                sentence_start_positions=_sentence_start_positions,
                sentence_end_positions=_sentence_end_positions,
                doc_offset=doc_offset)


            # 如果当前的的位置是处于最后的短句子的话 说明不包含任何一个完整的句子
            if not sentence_start_positions and not sentence_end_positions:
                continue

            context_tokens = list()
            context_tokens.append(BOS)    # 区分context tokens和tokens  context tokens表示的是文章  tokens表示的是加入query之后得到的模型的输入
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i   # doc_span.start表示tok在BPE分词之后的doc中的索引下标
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]    # 在span中加入query之后的tok的在原始文档中（未BPE分词）中的索引下标
                # 检验split_token_index是不是在上下文最大的窗口的span中
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)    # doc_span_index表示的划分的第几个span， split_token_index表示的是分词之后的tokend的index
                token_is_max_context[len(tokens)] = is_max_context   # 表示当前的tok是不是处在最大的上下文的span中
                tokens.append(all_doc_tokens[split_token_index])
                # segment_ids.append(1)
                context_tokens.append(all_doc_tokens[split_token_index])
            tokens.append(SEP)
            # segment_ids.append(1)
            context_tokens.append(SEP)

            # 构建交叉的segment ids   如果当前的span不包含任何完整句子的话  那么不可以使用get_segment_ids
            segment_ids = segment_ids + get_segment_ids(
                doc_offset=doc_offset,
                sentence_start_positions=sentence_start_positions,
                sentence_end_positions=sentence_end_positions,
                doc_span=doc_span
            )

            segment_ids.append(segment_ids[-1])  # 加入了最后的sep的segment

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(tokens)

            assert len(tokens) == len(input_mask)
            assert len(tokens) == len(segment_ids)
            assert len(input_mask) == len(segment_ids)

            # Zero-pad up to the sequence length.
            pad_segment_id = 1 - segment_ids[-1]
            while len(tokens) < max_seq_length:
                tokens.append(PAD)
                input_mask.append(0)
                segment_ids.append(pad_segment_id)
            if len(tokens) > max_seq_length:
                tokens = tokens[: max_seq_length]
                input_mask = input_mask[: max_seq_length]
                segment_ids = segment_ids[: max_seq_length]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_ids = np.asarray(input_ids, dtype=np.int32)
            input_mask = np.asarray(input_mask, dtype=np.uint8)
            segment_ids = np.asarray(segment_ids, dtype=np.uint8)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_positions = []    # BPE之后答案的最终位置
            end_positions = []

            start_targets = []
            end_targets = []

            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start     # 表示当前span在原始文档中开始的位置
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False

                # 针对summary-squad数据来说 规定如果答案中的三个句子一个都不包含那么就去除
                # 对于其中只有一两个句子属于答案的 需要重新确定答案的位置 答案的个数
                _tok_start_positions = []
                _tok_end_positions = []
                for tok_start_position, tok_end_position in zip(tok_start_positions, tok_end_positions):
                    if tok_start_position >= doc_start and tok_end_position <= doc_end:
                        _tok_start_positions.append(tok_start_position)
                        _tok_end_positions.append(tok_end_position)
                tok_start_positions = _tok_start_positions
                tok_end_positions = _tok_end_positions
                assert len(tok_start_positions) == len(tok_end_positions)
                if not tok_start_positions:
                    out_of_span = True

                if out_of_span:
                    start_positions = [0, 0, 0]
                    end_positions = [0, 0, 0]

                    start_targets = [0] * len(input_ids)
                    end_targets = [0] * len(input_ids)

                else:   # 答案的开始和结束的位置是在当前的span中
                    for index, tok_start_position in enumerate(tok_start_positions):
                        tok_end_position = tok_end_positions[index]
                        # 加入位置的偏移量，得到当前的span下答案的开始和结束位置
                        start_positions.append(tok_start_position - doc_start + doc_offset)
                        end_positions.append(tok_end_position - doc_start + doc_offset)

                    assert len(start_positions) == len(end_positions)
                    start_targets = len(input_ids) * [0]
                    end_targets = len(input_ids) * [0]
                    for index, start_pos in enumerate(start_positions):
                        end_pos = end_positions[index]
                        start_targets[start_pos] = 1    # 即答案的长度和input_ids的长度是一致的
                        end_targets[end_pos] = 1

                    assert len(start_targets) == len(input_ids)
                    assert len(start_targets) == len(end_targets)
                    assert len(end_targets) == len(input_ids)

                if out_of_span:
                    continue

            if is_training and example.is_impossible:
                start_targets = [0] * len(input_ids)
                end_targets = [0] * len(input_ids)

            if not is_training:
                # 对于验证集 同样需要target得到验证集上的损失函数的  和评估不同
                start_targets = [0] * len(input_ids)
                end_targets = [0] * len(input_ids)

                for index, start_pos in enumerate(start_positions):
                    end_pos = end_positions[index]
                    start_targets[start_pos] = 1
                    end_targets[end_pos] = 1

                assert len(start_targets) == len(input_ids)
                assert len(start_targets) == len(end_targets)
                assert len(end_targets) == len(input_ids)

                features.append(
                    ValidInputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        target=target,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        answer_texts=example.orig_answer_texts,
                        segment_ids=segment_ids,
                        q_ids=query_ids,
                        start_positions=start_targets,
                        end_positions=end_targets,
                        sentence_start_positions=sentence_start_positions,
                        sentence_end_positions=sentence_end_positions,
                        is_impossible=example.is_impossible))   # start_position表示的是当前span下答案的开始和结束的位置（包含问题）
                unique_id += 1
            else:
                features.append(
                   TrainInputFeatures(
                        input_ids=input_ids,
                        segment_ids=segment_ids,
                        input_mask=input_mask,
                        target=target,
                        start_positions=start_targets,
                        end_positions=end_targets,
                        sentence_start_positions=sentence_start_positions,
                        sentence_end_positions=sentence_end_positions,
                        ))  # start_position表示的是当前span下答案的开始和结束的位置（包含问题）
                unique_id += 1

    return features


class TrainInputFeatures(object):
    """A single set of features of train data"""
    def __init__(self,
                 input_ids,
                 segment_ids,
                 input_mask,
                 target,
                 start_positions=None,
                 end_positions=None,
                 sentence_start_positions=None,
                 sentence_end_positions=None,
                 ):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.sentence_start_positions = sentence_start_positions
        self.sentence_end_positions = sentence_end_positions
        self.target = target


class ValidInputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 target,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 answer_texts,
                 input_mask,
                 segment_ids,
                 q_ids,
                 start_positions=None,
                 end_positions=None,
                 sentence_start_positions=None,
                 sentence_end_positions=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.target = target
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.answer_texts = answer_texts
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.q_ids = q_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.sentence_start_positions = sentence_start_positions
        self.sentence_end_positions = sentence_end_positions
        self.is_impossible = is_impossible


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.检查当前的span是不是不是当前position位置token的最大上下文span"""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same因为窗口大小是相同的, of course).     maximum的上下文选择方式是选left token较少的
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def check_sentence_boundary(cur_span, sentence_start_positions, sentence_end_positions, doc_offset):
    """检查当前的span中包含的完整句子的界限"""
    sentence_start_positions_ = []
    sentence_end_positions_ = []

    doc_start = cur_span.start
    doc_end = cur_span.start + cur_span.length - 1

    for index, sentence_start_pos in enumerate(sentence_start_positions):
        sentence_end_pos = sentence_end_positions[index]
        if sentence_start_pos >= doc_start and sentence_end_pos <= doc_end:
            cur_sentence_start_pos = sentence_start_pos - doc_start + doc_offset
            cur_sentence_end_pos = sentence_end_pos - doc_start + doc_offset
            sentence_start_positions_.append(cur_sentence_start_pos)
            sentence_end_positions_.append(cur_sentence_end_pos)
    assert len(sentence_start_positions_) == len(sentence_end_positions_)

    return sentence_start_positions_, sentence_end_positions_


def get_segment_ids(doc_offset, sentence_start_positions, sentence_end_positions, doc_span):
    segment_ids = []

    if sentence_start_positions[0] == doc_offset:
        cur_segment_id = 1

    else:
        cur_segment_id = 1
        segment_ids.extend([cur_segment_id] * (sentence_start_positions[0] - doc_offset))
        cur_segment_id = 1 - cur_segment_id

    segment_start_pos = sentence_start_positions[0]
    for end_pos in sentence_end_positions:
        segment_ids.extend([cur_segment_id] * (end_pos - segment_start_pos + 1))
        segment_start_pos = end_pos + 1
        cur_segment_id = 1 - cur_segment_id

    # 可能结束的位置之后还有token 因此需要补充segment ids
    last_segment_ids = (doc_span.length - len(segment_ids)) * [cur_segment_id]
    segment_ids = segment_ids + last_segment_ids

    assert len(segment_ids) == doc_span.length, f"len(segment_ids)-{len(segment_ids)} != doc_len-{doc_span.length}"
    return segment_ids

