import os
import sys
import math
import random
import logging

# sys.path.append('../..')
import numpy as np
import pandas as pd

from dataclasses import asdict
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from seq2seq_utils import SimpleSummarizationDataset, Seq2SeqDataset
from simpletransformers.config.model_args import Seq2SeqArgs
from transformers import AdamW, get_linear_schedule_with_warmup, \
    BartConfig, BartTokenizer, BartForConditionalGeneration, \
    T5Config, T5Tokenizer, T5ForConditionalGeneration, \
    AutoModel, AutoConfig, AutoTokenizer, \
    BertConfig, BertModel, BertTokenizer, \
    RobertaConfig, RobertaModel, RobertaTokenizer

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),  # bart使用了roberta的vocab.txt和tokenizer.json
    "t5": (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
}

GENERATIVE_MODEL_NAME = ['bart', 't5']
ENCODER_MODEL_NAME = ['bert', 'roberta']


class Sequence2SequenceModels:
    """
     A universal Seq2Seq class
    """

    def __init__(self,
                 encoder_type=None,
                 encoder_name=None,
                 decoder_name=None,
                 encoder_decoder_type=None,
                 encoder_decoder_name=None,
                 config=None,
                 args=None,
                 use_cuda=True,
                 cuda_device=-1,
                 **kwargs
                 ):
        """
        A standard Seq2SeqModel

        Warnings:
            1.If you want to use local model, please set encoder_name or decoder_name to the path of local model
            2.Don't set

        Args:
            encoder_type (optional): The type of model to use as the encoder.
            encoder_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            decoder_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
                                    Must be the same "size" as the encoder model (base/base, large/large, etc.)
            encoder_decoder_type (optional): The type of encoder-decoder model. (E.g. bart)
            encoder_decoder_name (optional): The path to a directory containing the saved encoder and decoder of a Seq2SeqModel. (E.g. "outputs/") OR a valid BART or MarianMT model.
            config (optional): A configuration file to build an EncoderDecoderModel.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.

        TODO: Completing different type models
        TODO: Add wandb module
        """
        self.args = self._load_model_args(encoder_decoder_name)  # load model args from Seq2SeqArgs
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        # print(type(args), args)
        logger.info(" Arguments Details:")
        for attr, value in args.items():
            logger.info(" {}: {}.".format(attr, value))

        # update_from_dict function code
        # def update_from_dict(self, new_values):
        #     if isinstance(new_values, dict):
        #         for key, value in new_values.items():
        #             setattr(self, key, value)  # setattr为对象增加属性和值
        #     else:
        #         raise (TypeError(f"{new_values} is not a Python dict."))

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:  # 使用cuda进行计算
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    print("cuda_device:", cuda_device)
                    logger.info("cuda_device:", cuda_device)
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                logger.info('No GPU! Using CPU!')
                self.device = 'cpu'
        else:
            logger.info("Mixed precision not available")
            self.args.fp16 = False

        if encoder_decoder_type:  # 设置对应导入的类
            config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_decoder_type]
        else:
            config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]

        if encoder_decoder_type in GENERATIVE_MODEL_NAME:
            self.model = model_class.from_pretrained(encoder_decoder_name)
            self.encoder_tokenizer = tokenizer_class.from_pretrained(encoder_decoder_name)
            self.decoder_tokenizer = self.encoder_tokenizer
            self.config = self.model.config

        # Set model name
        if encoder_decoder_name:  #
            self.args.model_name = encoder_decoder_name

            # # Checking if we are loading from a saved model or using a pre-trained model
            # if not saved_model_args and encoder_decoder_type == "marian":
            # Need to store base pre-trained model name to get the tokenizer when loading a saved model
            self.args.base_marian_model_name = encoder_decoder_name

        elif encoder_name and decoder_name:
            self.args.model_name = encoder_name + "-" + decoder_name
        else:
            self.args.model_name = "encoder-decoder"

        # set model type
        if encoder_decoder_type:
            self.args.model_type = encoder_decoder_type
        elif encoder_type:
            self.args.model_type = encoder_type + "-bert"
        else:
            self.args.model_type = "encoder-decoder"

    def train_model(self, train_data, output_dir=None, verbose=False, args=None, **kwargs):
        """
        :param train_data: train set data got by pandas
        :param output_dir: path to output file
        :param verbose: if logging
        :param args:
        :param kwargs:
        :return:
        """
        if args:
            self.args.update_from_dict(args)
        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data, verbose=verbose)
        self.train(train_dataset, output_dir, verbose=verbose)

        self._save_model(self.args.output_dir, model=self.model)

        if verbose:
            logger.info("Training of {} model complete. Saved to {}.".format(self.args.model_name, output_dir))

    def train(self, train_dataset, output_dir, verbose=True, **kwargs):
        """
        train function
        :param train_dataset:
        :param output_dir:
        :param verbose:
        :param kwargs:
        :return: None
        TODO: plan to add a complicated and robust optimizer
        TODO: plan to add more optimizer strategies
        """
        model = self.model
        args = self.args
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=self.args.dataloader_num_workers)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)  # parallelize the model
        logger.info("Training......")
        if args.fp16:
            logger.info("Using mixed precision.")
            from torch.cuda import amp
            scaler = amp.GradScaler()
        model.train()
        for i in range(args.num_train_epochs):
            logger.info(
                "---------------------------------------epoch:{}------------------------------------".format(i + 1))
            for step, batch in enumerate(train_dataloader):
                inputs = self._get_inputs_dict(batch)
                if args.fp16:
                    with amp.autocast():
                        # print("input_ids:", inputs)
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    # print("input_ids:", inputs)
                    outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                current_loss = loss.item()
                if step % 100 == 99:
                    logger.info("step:{0}, current_loss:{1}".format(step, current_loss))
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:  # 混合精度
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        # global_step += 1

    def _save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model into {output_dir}")

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            self._save_model_args(output_dir)

            if self.args.model_type in GENERATIVE_MODEL_NAME:
                os.makedirs(os.path.join(output_dir), exist_ok=True)
                model_to_save.save_pretrained(output_dir)
                self.config.save_pretrained(output_dir)
                if self.args.model_type in ["bart", "blender", "blender-large"]:
                    self.encoder_tokenizer.save_pretrained(output_dir)
            else:
                os.makedirs(os.path.join(output_dir, "encoder"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "decoder"), exist_ok=True)
                self.encoder_config.save_pretrained(os.path.join(output_dir, "encoder"))
                self.decoder_config.save_pretrained(os.path.join(output_dir, "decoder"))

                model_to_save = (
                    self.model.encoder.module if hasattr(self.model.encoder, "module") else self.model.encoder
                )
                model_to_save.save_pretrained(os.path.join(output_dir, "encoder"))

                model_to_save = (
                    self.model.decoder.module if hasattr(self.model.decoder, "module") else self.model.decoder
                )

                model_to_save.save_pretrained(os.path.join(output_dir, "decoder"))

                self.encoder_tokenizer.save_pretrained(os.path.join(output_dir, "encoder"))
                self.decoder_tokenizer.save_pretrained(os.path.join(output_dir, "decoder"))

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            # TODO:
            # if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
            #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _load_model_args(self, model_name):
        """
        Loading Seq2Seq arguments

        :param model_name:
        :return: loaded arguments
        """
        args = Seq2SeqArgs()
        args.load(model_name)
        return args

    def _save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _move_model_to_device(self):
        self.model.to(self.device)

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, verbose=True, silent=False):
        """
        Creates a T5Dataset from data.
        data:train_train_df
        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        encoder_tokenizer = self.encoder_tokenizer
        decoder_tokenizer = self.decoder_tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(encoder_tokenizer, decoder_tokenizer, args, data, mode)
        else:
            if args.model_type in GENERATIVE_MODEL_NAME:
                return SimpleSummarizationDataset(encoder_tokenizer, self.args, data, mode)
            else:
                return Seq2SeqDataset(encoder_tokenizer, decoder_tokenizer, self.args, data, mode, )

    def _get_inputs_dict(self, batch):
        """
        change data to input
        :param batch:
        :return:
        """
        device = self.device
        if self.args.model_type in ["marian"]:
            pad_token_id = self.encoder_tokenizer.pad_token_id
            source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "lm_labels": lm_labels.to(device),
            }
        elif self.args.model_type in GENERATIVE_MODEL_NAME:
            pad_token_id = self.encoder_tokenizer.pad_token_id
            source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone()
            labels[y[:, 1:] == pad_token_id] = -100
            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": labels.to(device),
            }
        else:
            lm_labels = batch[1]
            lm_labels_masked = lm_labels.clone()
            lm_labels_masked[lm_labels_masked == self.decoder_tokenizer.pad_token_id] = -100

            inputs = {
                "input_ids": batch[0].to(device),
                "decoder_input_ids": lm_labels.to(device),
                "labels": lm_labels_masked.to(device),
            }

        return inputs
