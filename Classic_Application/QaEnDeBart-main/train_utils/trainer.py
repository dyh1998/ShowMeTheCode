import os
import logging
from tqdm import tqdm, trange
import random

import numpy as np
import torch

from transformers.optimization import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def trainer(
        args,
        train_dataloader,
        valid_dataloader,
        model):
    """Train model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if args.total_num_update > 0:
        t_total = args.total_num_update
        length = len(train_dataloader)
        args.epoch = args.total_num_update // (length // args.gradient_accumulation_steps) + 1
    else:
        length = len(train_dataloader)
        t_total = length // args.gradient_accumulation_steps * args.epochs

    # Prepare optimizer and schedule(polynomial_decay and warmup)
    no_decay = ["bias", "LayerNorm.weight"]  # 优化器这一块需要和fairseq进行对比和修正

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_eps)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                          num_warmup_steps=args.warmup_steps,
                                                          num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.output_dir, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.output_dir, "scheduler.pt")
    ):
        # load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from http://wwww.github.com/nvidia/apex/ to use"
                              "fp16 training")
        # opt_level设置
        # O0：纯FP32训练，可以作为accuracy的baseline；
        # O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
        # O2：“几乎FP16”混合精度训练，不存在黑白名单，除了BatchNorm，几乎都是用FP16计算。
        # O3：纯FP16训练，很不稳定，但是可以作为speed的baseline
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)  #

    # multi-gpu training(should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training(should be after apex fp16 initialization)  这一块不是很懂
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True
        )

    # training start
    logger.info("****** Running training ******")
    logger.info(" Num example = %d ", len(train_dataloader) * args.per_gpu_train_batch_size)
    logger.info(" Num Epoch = %d", args.epochs)
    logger.info(" Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)

    logger.info(
        " Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.per_gpu_train_batch_size * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    )

    logger.info(" Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info(" Total optimizer steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.output_dir):
        try:
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (length // args.gradient_accumulation_steps)

            logger.info(" continue training from checkpoint, will skip to save global_steps")
            logger.info(" Continue training from epoch %d", epochs_trained)
            logger.info(" Continue training from global steps %d", global_step)
            logger.info(" Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info(" Starting fine-tuning")

    tr_loss, logging_loss = 0.0, 0.0  # 全局的损失和日志的损失
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.epochs), desc="Epochs", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    best_valid_sum_loss = float("inf")
    best_valid_loss = float("inf")
    # 在训练的时候同时加载两个dataloader
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iterator", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps id resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch = -1
                continue

            model.train()

            source_ids = batch["input_ids"].long.to(args.device)
            attention_mask = batch["attention_mask"].long.to(args.device)
            decoder_input_ids = batch["target_ids"][:, :-1].long().to(args.device)
            summary_labels = batch["target_ids"][:, :-1].contiguous().long().to(args.device)
            segment_ids = batch["segment_ids"].long().to(args.device)
            start_positions = batch["start_positions"].long().to(args.device)
            end_positions = batch["end_positions"].long().to(args.device)
            sentence_start_positions = batch["sentence_start_positions"].long().to(args.device)
            sentence_end_positions = batch["sentence_end_positions"].long().to(args.device)

            inputs = {
                "input_ids": source_ids,
                "attention_mask": attention_mask,
                "segment_ids": segment_ids,
                "decoder_input_ids": decoder_input_ids,
                "summary_labels": summary_labels,
                "start_positions": start_positions,
                "end_positions": end_positions,
                "sentence_start_positions": sentence_start_positions,
                "sentence_end_positions": sentence_end_positions
            }

            outputs = model(**inputs)

            loss, qa_loss, sum_loss = outputs["qa_loss"], outputs["summary_loss"], outputs["loss"]
            if args.n_gpu > 1:
                loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(
                "epoch:{}, global_step:{}, qa_loss:{}, sum_loss:{}, loss:{}".format(
                    epoch, global_step, qa_loss, sum_loss, loss)
            )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # Log Metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps,
                                         global_step)  # 日志步数之内的平均损失
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Take care of distribution/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)

                    valid_loss, valid_sum_loss, valid_qa_loss = validation(valid_dataloader, model, args)

                    # 保存在摘要数据上表现好的模型
                    if valid_sum_loss < best_valid_sum_loss:
                        best_valid_sum_loss = valid_sum_loss
                        logger.info('Saving best valid summary loss model')
                        best_sum_loss_dir = os.path.join(args.output_dir, "best_sum")
                        if not os.path.exists(best_sum_loss_dir):
                            os.makedirs(best_sum_loss_dir)
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(best_sum_loss_dir)

                    # 保存在总体损失上的最小值
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        logger.info('Saving best valid loss model')
                        best_valid_loss_dir = os.path.join(args.output_dir, "best_loss")
                        if not os.path.exists(best_valid_loss_dir):
                            os.makedirs(best_valid_loss_dir)

                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrianed(best_valid_loss_dir)

            if (args.total_num_update > 0) and (global_step > args.total_num_update):
                epoch_iterator.close()
                break

        if (args.total_num_update > 0) and (global_step > args.total_num_update):
            epoch_iterator.close()
            break

    tb_writer.close()
    logger.info(
        "training has done!"
    )


# valid summary loss, not compute rouge or em ir f1
def validation(validation_dataloader, model, args):
    model.eval()
    epoch_valid_dataloader = tqdm(validation_dataloader, desc="Summary Valid dataloader")

    valid_loss = 0
    valid_qa_loss = 0
    valid_sum_loss = 0
    global_steps = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(epoch_valid_dataloader)):
            global_steps += 1

            source_ids = batch["input_ids"].long.to(args.device)
            attention_mask = batch["attention_mask"].long.to(args.device)
            decoder_input_ids = batch["target_ids"][:, :-1].long().to(args.device)
            summary_labels = batch["target_ids"][:, :-1].contiguous().long().to(args.device)
            segment_ids = batch["segment_ids"].long().to(args.device)
            start_positions = batch["start_positions"].long().to(args.device)
            end_positions = batch["end_positions"].long().to(args.device)
            sentence_start_positions = batch["sentence_start_positions"].long().to(args.device)
            sentence_end_positions = batch["sentence_end_positions"].long().to(args.device)

            inputs = {
                "input_ids": source_ids,
                "attention_mask": attention_mask,
                "segment_ids": segment_ids,
                "decoder_input_ids": decoder_input_ids,
                "summary_labels": summary_labels,
                "start_positions": start_positions,
                "end_positions": end_positions,
                "sentence_start_positions": sentence_start_positions,
                "sentence_end_positions": sentence_end_positions
            }

            outputs = model(**inputs)
            valid_qa_loss += outputs["qa_loss"]
            valid_sum_loss += outputs["summary_loss"]
            valid_loss += outputs["loss"]

        valid_loss = valid_loss / global_steps
        valid_sum_loss = valid_sum_loss / global_steps
        valid_qa_loss = valid_qa_loss / global_steps
        logger.info(
            "Evaluate on valid summary squad dataset,average loss:{}".format(valid_loss)
        )
        logger.info('Valid loss:{}, valid_sum_loss: {}, valid_qa_loss:{}'.format(
            valid_loss, valid_sum_loss, valid_qa_loss))

        return valid_loss, valid_sum_loss, valid_qa_loss
