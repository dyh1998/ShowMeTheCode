import torch
from torch import optim

"""
分层学习率设计
"""
model = Bert_Blend_CNN().to(device)
bert_params = list(model.bert.named_parameters())
cnn_params = list(model.textcnn.named_parameters())
optimizer_grouped_parameters = [{'params': [p for n, p in bert_params]},
                                {'params': [p for n, p in cnn_params], 'lr': args.lr2}]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr1)

"""
学习率介绍
"""

# 学习率衰减设置
# weight_decay表示权重衰减，目的是防止过拟合，在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，
# 正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。

# eps用于数值稳定，防止出现下溢或者上溢的情况，一般是一个较小的数，默认为1e-8

# betas = （beta1，beta2）
# beta1：一阶矩估计的指数衰减率（如0.9）。
# beta2：二阶矩估计的指数衰减率（如0.999）。该超参数在稀疏梯度（如在NLP或计算机视觉任务中）中应该设置为接近1的数。

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, eps=1e-6)

"""
学习率预热
"""
from transformers import AdanW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=lr, eps=adam_epsilon)
len_dataset = 3821  # 可以根据pytorch中的len(Dataset)计算
epoch = 30
batch_size = 32
# total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size = 0 else (len_dataset // batch_size + 1) * epoch  # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch

warm_up_ratio = 0.1  # 定义要预热的step，即在训练进程的warm_up_ratio*total位置预热到预设的lr，这里是在第382步线性从0预热到预设的lr，然后再逐渐回落到0
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps,
                                            num_training_steps=total_steps)  # linear是线性增加的学习率，

optimizer.step()
scheduler.step()
optimizer.zero_grad()

"""
学习率预热函数
"""


def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """

    scheduler = scheduler.lower()
    if scheduler == 'constantlr':
        return transformers.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=t_total)
    elif scheduler == 'warmupcosine':
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                               num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))


if __name__ == '__main__':
    """
    example:
    """


    def train(trainset, evalset, model, tokenizer, model_dir, lr, epochs, device):
        optimizer = AdamW(model.parameters(), lr=lr)
        batch_size = 3
        # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch
        total_steps = (len(trainset)) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=total_steps)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        lr_record = []
        for epoch in tqdm(range(epochs), desc="epoch"):
            train_loss, steps = 0, 0
            for batch in tqdm(trainset, desc="train"):
                batch = tuple(
                    input_tensor.to(device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
                input_ids, label, mc_ids = batch
                steps += 1
                model.train()
                loss, logits = model(input_ids=input_ids, mc_token_ids=mc_ids, labels=label)
                # loss.backward()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                lr_record.append(scheduler.get_lr()[0])
                if steps % 500 == 0:
                    print("step:%d  avg_loss:%.3f" % (steps, train_loss / steps))
            plot(lr_record)
            eval_res = evaluate(evalset, model, device)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "gpt2clsnews.model%d.ckpt" % epoch)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(os.path.join(model_dir, "gpt2clsnews.tokinizer"))
            logging.info("checkpoint saved in %s" % model_dir)
