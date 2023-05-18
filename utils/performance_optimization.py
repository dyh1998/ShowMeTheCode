"""
该示例中包含梯度累加，半精度混合训练以及梯度裁剪等技术
"""
def __train(self):
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    scaler = GradScaler()
    accum_iter = 4  # accumulate gradient for 4 iterations

    for index, batch in enumerate(tqdm(self.train)):
        inputs = {'input_ids': batch['input_ids'],
                  'attention_mask': batch['attention_mask']
                  }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = batch['label'].to(self.device)
        labels = labels.reshape(-1)
        with autocast():  # 开启自动混合精度上下文
            output = self.model(**inputs)
            logits = output.logits
            loss = self.loss_func(logits, labels)
        loss = loss / accum_iter  # 梯度累加
        # loss.backward()
        scaler.scale(loss).backward()  # 反向传播并放大损失
        if ((index + 1) % accum_iter == 0) or (index + 1 == len(self.train)):  # 梯度累加
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # optimizer.step()
            scaler.step(optimizer)  # 更新权重并缩小损失
            scaler.update()  # 更新权重并缩小损失
            optimizer.zero_grad()
        # optimizer.step()
        # optimizer.zero_grad()


"""
8bit optimizer:不好用，不如半精度自动混合好用（显存占用比半精度大，精度也不是很好）
Using Int8 Matrix Multiplication
For straight Int8 matrix multiplication with mixed precision decomposition you can use bnb.matmul(...). To enable mixed precision decomposition, use the threshold parameter:

bnb.matmul(..., threshold=6.0)
For instructions how to use LLM.int8() inference layers in your own code, see the TL;DR above or for extended instruction see this blog post.

Using the 8-bit Optimizers
With bitsandbytes 8-bit optimizers can be used by changing a single line of code in your codebase. For NLP models we recommend also to use the StableEmbedding layers (see below) which improves results and helps with stable 8-bit optimization. To get started with 8-bit optimizers, it is sufficient to replace your old optimizer with the 8-bit optimizer in the following way:

import bitsandbytes as bnb

# adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # comment out old optimizer
adam = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # add bnb optimizer
adam = bnb.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995), optim_bits=8) # equivalent


torch.nn.Embedding(...) ->  bnb.nn.StableEmbedding(...) # recommended for NLP models
Note that by default all parameter tensors with less than 4096 elements are kept at 32-bit even if you initialize those parameters with 8-bit optimizers. This is done since such small tensors do not save much memory and often contain highly variable parameters (biases) or parameters that require high precision (batch norm, layer norm). You can change this behavior like so:

# parameter tensors with less than 16384 values are optimized in 32-bit
# it is recommended to use multiplies of 4096
adam = bnb.optim.Adam8bit(model.parameters(), min_8bit_size=16384)
"""