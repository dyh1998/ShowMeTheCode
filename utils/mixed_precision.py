"""
混合精度的训练代码实践
cites:https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/124232309
"""
# 第一种实践的方式
from apex import amp

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
"""
其中 opt_level 为精度的优化设置，O0（第一个字母是大写字母O）：
O0：纯FP32训练，可以作为accuracy的baseline；
O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算。
O3：纯FP16训练，很不稳定，但是可以作为speed的baseline
"""


# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion_train(output, targets)
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()


# 第二种实践的方式
import torch

# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()
for data, label in data_iter:
    optimizer.zero_grad()
    # Casts operations to mixed precision
    with torch.cuda.amp.autocast():
        loss = model(data)
    # Scales the loss, and calls backward()
    # to create scaled gradients
    scaler.scale(loss).backward()

    # Unscales gradients and calls
    # or skips optimizer.step()
    scaler.step(optimizer)
    # Updates the scale for next iteration
    scaler.update()
