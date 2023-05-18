from transformers import BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("../local_models/bart-base")

# 获取模型参数量
total_trainable_params = sum(p.numel() for p in model_cg.parameters() if p.requires_grad)
print(total_trainable_params)

# 获取模型的参数及其名称
for name, params in model.named_parameters():
    print("name:", name, "params:", params)

# 打印模型数值及梯度值
for param in model.parameters():
    print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))
