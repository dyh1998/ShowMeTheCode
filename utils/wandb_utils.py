import wandb
from time import sleep
from transformers import BertModel

"""
1、wandb官网注册账号
2、找到setting里面的api_key
3、在本地安装好wandb的环境中使用命令`wandb login api_key`进行登录
4、创建一个项目
5、使用以下的代码进行
"""
model = BertModel.from_pretrained('../shared/local_models/bert-base-uncased')

wandb.init(project='test-project')
wandb.watch(model)
config = wandb.config
config.lr = 0.01  # 声明超参数必须在wandb.init后面


def my_train_loop():
    loss = 0
    for epoch in range(10):
        sleep(10)
        loss += 0.1  # change as appropriate :)
        wandb.log({'epoch': epoch, 'loss': loss})


my_train_loop()
