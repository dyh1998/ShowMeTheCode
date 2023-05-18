"""
对抗训练：目的是提高模型的泛化能力，提高模型的鲁棒性
对抗训练是指通过添加扰动构造一些对抗样本，提高模型在遇到对抗样本时候的鲁棒性，
同时一定程度上也能提高模型的表现和泛化能力，对抗样本相比原始样本添加的扰动是微小的且可以使模型犯错
1、扰动
    给样本添加扰动：x+r_adv
    损失函数为min(-logP(y|x+r_adv;theta))
    由于对抗样本是要让样本尽可能出错，那么就是要让损失越大越好，但是也不能太大，要在合理的范围内，r_adv<=|r|
    loss减小的方法是梯度下降，loss增大自然就是梯度上升
    那么r_adv可表示为r_adv = -log(x,y,theta)，于是将扰动定义为c*sign(-log(x,y,theta))，其中c表示一个常数，sign表示符号函数
    有了这个公式后就可对其进行优化
2、Min-Max公式
    改方法将公式两个部分，一个是内部损失函数的最大化，一个是外部经验风险的最小化。
    内部的max是为了找到worst-case的扰动，也就是攻击，其中，L为损失函数，S为扰动的范围空间
    外部min是为了基于改攻击找到最鲁棒的模型参数
3、FGM（Fast Gradient Method）
    假设文本序列的序列为embedding，那么embedding扰动为r_adv = c*g/|g|,其中更g = -log(x,y,theta)
"""
import torch


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 对梯度求p范数，默认为2范数
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


if __name__ == '__main__':

    fgm = FGM(model)
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward()  # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack()  # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
