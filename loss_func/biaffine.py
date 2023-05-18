from torch import nn


class Biaffine(nn.Module):
    """
    关系抽取
    """

    def __init__(self, in_size, out_size):
        super(Biaffine, self).__init__()
        self.w1 = torch.nn.Parameter(nn.init.xavier_uniform_(torch.ones((in_size, out_size, in_size))),
                                     requires_grad=True)
        self.w2 = torch.nn.Parameter(nn.init.xavier_uniform_(torch.ones((2 * in_size + 1, out_size))),
                                     requires_grad=True)

    def forward(self, input1, input2, seq_len):
        f1 = input1.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, L, L, 128+128]
        f2 = input1.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, L, L, 128+128]
        concat_f1f2 = torch.cat((f1, f2), axis=-1)  # [B, L, L, 256*2]
        concat_f1f2 = torch.cat((concat_f1f2, torch.ones_like(concat_f1f2[..., :1])), axis=-1)  # [B, L, L, 256*2+1]

        # bxi,oij,byj->boxy
        logits_1 = torch.einsum('bxi,ioj,byj->bxyo', input1, self.w1, input2)
        logits_2 = torch.einsum('bijy,yo->bijo', concat_f1f2, self.w2)

        return logits_1 + logits_2


class biaffine(nn.Module):
    """
    实体抽取：https://zhuanlan.zhihu.com/p/369851456
    """

    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x), out_size, in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        # U.shape = [in_size,out_size,in_size]

    def forward(self, x, y):
        # x,y 分别表示头和尾token
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping
