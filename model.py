import torch
import torch.nn as nn
import torch.nn.functional as F


class Atten_Conv(nn.Module):
    def __init__(self,n,d,m,h,l,need_g=False):
        super(Atten_Conv, self).__init__()
        self.cnn=nn.Conv1d(in_channels=int(d/h),out_channels=m,kernel_size=n,padding='same')
        self.n=n
        self.h=h
        self.d=d
        self.m=m
        self.l=l
        self.need_g=need_g
        if need_g:
            self.maxpool = nn.MaxPool1d(kernel_size=l)
              # nd*1

    def forward(self,Q):#batch  h  *  d/h  *  L
        M = self.cnn(Q)
        F = self.cnn.weight.permute(1, 2, 0)
        F = F.reshape(-1, F.shape[2])
        O = torch.matmul(F, M)#batch h * nd/h *l
        O = O.permute(0, 2, 1)#batch h   *   l   *nd/h
        G=None
        if self.need_g:
            M_max = self.maxpool(M)
            g = torch.matmul(F, M_max)
            g = g.permute(0, 2, 1)
            G = transpose_output(g, self.h)
        return O,G


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        output = self.relu(self.linear1(inputs))
        output = self.linear2(output)
        return output

class Multihead(nn.Module):
    def __init__(self, n, d, m, h,l,num_of_hid,need_g=False):
        super(Multihead, self).__init__()
        self.W_q = nn.Linear(d,num_of_hid)
        self.W_o = nn.Linear(num_of_hid * n, d)
        self.Atten_conv=Atten_Conv(n, num_of_hid, m, h,l,need_g)
        self.n=n
        self.h=h
        self.d=d
        self.m=m
        self.l=l
        self.num_of_hid=num_of_hid
        self.need_g=need_g

    def forward(self, Q):  #batch,l,d
        Q = transpose_qkv((self.W_q(Q)), self.h)
        Q = Q.permute(0, 2, 1)
        O,G=self.Atten_conv(Q)
        if self.need_g:
            G=self.W_o(G)
        output_concat = transpose_output(O, self.h)
        re = self.W_o(output_concat)
        return re,G




class Act_block(nn.Module):
    def __init__(self, n, d, m, h,l,d_ff,num_of_hid,dropout=0.05,need_g=False):
        super(Act_block, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.drop1=nn.Dropout(dropout)
        self.drop2=nn.Dropout(dropout)
        self.Multi=Multihead(n, d, m, h,l,num_of_hid=num_of_hid,need_g=need_g)
        self.n=n
        self.h=h
        self.d=d
        self.m=m
        self.l = l
        self.need_g = need_g
        self.num_of_hid=num_of_hid
        self.d_ff=d_ff
        self.ffn = PositionWiseFeedForwardNetwork(d, d_ff)


    def forward(self,Q):#batch,l,d
        # print(Q.shape)
        out,G=self.Multi(Q)
        out=self.drop1(out)
        out=self.ln1(out+Q)
        out=self.ffn(out)
        out=self.drop2(out)
        out=self.ln2(out+Q)
        return out,G




class Act_net(nn.Module):
    def __init__(self, l, d, m, h,cnn_kernel_size,lstm_num_layers,lstm_dropout,n_list,
                 act_dropout,lstm_hidden_dim, l_dropout,repeat_num,num_of_hid,d_ff):
        super(Act_net, self).__init__()
        self.ln = nn.LayerNorm(d)
        self.num_layer=len(n_list)*repeat_num
        self.n_list=n_list
        self.h=h
        self.d=d
        self.m=m
        self.l = l
        self.blks = nn.Sequential()
        self.num_of_hid=num_of_hid
        self.d_ff=d_ff
        self.textCNN = nn.Conv1d(in_channels=d,
                                 out_channels=d,
                                 kernel_size=cnn_kernel_size,
                                 padding='same')
        self.layerNorm = nn.LayerNorm(d)
        self.biLSTM = nn.LSTM(d,
                              lstm_hidden_dim,
                              bidirectional=True,
                              batch_first=True,
                              num_layers=lstm_num_layers,
                              dropout=lstm_dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.generator = nn.Sequential(nn.Linear(lstm_hidden_dim * 2, 2),
                                       nn.Dropout(l_dropout),
                                       nn.Softmax(dim=-1))
        for i in range(len(n_list)):
            for k in range(repeat_num):
                self.blks.add_module("block"+str(i)+str(k),Act_block(n_list[i], d, m, h,l,d_ff=self.d_ff,dropout=act_dropout,num_of_hid=self.num_of_hid,need_g=((i+1)*(k+1)==self.num_layer)))

    def forward(self, X):
        G=None
        X=X.permute(0,2,1)
        X=X+F.relu(self.textCNN(X))
        X = X.permute(0, 2, 1)
        residual=X
        for blk in (self.blks):
            X,G= blk(X)
        G=G.permute(0,2,1)
        atten = torch.bmm(X, G)
        X=self.layerNorm(X)+residual
        X, _ = self.biLSTM(X)
        X = X.permute(0, 2, 1)
        X = self.generator(self.pool(X).squeeze(-1))
        return X,atten

