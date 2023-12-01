import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as debug
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from configs_stock import *

device = torch.device("cuda")
class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        # print("input_size", input_size)
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, hidden_states, reverse=False):

        b, seq, embed = inputs.size()
        h = hidden_states[0]
        c = hidden_states[1]

        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []

        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))  # short term mem
            # discounted short term mem
            c_s2 = c_s1 * timestamps[:, s: s + 1].expand_as(c_s1)
            c_l = c - c_s1  # long term mem
            c_adj = c_l + c_s2  # adjusted = long + disc short term mem
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(o)
            hidden_state_c.append(c)
            hidden_state_h.append(h)

        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()

        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        return outputs, (h, c)

class attn(torch.nn.Module):
    def __init__(self, in_shape, use_attention=True, maxlen=None):
        super(attn, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.W1 = torch.nn.Linear(in_shape, in_shape).to(device)
            self.W2 = torch.nn.Linear(in_shape, in_shape).to(device)
            self.V = torch.nn.Linear(in_shape, 1).to(device)
        if maxlen != None:
            self.arange = torch.arange(maxlen).to(device)

    def forward(self, full, last, lens=None, dim=1):
        """
        full : B*30*in_shape
        last : B*1*in_shape
        lens: B*1
        """
        if self.use_attention: #これが実行される
            score = self.V(F.tanh(self.W1(last) + self.W2(full)))
            # print(full.shape, last.shape)
            # print(score.shape) #-> B*30*1 この形にならない 30を今回のinputデータに合う形に変更必要　config的な感じで変更
            # print(self.W1(last).shape, self.W2(full).shape)

            if lens != None:
                mask = self.arange[None, :] < lens[:, None]  # B*30
                # print(score.shape, mask.shape)
                # print(score)
                score[~mask] = float("-inf")

            attention_weights = F.softmax(score, dim=dim)
            context_vector = attention_weights * full
            context_vector = torch.sum(context_vector, dim=dim)
            return context_vector  # B*in_shape
        else:
            if lens != None:
                mask = self.arange[None, :] < lens[:, None]  # B*30
                mask = mask.type(torch.float).unsqueeze(-1).cuda()
                context_vector = full * mask
                context_vector = torch.mean(context_vector, dim=dim)
                return context_vector
            else:
                return torch.mean(full, dim=dim)
            


class attn_x(nn.Module): #bardそのまま
    def __init__(self, input_size, output_size):
        super().__init__()
        self.q = nn.Linear(input_size, output_size)
        self.k = nn.Linear(input_size, output_size)
        self.v = nn.Linear(input_size, output_size)

    def forward(self, x1, x2):
        """
        Attention層のforwardメソッド
        Args:
          x1: 特徴量x1
          x2: 特徴量x2
        Returns:
          attn_out: Attention層の出力
        """
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        # スコア計算
        score = torch.matmul(q, k.transpose(-1, -2))
        # ソフトマックス関数で正規化
        attention_weight = F.softmax(score, dim=-1)
        # コンテキストベクトル計算
        attn_out = torch.matmul(attention_weight, v)
        # print("past attn_out.shape", attn_out.shape)
        # 出力サイズを変更
        attn_out = attn_out.view(1, -1)
        # print("attn_out.shape", attn_out.shape)
        return attn_out
    
    

class Actor(nn.Module):
    """
    Actor:
        Gets the text: news/tweets about the stocks,
        current balance, price and holds on the stocks.
    """

    def __init__(
        self,
        num_stocks=STOCK_DIM,
        text_embed_dim=TWEETS_EMB,
        intraday_hiddenDim=128,
        interday_hiddenDim=128,
        intraday_numLayers=1,
        interday_numLayers=1,
        use_attn1=False,
        use_attn2=False,
        # maxlen=30, 
        maxlen=MAX_LEN, #修正　B*30をB*2に
        device=torch.device("cuda"),
    ):
        """
        num_stocks: number of stocks for which the agent is trading
        """
        super(Actor, self).__init__()

        self.lstm1s = [
            TimeLSTM(
                input_size=text_embed_dim,
                hidden_size=intraday_hiddenDim,
            )
            for _ in range(num_stocks)
        ]

        for i, tweet_lstm in enumerate(self.lstm1s):
            self.add_module("lstm1_{}".format(i), tweet_lstm)

        self.lstm1_outshape = intraday_hiddenDim
        self.lstm2_outshape = interday_hiddenDim

        self.attn1s = [
            attn(self.lstm1_outshape, maxlen=maxlen)
            for _ in range(num_stocks)
        ]

        for i, tweet_attn in enumerate(self.attn1s):
            self.add_module("attn1_{}".format(i), tweet_attn)

        self.lstm2s = [
            nn.LSTM(
                input_size=self.lstm1_outshape,
                hidden_size=interday_hiddenDim,
                num_layers=interday_numLayers,
                batch_first=True,
                bidirectional=False,
            )
            for _ in range(num_stocks)
        ]
        
        self.sec_lstm2s = [
            nn.LSTM(
                input_size=TWEETS_EMB,
                hidden_size=interday_hiddenDim,
                num_layers=interday_numLayers,
                batch_first=True,
                bidirectional=False,
            )
            for _ in range(num_stocks)
        ]
        
        for i, day_lstm in enumerate(self.lstm2s):
            self.add_module("lstm2_{}".format(i), day_lstm)
            
        for i, day_lstm in enumerate(self.sec_lstm2s): #これによってsec_lstm2sのパラメータがcpuからcudaに置かれる　これがないとエラー　おまじないのようなものかと
            self.add_module("sec_lstm2_{}".format(i), day_lstm)

        self.attn2s = [
            attn(self.lstm2_outshape)
            for _ in range(num_stocks)
        ]

        for i, day_attn in enumerate(self.attn2s):
            self.add_module("attn2_{}".format(i), day_attn)


        self.sec_attn2s = [
            attn(self.lstm2_outshape)
            for _ in range(num_stocks)
        ]

        for i, day_attn in enumerate(self.sec_attn2s):
            self.add_module("sec_attn2_{}".format(i), day_attn)


        self.linearx1 = [
            nn.Linear(self.lstm2_outshape, self.lstm2_outshape)
            for _ in range(num_stocks)
        ]
        for i, linear_x in enumerate(self.linearx1):
            self.add_module("linearx1_{}".format(i), linear_x)
            
    
        self.sec_linearx1 = [
        nn.Linear(self.lstm2_outshape, self.lstm2_outshape)
        for _ in range(num_stocks)
        ]
        for i, linear_x in enumerate(self.sec_linearx1):
            self.add_module("sec_linearx1_{}".format(i), linear_x)
            

        self.linearx2 = [nn.Linear(self.lstm2_outshape, 64)
                         for _ in range(num_stocks)]
        for i, linear_x in enumerate(self.linearx2):
            self.add_module("linearx2_{}".format(i), linear_x)
            
            
        self.sec_linearx2 = [nn.Linear(self.lstm2_outshape, 64)
                         for _ in range(num_stocks)]
        for i, linear_x in enumerate(self.sec_linearx2):
            self.add_module("sec_linearx2_{}".format(i), linear_x)
            

        self.drop = nn.Dropout(p=0.3)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.device = device
        self.maxlen = maxlen

        self.linear1 = nn.Linear(2 * num_stocks + 1, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear_c = nn.Linear(64 * num_stocks + 32, num_stocks)
        self.tanh = nn.Tanh()
        self.num_stocks = num_stocks
        self.device = device
        
        self.attn_x = attn_x(64, 64)

    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.lstm1_outshape)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.lstm2_outshape)).to(self.device)

        return (h, c)

    def forward(self, state):
        state = state.view(-1, FEAT_DIMS)
        # print("state.shape", state.shape)
        stock_feats = state[:, 0: 2 * self.num_stocks + 1].view(
            -1, 2 * self.num_stocks + 1
        )
        sentence_feat = state[:, EMB_IDX:LEN_IDX].view(
            -1, self.num_stocks, N_DAYS, MAX_TWEETS, TWEETS_EMB
        )
        len_tweets = state[:,
                           LEN_IDX:TARGET_IDX].view(-1, self.num_stocks, N_DAYS)
        time_feats = state[:,
                           TIME_IDX:SECEMB_IDX].view(-1, self.num_stocks, N_DAYS, MAX_TWEETS)
        
        sec_feats = state[:,
                           SECEMB_IDX:].view(-1, self.num_stocks, MAX_SECS, TWEETS_EMB) 
        
        # print("shapes:", stock_feats.shape, sentence_feat.shape, len_tweets.shape, time_feats.shape, sec_feats.shape)

        self.bs = sentence_feat.size(0)
        # print("self.bs:", self.bs)
        sentence_feat = sentence_feat.permute(1, 2, 0, 3, 4)
        len_tweets = len_tweets.permute(1, 2, 0)
        time_feats = time_feats.permute(1, 2, 0, 3)
        sec_feats =sec_feats.permute(1, 0, 2, 3) #num_stock, 1, secの数, embeddingsize
        # print("shapes:", stock_feats.shape, sentence_feat.shape, len_tweets.shape, time_feats.shape, sec_feats.shape)


        num_days = N_DAYS
        text_out = torch.zeros(self.num_stocks, self.bs, 64).to(self.device)
        for i in range(self.num_stocks): #各企業
            h_init, c_init = self.init_hidden()

            lstm1_out = torch.zeros(num_days, self.bs, self.lstm1_outshape).to(
                self.device
            )
            for j in range(num_days): #各日付
                temp_sent = sentence_feat[i, j, :, :, :] #ここで企業と日付決定 ある企業のある日のtweetsそのもの
                temp_len = len_tweets[i, j, :] #ある企業のある日のtweetsの数
                temp_timefeats = time_feats[i, j, :, :] #ある企業のある日のtweets同士の時間差　temp_len temp_timefeats 決算報告書にはいらん
                # print("temp_len, temp_timefeats:", temp_len, temp_timefeats)

                temp_lstmout, (_, _) = self.lstm1s[i]( #TLSTM
                    temp_sent, temp_timefeats, (h_init, c_init)
                )
                
                # print("temp_lstmout:", temp_lstmout)
                # print("temp_lstmout.shape:", temp_lstmout.shape) #torch.Size([1, 30, 128])

                last_idx = temp_len.type(torch.int).tolist()
                temp_hn = torch.zeros(self.bs, self.lstm1_outshape).to(self.device)
                for k in range(self.bs): #self.bs = 1
                    if last_idx[k] != 0:
                        temp_hn[k] = temp_lstmout[k, last_idx[k] - 1, :]
                lstm1_out[j] = self.attn1s[i](temp_lstmout, temp_hn, temp_len.to(self.device)) #Attn 第二引数は最後の単語
                # print("lstm1_out[j].shape", lstm1_out[j].shape)
                # print("lstm1_out[j]", lstm1_out[j])

            # print("lstm1_out.shape ", lstm1_out.shape)#lookback_len, 1, 128
            lstm1_out = lstm1_out.permute(1, 0, 2)#1, lookback_len, 128
            lstm2_out, (h2_out, _) = self.lstm2s[i](lstm1_out) #隠れ層h2outを得るため
            # print("lstm2_out.shape, h2_out.shape", lstm2_out.shape, h2_out.shape)
            h2_out = h2_out.permute(1, 0, 2)
            x1 = self.attn2s[i](lstm2_out, h2_out) #lstm2_outは7日分のツイート30*7を含む　h2outはその隠れ層
            x1 = self.drop(self.relu(self.linearx1[i](x1))) #drop:過学習防止　relu:入力が0以下の時0を出力　
            x1 = self.linearx2[i](x1)
            # print("x1.shape ", x1.shape)
            # text_out[i] = x1 #これが最終的な特徴量p1　これをp10としてp11とattnする
            
            #267-273を模倣　かつtext_out[i]とsecのtextoutにattn
            temp_sec = sec_feats[i, :, :, :]
            # print("temp_sec.shape: ", temp_sec.shape) #1, secの数, 768
            # lstm1_out = lstm1_out.permute(1, 0, 2) #不要
            sec_lstm2_out, (sec_h2_out, _) = self.sec_lstm2s[i](temp_sec.to(self.device)) #secに対してlstm層を入れる意味はあるか？
            # print("sec_lstm2_out.shape, sec_h2_out.shape", sec_lstm2_out.shape, sec_h2_out.shape)
            sec_h2_out = sec_h2_out.permute(1, 0, 2)
            x2 = self.sec_attn2s[i](lstm2_out, h2_out)
            x2 = self.drop(self.relu(self.sec_linearx1[i](x2)))
            x2 = self.sec_linearx2[i](x2)
            # print("x2.shape", x2.shape) #[1, 64]
            
            #x1, x2に対してattn
            x = self.attn_x(x1, x2) #これ正しいか怪しい
            # print("x.shape", x.shape)
            text_out[i] = x
            

        text_out = text_out.permute(1, 0, 2)
        text_out = text_out.view(self.bs, -1)
        x_stock = self.relu(self.linear1(stock_feats))
        x_stock = self.linear2(x_stock)

        full = torch.cat([x_stock, text_out], dim=1)
        full = self.tanh(self.linear_c(full))
        # print("full.shape", full.shape)
        return full

class Critic(nn.Module):
    """
    Actor:
        Gets the text tweets about the stocks,
        current balance, price and holds on the stocks.
    """

    def __init__(
        self,
        num_stocks=STOCK_DIM,
        text_embed_dim=TWEETS_EMB,
        intraday_hiddenDim=128,
        interday_hiddenDim=128,
        intraday_numLayers=1,
        interday_numLayers=1,
        use_attn1=False,
        use_attn2=False,
        # maxlen=30,
        maxlen=MAX_LEN, #修正 
        device=torch.device("cuda"),
    ):
        """
        num_stocks: number of stocks for which the agent is trading
        """
        super(Critic, self).__init__()

        self.lstm1s = [
            TimeLSTM(
                input_size=text_embed_dim,
                hidden_size=intraday_hiddenDim,
            )
            for _ in range(num_stocks)
        ]

        for i, tweet_lstm in enumerate(self.lstm1s):
            self.add_module("lstm1_{}".format(i), tweet_lstm)

        self.lstm1_outshape = intraday_hiddenDim
        self.lstm2_outshape = interday_hiddenDim

        self.attn1s = [
            attn(self.lstm1_outshape, maxlen=maxlen)
            for _ in range(num_stocks)
        ]

        for i, tweet_attn in enumerate(self.attn1s):
            self.add_module("attn1_{}".format(i), tweet_attn)

        self.lstm2s = [
            nn.LSTM(
                input_size=self.lstm1_outshape,
                hidden_size=interday_hiddenDim,
                num_layers=interday_numLayers,
                batch_first=True,
                bidirectional=False,
            )
            for _ in range(num_stocks)
        ]
        for i, day_lstm in enumerate(self.lstm2s):
            self.add_module("lstm2_{}".format(i), day_lstm)
            
            
            
        self.sec_lstm2s = [
            nn.LSTM(
                input_size=TWEETS_EMB,
                hidden_size=interday_hiddenDim,
                num_layers=interday_numLayers,
                batch_first=True,
                bidirectional=False,
            )
            for _ in range(num_stocks)
        ]
            
        for i, day_lstm in enumerate(self.sec_lstm2s): #これによってsec_lstm2sのパラメータがcpuからcudaに置かれる　これがないとエラー　おまじないのようなものかと
            self.add_module("sec_lstm2_{}".format(i), day_lstm)


        self.attn2s = [
            attn(self.lstm2_outshape)
            for _ in range(num_stocks)
        ]
        for i, day_attn in enumerate(self.attn2s):
            self.add_module("attn2_{}".format(i), day_attn)


        self.sec_attn2s = [
            attn(self.lstm2_outshape)
            for _ in range(num_stocks)
        ]
        for i, day_attn in enumerate(self.sec_attn2s):
            self.add_module("sec_attn2_{}".format(i), day_attn)

        self.linearx1 = [
            nn.Linear(self.lstm2_outshape, self.lstm2_outshape)
            for _ in range(num_stocks)
        ]
        for i, linear_x in enumerate(self.linearx1):
            self.add_module("linearx1_{}".format(i), linear_x)

        self.linearx2 = [nn.Linear(self.lstm2_outshape, 64)
                         for _ in range(num_stocks)]
        for i, linear_x in enumerate(self.linearx2):
            self.add_module("linearx2_{}".format(i), linear_x)

        self.drop = nn.Dropout(p=0.3)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.num_stocks = num_stocks
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)

        self.device = device
        self.maxlen = maxlen

        self.linear1 = nn.Linear(2 * num_stocks + 1, 64)
        self.linear2 = nn.Linear(64, 32)
        
        self.sec_linearx1 = [
        nn.Linear(self.lstm2_outshape, self.lstm2_outshape)
        for _ in range(num_stocks)
        ]
        for i, linear_x in enumerate(self.sec_linearx1):
            self.add_module("sec_linearx1_{}".format(i), linear_x)
            
        
        self.sec_linearx2 = [nn.Linear(self.lstm2_outshape, 64)
                         for _ in range(num_stocks)]
        for i, linear_x in enumerate(self.sec_linearx2):
            self.add_module("sec_linearx2_{}".format(i), linear_x)
        

        self.linear_c = nn.Linear(64 * num_stocks + 32, 32)

        # * Critic Layers
        self.linear_critic = nn.Linear(num_stocks, 32)

        # * Actions and States
        self.linear_sa1 = nn.Linear(64, 32)
        self.linear_sa2 = nn.Linear(32, 1)
        self.device = device
        self.attn_x = attn_x(64, 64)
    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.lstm1_outshape)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.lstm2_outshape)).to(self.device)
        

        return (h, c)

    def forward(self, state, actions):
        state = state.view(-1, FEAT_DIMS)
        actions = actions.view(-1, STOCK_DIM)

        stock_feats = state[:, 0: 2 * self.num_stocks + 1].view(
            -1, 2 * self.num_stocks + 1
        )
        sentence_feat = state[:, EMB_IDX:LEN_IDX].view(
            -1, self.num_stocks, N_DAYS, MAX_TWEETS, TWEETS_EMB
        )
        len_tweets = state[:,
                           LEN_IDX:TARGET_IDX].view(-1, self.num_stocks, N_DAYS)
        
        time_feats = state[:,
                           TIME_IDX:SECEMB_IDX].view(-1, self.num_stocks, N_DAYS, MAX_TWEETS)
        
        sec_feats = state[:,
                           SECEMB_IDX:].view(-1, self.num_stocks, MAX_SECS, TWEETS_EMB) 

        self.bs = sentence_feat.size(0)
        sentence_feat = sentence_feat.permute(1, 2, 0, 3, 4)
        len_tweets = len_tweets.permute(1, 2, 0)
        time_feats = time_feats.permute(1, 2, 0, 3)
        sec_feats =sec_feats.permute(1, 0, 2, 3) #num_stock, 1, secの数, embeddingsize
        
        num_days = N_DAYS
        text_out = torch.zeros(self.num_stocks, self.bs, 64).to(self.device)
        for i in range(self.num_stocks):
            h_init, c_init = self.init_hidden()
            lstm1_out = torch.zeros(num_days, self.bs, self.lstm1_outshape).to(
                self.device
            )
            for j in range(num_days):
                temp_sent = sentence_feat[i, j, :, :, :]
                temp_len = len_tweets[i, j, :]
                temp_timefeats = time_feats[i, j, :, :]

                temp_lstmout, (_, _) = self.lstm1s[i](
                    temp_sent, temp_timefeats, (h_init, c_init)
                )
                last_idx = temp_len.type(torch.int).tolist()
                temp_hn = torch.zeros(self.bs, self.lstm1_outshape).to(self.device)
                for k in range(self.bs):
                    if last_idx[k] != 0:
                        temp_hn[k] = temp_lstmout[k, last_idx[k] - 1, :]

                lstm1_out[j] = self.attn1s[i](temp_lstmout, temp_hn, temp_len.to(self.device))

            lstm1_out = lstm1_out.permute(1, 0, 2)
            lstm2_out, (h2_out, _) = self.lstm2s[i](lstm1_out)
            h2_out = h2_out.permute(1, 0, 2)
            x1 = self.attn2s[i](lstm2_out, h2_out)

            x1 = self.drop(self.relu(self.linearx1[i](x1)))
            x1 = self.linearx2[i](x1)
            # text_out[i] = x
            
            #267-273を模倣　かつtext_out[i]とsecのtextoutにattn
            temp_sec = sec_feats[i, :, :, :]
            # print("temp_sec.shape: ", temp_sec.shape) #1, secの数, 768
            # lstm1_out = lstm1_out.permute(1, 0, 2) #不要
            sec_lstm2_out, (sec_h2_out, _) = self.sec_lstm2s[i](temp_sec.to(self.device)) #secに対してlstm層を入れる意味はあるか？
            # print("sec_lstm2_out.shape, sec_h2_out.shape", sec_lstm2_out.shape, sec_h2_out.shape)
            sec_h2_out = sec_h2_out.permute(1, 0, 2)
            x2 = self.sec_attn2s[i](lstm2_out, h2_out)
            x2 = self.drop(self.relu(self.sec_linearx1[i](x2)))
            x2 = self.sec_linearx2[i](x2)
            # print("x2.shape", x2.shape) #[1, 64]
            
            #x1, x2に対してattn
            x = self.attn_x(x1, x2) #これ正しいか怪しい
            # print("x.shape", x.shape)
            text_out[i] = x

        text_out = text_out.permute(1, 0, 2)
        text_out = text_out.view(self.bs, -1)

        x_stock = self.relu(self.linear1(stock_feats))
        x_stock = self.linear2(x_stock)

        full = torch.cat([x_stock, text_out], dim=1)
        full = self.linear_c(full)

        actions = self.linear_critic(actions)

        full = torch.cat([full, actions], dim=1)
        full = self.relu(self.linear_sa1(full))
        full = self.linear_sa2(full)

        return full