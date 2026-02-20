import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):  # 1. Positional Encoding
    def __init__(self, d_model, max_len=300):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x, start_pos=0):
        x = x + self.pe[:, start_pos:start_pos + x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.dk = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mem=None, attn_mask=None, key_padding_mask=None):
        mem = x if mem is None else mem
        B, T, D = x.shape
        Q = self.q_linear(x).view(B, T, self.num_heads, self.dk).transpose(1, 2)
        K = self.k_linear(mem).view(B, -1, self.num_heads, self.dk).transpose(1, 2)
        V = self.v_linear(mem).view(B, -1, self.num_heads, self.dk).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask==0, float('-inf'))

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask==0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context  = context.transpose(1, 2).contiguous() .view(B, -1, self.num_heads * self.dk)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(FeedForward, self).__init__()
        self.cnn1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.cnn2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.act(self.cnn1(x.permute(0, 2, 1)))
        x = self.cnn2(x)
        x = x.permute(0, 2, 1)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model=768, num_heads=8, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.mhsa = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x, mask=None, key_padding_mask=None):
        x2 = self.mhsa(x, attn_mask=mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + x2)
        x2 = self.ffn(x)
        x = self.norm2(x + x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=768, num_heads=8, d_ff=2048):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.norm3 = nn.BatchNorm1d(d_model)

    def forward(self, x, enc_out, tgt_mask=None, memory_key_padding_mask=None):
        x2 = self.self_attn(x, attn_mask=tgt_mask)
        x = self.norm1(x + x2)
        x2 = self.cross_attn(x, mem=enc_out, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + x2)
        x2 = self.ffn(x)
        x = self.norm3(x + x2)
        return x


class Chomp1d(nn.Module):
    """remove padding on the right to ensure causality"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class CausalConv1d(nn.Module):
    def __init__(self, d_model=768, kernel_size=3, stride=3, dilation=1, padding=0):
        super(CausalConv1d, self).__init__()
        self.conv1 = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=1)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, 
            self.conv2, self.chomp2, self.relu2
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        return self.relu(out + x)

class DilatedCausalCNNMapper(nn.Module):
    def __init__(self, d_model=768, num_layers=3, kernel_size=3):
        super(DilatedCausalCNNMapper, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                CausalConv1d(d_model=d_model, kernel_size=kernel_size, stride=1,
                             dilation=dilation, padding=(kernel_size-1)*dilation))

        self.causal_layers = nn.Sequential(*layers)

    def forward(self, x, target_seq_len):
        x = x.transpose(1, 2)
        x = self.causal_layers(x)
        x = x[:, :, -target_seq_len:]
        x = x.transpose(1, 2)
        return x



class TeacherEncoder(nn.Module):
    def __init__(self, channels=6, d_model=768, n_heads=8, num_layers=1, d_ff=2048):
        super(TeacherEncoder, self).__init__()
        self.input_fc1 = nn.Linear(channels, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        x = self.pos_enc(self.input_fc1(x), start_pos=0)
        for layer in self.layers:
            x = layer(x)
        return x


class SharedDecoder1(nn.Module):
    def __init__(self, pred_horizon=12, seq_len=24, d_model=768, avg_pool=True):
        super(SharedDecoder1, self).__init__()
        self.pred_horizon = pred_horizon
        self.avg_pool = avg_pool
        self.linear1 = nn.Linear(d_model, 256)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 32)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(32, 1)

    def forward(self, mem):
        x = mem[:,-1,:]
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x.squeeze(-1)


class Transformer_baseline_model(nn.Module):
    def __init__(self, map_dim=8, seq_len=24, pred_horizon=12, d_model=64, num_heads=4,
                 num_enc_layer=2, d_ff=128):
        super(Transformer_baseline_model, self).__init__()
        self.encoder_layers = TeacherEncoder(channels=map_dim, d_model=d_model, n_heads=num_heads, num_layers=num_enc_layer, d_ff=d_ff)
        self.pred_horizon = pred_horizon
        self.decoder = SharedDecoder1(pred_horizon=pred_horizon, seq_len=seq_len, d_model=d_model)

    def forward(self, x):
        enc_out = self.encoder_layers(x)
        out = self.decoder(enc_out)
        return out.squeeze()


class Transformer_seq2seq_model(nn.Module):
    def __init__(self, channels=8, pred_horizon=12, d_model=64, num_heads=4,
                 num_enc_layer=2, d_ff=128):
        super(Transformer_seq2seq_model, self).__init__()
        # self.embedding =
        self.encoder_layers = TeacherEncoder(channels=channels, d_model=d_model, n_heads=num_heads, num_layers=num_enc_layer, d_ff=d_ff)
        self.pred_horizon = pred_horizon
        self.decoder_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(d_model, 256),
                                     nn.Linear(256, 32),
                                     nn.Linear(32, pred_horizon))

    def forward(self, x):
        enc_out = self.encoder_layers(x)
        x = enc_out.mean(dim=1)
        out = self.decoder(x)
        return out.squeeze()



class Transformer_seq2seq_Teacher_model(nn.Module):
    def __init__(self, channels=8, pred_horizon=12, d_model=64, num_heads=4,
                 num_enc_layer=2, d_ff=128):
        super(Transformer_seq2seq_Teacher_model, self).__init__()
        # self.embedding =
        self.encoder1 = TeacherEncoder(channels=channels, d_model=d_model, n_heads=num_heads, num_layers=num_enc_layer, d_ff=d_ff)
        self.encoder2 = TeacherEncoder(channels=channels-1, d_model=d_model, n_heads=num_heads, num_layers=num_enc_layer,
                                       d_ff=d_ff)
        self.fusion = ConcatCausalSelfAttention(d_model=d_model, n_head=num_heads)
        self.fusion = FutureSelfAttention(d_model=d_model, n_head=num_heads, pred_horizon=pred_horizon)
        self.pred_horizon = pred_horizon
        self.decoder = nn.Sequential(nn.Linear(d_model, 256),
                                     nn.Linear(256, 32),
                                     nn.Linear(32, 1))

    def forward(self, x):
        x1 = x[:, :-self.pred_horizon]
        x2 = x[:, -self.pred_horizon:, 1:]
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x = self.fusion(x1, x2)
        out = self.decoder(x)
        return out.squeeze()


class Transformer_seq2seq_Encoder_model(nn.Module):
    def __init__(self, channels=8, pred_horizon=12, d_model=64, num_heads=4,
                 num_enc_layer=2, d_ff=128):
        super(Transformer_seq2seq_Encoder_model, self).__init__()
        # self.embedding =
        self.encoder1 = TeacherEncoder(channels=channels, d_model=d_model, n_heads=num_heads, num_layers=num_enc_layer, d_ff=d_ff)
        self.encoder2 = TeacherEncoder(channels=channels-1, d_model=d_model, n_heads=num_heads, num_layers=num_enc_layer,
                                       d_ff=d_ff)
        self.map = nn.Sequential(nn.Linear(d_model*2, d_model))
        self.pred_horizon = pred_horizon
        self.fusion = FutureSelfAttention(d_model=d_model, n_head=num_heads, pred_horizon=pred_horizon)

    def forward(self, x):
        x1 = x[:, :-self.pred_horizon]
        x2 = x[:, -self.pred_horizon:, 1:]
        enc1 = self.encoder1(x1)
        enc2 = self.encoder2(x2)
        out = self.fusion(enc1, enc2)
        return out.squeeze()


class Transformer_seq2seq_Decoder_model(nn.Module):
    def __init__(self, pred_horizon=12, d_model=64):
        super(Transformer_seq2seq_Decoder_model, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(d_model, 256),
                                     nn.Linear(256, 32),
                                     nn.Linear(32, 1)) #pred_horizon

    def forward(self, x):
        out = self.decoder(x)
        return out.squeeze()


class Transformer_seq2seq_Student_Encoder_model(nn.Module):
    def __init__(self, channels=8, seq_len=24, pred_horizon=12, d_model=64, num_heads=4,
                 num_enc_layer=2, d_ff=128):
        super(Transformer_seq2seq_Student_Encoder_model, self).__init__()
        self.encoder = TeacherEncoder(channels=channels, d_model=d_model, n_heads=num_heads, num_layers=num_enc_layer, d_ff=d_ff)
        self.map = nn.Sequential(nn.Linear(seq_len, d_ff), nn.Linear(d_ff, pred_horizon))

    def forward(self, x):
        enc_out = self.encoder(x)
        out = self.map(enc_out.transpose(1, 2)).transpose(1, 2)
        return out.squeeze()


