import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMSeq2seq(nn.Module):
    def __init__(self, channels=3, hidden_size=200, pred_horizon=12):
        super(LSTMSeq2seq, self).__init__()
        self.hidden_dim = hidden_size
        self.pred_horizon = pred_horizon
        self.encoder_lstm = nn.LSTM(channels, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, 150),
                                     nn.ReLU(),
                                     nn.Linear(150, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, 20),
                                     nn.ReLU(),
                                     nn.Linear(20, 1))


    def forward(self, x):
        enc_out, _ = self.encoder_lstm(x)
        enc_out = enc_out[:, -1, :]
        repreated_h = enc_out.unsqueeze(1).repeat(1, self.pred_horizon, 1)
        dec_out, _ = self.decoder_lstm(repreated_h)
        out = self.decoder(dec_out)
        return out.squeeze()


class BiLSTMSeq2seq(nn.Module):
    def __init__(self, channels=3, hidden_size=200, pred_horizon=12):
        super(BiLSTMSeq2seq, self).__init__()
        self.hidden_dim = hidden_size
        self.pred_horizon = pred_horizon
        self.encoder_lstm = nn.LSTM(channels, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(hidden_size*2, hidden_size, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_size*2, 150),
                                     nn.ReLU(),
                                     nn.Linear(150, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, 20),
                                     nn.ReLU(),
                                     nn.Linear(20, 1))

    def forward(self, x):
        enc_out, _ = self.encoder_lstm(x)
        h_cat = enc_out[:, -1, :]
        repeated_h = h_cat.unsqueeze(1).repeat(1, self.pred_horizon, 1)
        dec_out, _ = self.decoder_lstm(repeated_h)
        out = self.decoder(dec_out)
        return out.squeeze()


class CRNNSeq2seq(nn.Module):
    def __init__(self, channels=3, seq_len =36, hidden_size=200, pred_horizon=12):
        super(CRNNSeq2seq, self).__init__()
        self.hidden_dim = hidden_size
        self.pred_horizon = pred_horizon
        self.encoder = nn.Sequential(nn.Conv1d(channels, 8, kernel_size=3, padding='same'),
                                     nn.MaxPool1d(kernel_size=2, stride=2),
                                     nn.Conv1d(8, 16, kernel_size=3, padding='same'),
                                     nn.MaxPool1d(kernel_size=2, stride=2),
                                     nn.Conv1d(16, 32, kernel_size=3, padding='same'),
                                     nn.MaxPool1d(kernel_size=2, stride=2))
        self.encoder_lstm = nn.LSTM(32, hidden_size, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, 256),
                                     nn.Linear(256, 32),
                                     nn.Linear(32, pred_horizon))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x, _ = self.encoder_lstm(x)
        x = x[:, -1]
        out = self.decoder(x)
        return out.squeeze()



class LSTM_model(nn.Module):
    def __init__(self, channels=3, hidden_size=128):
        super(LSTM_model, self).__init__()
        self.LSTM = nn.LSTM(input_size=channels, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, 150),
                                     nn.ReLU(),
                                     nn.Linear(150, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, 20),
                                     nn.ReLU(),
                                     nn.Linear(20, 1))

    def forward(self, x):
        x, _ = self.LSTM(x)
        x = x[:, -1, :]
        out = self.decoder(x)   # 0.6, 0.9, 0.9, 1.0
        return out.squeeze()


class BiLSTM_model(nn.Module):
    def __init__(self, channels=3, hidden_size=128):
        super(BiLSTM_model, self).__init__()
        self.BiLSTM = nn.LSTM(input_size=channels, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_size*2, 150),
                                     nn.ReLU(),
                                     nn.Linear(150, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, 20),
                                     nn.ReLU(),
                                     nn.Linear(20, 1))

    def forward(self, x):
        x, _ = self.BiLSTM(x)
        x = x[:, -1, :]
        out = self.decoder(x)
        return out.squeeze()


class CRNN_model(nn.Module):
    def __init__(self, channels=3):
        super(CRNN_model, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding='same')
        self.pool1 = nn.MaxPool1d(2)
        self.cnn2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.cnn3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.pool3 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(64, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.cnn1(x))
        x = self.pool2(self.cnn2(x))
        x = self.pool3(self.cnn3(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        out = self.decoder(x)
        return out.squeeze()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=768, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term  = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class Transformer_model(nn.Module):
    def __init__(self, input_dim=3, d_model=512, nhead=4, num_layers=1, dim_forward=2048,
                 dropout=0.1, pooling='last', enable_classification=False,
                 norm_first=True):
        super(Transformer_model, self).__init__()
        assert pooling in ('last', 'mean', 'cls')
        self.pooling = pooling
        self.enable_classifications = enable_classification
        self.input_proj = nn.Linear(input_dim, d_model)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_forward,
                                                   dropout=dropout, batch_first=True,
                                                   norm_first=norm_first, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        if enable_classification:
            self.cls_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        else:
            self.cls_head = None
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0, std=0.02)

    def _pool(self, enc_out):
        if self.pooling == 'last':
            return enc_out[:, -1, :]
        elif self.pooling == 'mean':
            return enc_out.mean(dim=1)
        elif self.pooling == 'cls':
            return enc_out[:, 0, :]

    def forward(self, x, src_key_padding_mask=None):
        B, T, C = x.shape
        x = self.input_proj(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if src_key_padding_mask is not None:
                pad = torch.zeros((B, 1), dtype=torch.bool, device=src_key_padding_mask.device)
                src_key_padding_mask = torch.cat([pad, src_key_padding_mask], dim=1)
        x = self.pos_encoder(x)
        enc_out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        h = self._pool(enc_out)
        y_reg = self.reg_head(h)
        # print(y_reg.shape)
        return y_reg.squeeze()


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (1D time)."""
    def __init__(self, d_model=128, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class FeedForward(nn.Module):
    def __init__(self, d_model=128, d_ff=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        return self.norm(x)


# ---------- DSW (Dimension-Segment-Wise) Embedding ----------
class PatchEmbedding(nn.Module):
    """
    Segment the time axis into non-overlapping patches of length patch_len.
    Project each patch to d_model with a linear layer.
    Input  : (B, T, C) where C=3 (glucose, insulin, meal)
    Output : tokens (B, C, P, d_model) where P = T // patch_len
    Also returns (P) for convenience.
    """
    def __init__(self, C=3, patch_len=12, d_model=256):
        super().__init__()
        self.C = C
        self.patch_len = patch_len
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.C, f"Expected {self.C} variables, got {C}"
        assert T % self.patch_len == 0, "T must be divisible by patch_len (use padding or trim)"
        P = T // self.patch_len
        # reshape to (B, C, P, patch_len)
        x = x.permute(0, 2, 1)                      # (B, C, T)
        x = x.view(B, C, P, self.patch_len)         # (B, C, P, Lp)
        # linear projection per patch (share across variables)
        x = self.proj(x)                            # (B, C, P, d_model)
        return x, P


# ---------- Two-Stage Attention (TSA) block ----------
class TSA(nn.Module):
    """Two-Stage Attention: time-wise attention then dimension-wise attention.
    Operates on tokens shaped (B, C, P, d_model).
    Stage 1 (time): for each variable c, attend across P time tokens.
    Stage 2 (dim):  for each patch p,  attend across C variables.
    """
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_ln1 = LayerNorm(d_model)
        self.time_ff  = FeedForward(d_model, d_ff, dropout)
        self.time_ln2 = LayerNorm(d_model)

        self.dim_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dim_ln1 = LayerNorm(d_model)
        self.dim_ff  = FeedForward(d_model, d_ff, dropout)
        self.dim_ln2 = LayerNorm(d_model)

    def forward(self, x):
        B, C, P, D = x.shape
        # --- Stage 1: cross-time attention (within each variable) ---
        # reshape to (B*C, P, D)
        xt = x.reshape(B*C, P, D)
        attn_out, _ = self.time_attn(xt, xt, xt, need_weights=False)
        xt = self.time_ln1(xt + attn_out)
        xt2 = self.time_ff(xt)
        xt = self.time_ln2(xt + xt2)                # (B*C, P, D)
        x_time = xt.view(B, C, P, D)

        # --- Stage 2: cross-dimension attention (within each time patch) ---
        # transpose axes to group by patch: (B, P, C, D) -> (B*P, C, D)
        xd = x_time.permute(0, 2, 1, 3).reshape(B*P, C, D)
        attn_out2, _ = self.dim_attn(xd, xd, xd, need_weights=False)
        xd = self.dim_ln1(xd + attn_out2)
        xd2 = self.dim_ff(xd)
        xd = self.dim_ln2(xd + xd2)                 # (B*P, C, D)
        x_dim = xd.view(B, P, C, D).permute(0, 2, 1, 3)  # (B, C, P, D)
        return x_dim


# ---------- Crossformer Encoder ----------
class CrossformerEncoder(nn.Module):
    def __init__(self, depth=1, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TSA(d_model, n_heads, d_ff, dropout) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # (B, C, P, D)


# ---------- Projection Head for Forecasting ----------
class ForecastHead(nn.Module):
    """Map encoded tokens -> future sequence for a target variable.
    Strategy:
      1) Aggregate encoded representations over variables via attention pooling conditioned on the target variable index.
      2) Upsample from patch tokens back to full time resolution (pred_len).
    """
    def __init__(self, d_model=128, patch_len=12, pred_len=12, C=3, target_index=0):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.pred_len = pred_len
        self.C = C
        self.target_index = target_index  # 0 for glucose by default
        # Attention pooling over variables (query = target var token)
        self.var_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        # Simple decoder: flatten patch tokens and project to pred_len
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.patch_len),
        )

    def forward(self, x):
        # x: (B, C, P, D)
        B, C, P, D = x.shape
        # pick target variable tokens (B, P, D)
        q = x[:, self.target_index, :, :]  # (B, P, D)
        # keys/values are all variables concatenated along sequence dim
        kv = x.permute(0, 2, 1, 3).reshape(B, P*C, D)  # (B, P*C, D)
        attn_out, _ = self.var_attn(q, kv, kv, need_weights=False)  # (B, P, D)
        h = attn_out  # (B, P, D)

        # Project each patch token to patch_len points
        patch_preds = self.proj(h)  # (B, P, patch_len)

        # Reconstruct full pred_len sequence by concatenating patch_len blocks then cropping/padding
        y = patch_preds.reshape(B, P * self.patch_len)  # (B, P*Lp)
        if y.size(1) < self.pred_len:
            pad = self.pred_len - y.size(1)
            y = F.pad(y, (0, pad))
        elif y.size(1) > self.pred_len:
            y = y[:, :self.pred_len]
        return y  # (B, pred_len)


# ---------- Full Model ----------
class CrossformerGlucose(nn.Module):
    def __init__(
        self,
        seq_len=36,
        pred_len=1,
        d_model=64,
        d_ff=128,
        n_heads=2,
        depth=2,
        patch_len=6,
        dropout=0.1,
        C=3,
        target_index=0,
        add_time_pe=True,
    ):
        super().__init__()
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.C = C

        self.patch_embed = PatchEmbedding(C=C, patch_len=patch_len, d_model=d_model)
        self.time_pe = PositionalEncoding(d_model) if add_time_pe else None
        self.encoder = CrossformerEncoder(depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.head = ForecastHead(d_model=d_model, patch_len=patch_len, pred_len=pred_len, C=C, target_index=target_index)

        # variable embeddings to give each variable an ID
        self.var_emb = nn.Parameter(torch.randn(C, d_model) * 0.02)

    def forward(self, x, mask=None):
        """
        x: (B, seq_len, C) in the order [glucose, insulin, meal]
        mask: optional (B, seq_len, C) with 1 for observed and 0 for missing (unused in this minimal version)
        returns: (B, pred_len) glucose forecast
        """
        B, T, C = x.shape
        tokens, P = self.patch_embed(x)                 # (B, C, P, D)
        # add variable id embedding
        tokens = tokens + self.var_emb.view(1, C, 1, -1)

        if self.time_pe is not None:
            # add sinusoidal time PE to each variable stream
            # reshape to (B*C, P, D) -> add PE -> back
            BC = B * C
            t = tokens.reshape(BC, P, -1)
            t = self.time_pe(t)
            tokens = t.view(B, C, P, -1)

        enc = self.encoder(tokens)                      # (B, C, P, D)
        y = self.head(enc)                              # (B, pred_len)
        return y.squeeze()

