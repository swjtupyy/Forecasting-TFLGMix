import torch
from torch import nn
from layers.Transformer_EncDec import Encoder_Xjt, EncoderLayer_Xjt
from layers.SelfAttention_Family import  AttentionLayer_Xjt1, TempAttention1
from layers.Embed import PatchEmbedding, DataEmbedding, PositionalEmbedding
from einops import rearrange, repeat


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
    

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in

        padding = stride
        
        self.patch_size = patch_len
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        # inter patch embedding + intra patch embedding
        self.embedding_inter_patch = DataEmbedding(patch_len, configs.d_model, dropout=configs.dropout)
        self.embedding_intra_patch = DataEmbedding(self.seq_len // patch_len, configs.d_model, dropout=configs.dropout)
        self.embedding_position = PositionalEmbedding(configs.d_model)
        self.first_projection = nn.Linear(1, configs.d_model)

        # Encoder
        self.encoder1 = Encoder_Xjt(
            [
                EncoderLayer_Xjt(
                    AttentionLayer_Xjt1(
                        TempAttention1(configs, self.seq_len, patch_len, inter_flag=True, attention_dropout=configs.dropout, output_attention=configs.output_attention), 
                            configs.d_model, configs.n_heads, win_size=self.seq_len, patch_size=self.patch_size),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.encoder2 = Encoder_Xjt(
            [
                EncoderLayer_Xjt(
                    AttentionLayer_Xjt1(
                        TempAttention1(configs, self.seq_len, patch_len, inter_flag=False, attention_dropout=configs.dropout, output_attention=configs.output_attention), 
                            configs.d_model, configs.n_heads, win_size=self.seq_len, patch_size=self.patch_size),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        


        # Prediction Head
        self.head_nf = configs.d_model * self.seq_len
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)
        
    

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        
        x_inter = x_enc.permute(0, 2, 1) 
        x_intra = x_enc.permute(0, 2, 1)    
        
        x_inter = rearrange(x_inter, 'b m (n p) -> (b m) n p', p = self.patch_size)
        x_inter = self.embedding_inter_patch(x_inter)
        
        x_intra = rearrange(x_intra, 'b m (p n) -> (b m) p n', p = self.patch_size)
        x_intra = self.embedding_intra_patch(x_intra)
        
        inter_out, _ = self.encoder1(x_inter)  # bs*n_vars, patch_num, d_model
        intra_out, _ = self.encoder2(x_intra)  # bs*n_vars, patch_size, d_model
        
        inter_out = repeat(inter_out, 'b patch_num d -> b (patch_num patch_size) d', patch_size=self.patch_size)
        intra_out = repeat(intra_out, 'b patch_size d -> b (patch_size patch_num) d', patch_num=self.seq_len // self.patch_size)
        
        enc_out = inter_out + intra_out
        
        
        
        enc_out = torch.reshape(
            enc_out, (-1, self.n_vars, enc_out.shape[-2], enc_out.shape[-1]))   # # z: [bs x nvars x (patch_num * patch_size) x d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)   # z: [bs x nvars x d_model x (patch_num * patch_size)]
        
        # Decoder
        dec_out = self.head(enc_out)   # [bs x nvars x pred_len]
        dec_out = dec_out.permute(0, 2, 1)
         # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:, -self.pred_len:, :]

