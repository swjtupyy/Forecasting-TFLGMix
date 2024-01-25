import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat

from layers.Embed import TokenEmbedding


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)





class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None): # xjt: x: bs, ts_d, seq_num, d_model 
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]  
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )   # xjt: time_in : bs, ts_d, seg_num, d_model ///// attn: bs * ts_d, seq_num, seq_num
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)  # xjt: dim_buffer = bs * seg_num, factor, d_model
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)   # xjt:  dim_send= bs * seg_num, ts_d,  d_model
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)   # xjt: bs * seg_num, ts_d, d_model

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
    
    

# xjt: TempAttention包含对inter-patch 和 intra-patch的attention计算
class TempAttention(nn.Module):
    def __init__(self, configs, win_size, patch_size, mask_flag=True, scale = None, attention_dropout = 0.05, output_attention = False) -> None:
        super().__init__()
        self.configs = configs
        self.win_size = win_size
        self.patch_size = patch_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        
    def forward(self, q_inter, k_inter, v_inter, q_intra, k_intra, v_intra, attn_mask=None, tau=None, delta=None):
        
        # inter-attention
        B, L, H, E = q_inter.shape  #   bs*n_vars, patch_num, n_head, d_model/n_head
        scale_inter = self.scale or 1. / sqrt(E)        
        score = torch.einsum("blhe,bshe->bhls", q_inter, k_inter) * scale_inter   # bs*n_vars, n_heads, patch_num, patch_num
        att_inter = self.dropout(torch.softmax(score, dim=-1))  # bs*n_vars, n_heads, patch_num, patch_num
        V_inter = torch.einsum("bhls,bshd->blhd", att_inter, v_inter)  # bs*n_vars, patch_num, n_heads, d_model/n_head
        
        # intra-attention
        B, L, H, E = q_intra.shape  #  bs*n_vars, patch_size, n_head, d_model/n_head
        scale_intra = self.scale or 1. / sqrt(E)
        score = torch.einsum("blhe,bshe->bhls", q_intra, k_intra) * scale_intra
        att_intra = self.dropout(torch.softmax(score, dim=-1))  # bs*n_vars, n_heads, patch_size, patch_size
        V_intra = torch.einsum("bhls,bshd->blhd", att_intra, v_intra)   # bs*n_vars, patch_size, n_heads, d_model/n_head
        
        
        # inter和intra的结果有不同的维度，需要进行upsample
        V_inter_upsam = repeat(V_inter, 'b patch_num n_heads d -> b (patch_num patch_size) n_heads d', patch_size=self.patch_size)
        V_intra_upsam = repeat(V_intra, 'b patch_size n_heads d -> b (patch_size patch_num) n_heads d', patch_num=self.win_size / self.patch_size)

        att_output = V_inter_upsam + V_intra_upsam  # bs*n_vars, patch_num*patch_size, n_heads, d_model/n_head
        
        return att_output, None        
    
    

class AttentionLayer_Xjt(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.n_heads = n_heads 
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        
        self.out_projection = nn.Linear(d_values * n_heads, d_model)      
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_inter, x_intra, attn_mask):
        
        # inter-patch
        B, L, M = x_inter.shape
        H = self.n_heads
        q_inter, k_inter, v_inter = x_inter, x_inter, x_inter
        q_inter = self.query_projection(q_inter).view(B, L, H, -1) 
        k_inter = self.key_projection(k_inter).view(B, L, H, -1)
        v_inter = self.value_projection(v_inter).view(B, L, H, -1)  

        # intra-patch
        B, L, M = x_intra.shape
        q_intra, k_intra, v_intra = x_intra, x_intra, x_intra
        q_intra = self.query_projection(q_intra).view(B, L, H, -1) 
        k_intra = self.key_projection(k_intra).view(B, L, H, -1)
        v_intra = self.value_projection(v_intra).view(B, L, H, -1)
        
           
        out, att = self.inner_attention(
            q_inter,
            k_inter,
            v_inter,
            q_intra,
            k_intra,
            v_intra,
            attn_mask,
            tau=None,
            delta=None
        )
        # out: bs*n_vars, patch_num*patch_size, n_heads, d_model/n_head
        out = out.view(B, L, -1)  # bs*n_vars, patch_num*patch_size, d_model
        out = self.out_projection(out)
        return out, att


class TempAttention1(nn.Module):
    def __init__(self, configs, win_size, patch_size, inter_flag = True, mask_flag=True, scale = None, attention_dropout = 0.05, output_attention = False) -> None:
        super().__init__()
        self.configs = configs
        self.win_size = win_size
        self.patch_size = patch_size
        self.mask_flag = mask_flag
        self.inter_flag = inter_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        
    def forward(self, q, k, v, inter_flag = True, attn_mask=None, tau=None, delta=None):
        
        # inter-attention   or  intra-attention
        B, L, H, E = q.shape  #   bs*n_vars, patch_num, n_head, d_model/n_head
        scale_inter = self.scale or 1. / sqrt(E)        
        score = torch.einsum("blhe,bshe->bhls", q, k) * scale_inter   # bs*n_vars, n_heads, patch_num, patch_num
        att = self.dropout(torch.softmax(score, dim=-1))  # bs*n_vars, n_heads, patch_num, patch_num
        out = torch.einsum("bhls,bshd->blhd", att, v)  # bs*n_vars, patch_num, n_heads, d_model/n_head
        # if self.inter_flag:
        #     out = repeat(out, 'b patch_num n_heads d -> b (patch_num patch_size) n_heads d', patch_size=self.patch_size)
        # else:
        #     out = repeat(out, 'b patch_size n_heads d -> b (patch_size patch_num) n_heads d', patch_num=self.win_size // self.patch_size)
        
        return out, None        
    
    

class AttentionLayer_Xjt1(nn.Module):
    def __init__(self, attention, d_model, n_heads, win_size, patch_size = 16, d_keys=None, d_values=None):
        super(AttentionLayer_Xjt1, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.n_heads = n_heads 
        
        # xjt: add----------------------
        self.patch_size = patch_size
        self.win_size = win_size
        self.inter_value_embedding = TokenEmbedding(patch_size, d_model)
        self.intra_value_embedding = TokenEmbedding(self.win_size // patch_size, d_model)
        # ------------------------------
        
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        
        self.out_projection = nn.Linear(d_values * n_heads, d_model)      
        

    def forward(self, x_enc, attn_mask = None, tau=None, delta=None):
        # x_enc : bs, l, d_model 
        
        # inter-patch  or intra-patch
        B, L, M = x_enc.shape     # bs*n_vars, patch_num, d_model 
        H = self.n_heads     
        q_inter, k_inter, v_inter = x_enc, x_enc, x_enc
        q_inter = self.query_projection(q_inter).view(B, L, H, -1) 
        k_inter = self.key_projection(k_inter).reshape(B, L, H, -1)
        v_inter = self.value_projection(v_inter).reshape(B, L, H, -1)  
      
        out, att = self.inner_attention(
            q_inter,
            k_inter,
            v_inter,
            attn_mask,
            tau=None,
            delta=None
        )
        # out: bs*n_vars, patch_num*patch_size, n_heads, d_model/n_head
        new_L = out.shape[1]
        out = out.reshape(B, new_L, -1)  # bs*n_vars, patch_num*patch_size, d_model
        out = self.out_projection(out)
        return out, att