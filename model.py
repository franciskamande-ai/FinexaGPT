import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Welcome:
'''
TODO:
-Implement model parallelism for large models
-Change positional encoding to rotary embeddings for better extrapolation
'''

# Helper function to create look-ahead mask
def look_ahead_mask(seq_length):
    mask = torch.ones(seq_length,seq_length)
    mask = torch.tril(mask)
    mask = (mask == 0)
    return mask

class RMSNorM(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self,x):
        rsm = torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)

        normalized = x/rsm

        return normalized * self.weight

class GroupedQueryAttention(nn.Module):
    def __init__(self,d_model,num_q_heads,num_kv_heads,init_method = "xavier_uniform",use_flash=True):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self.init_method = init_method

        self.use_flash = use_flash

        self.d_k = d_model // num_q_heads

        assert d_model // num_q_heads,"d_model must be divisible by num_heads"

        self.scale = math.sqrt(self.d_k)

        self.w_0 = nn.Linear(d_model,self.d_k)
        self.w_k = nn.Linear(d_model,self.d_k)
        self.w_q = nn.Linear(d_model,self.d_k)
        self.w_v = nn.Linear(d_model,self.d_k)

        self._initialize_weights()

    def _initialize_weights(self,init_method = "xavier_uniform"):
        for layer in [self.w_k,self.w_q,self.w_v,self.w_0]:
            if init_method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight)
            elif  init_method == "xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight)

            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias,0.0)

    def forward(self,x,mask=None):
        batch_size,seq_length,_ = x.shape

        Q = self.w_q(x).view(batch_size,seq_length,self.num_q_heads,self.d_k).transpose(1,2)

        K = self.w_k(x).view(batch_size,seq_length,self.num_kv_heads,self.d_k).transpose(1,2)
        V = self.w_v(x).view(batch_size,seq_length,self.num_kv_heads,self.d_k).transpose(1,2)

        repeat_factor = self.d_model/(self.num_q_heads/self.num_kv_heads)

        K = K.repeat_interleave(repeat_factor,dim=2)

        V = V.repeat_interleave(repeat_factor,dim=2)
        
        if self.use_flash:
            scores = F.scaled_dot_product_attention(
                Q,K,V,
                attention_mask=None,
                dropout_p = 0.0,
                is_causal = True
            )

        else:
            scores = torch.matmul(Q,K.transpose(-2,-1))/self.scale

            if mask is None:
                mask = look_ahead_mask(seq_length).to(x.device)
            if mask is not None:
                scores = scores.masked_fill(mask,float('-inf'))

            scores = F.softmax(scores,dim=-1)

            scores = torch.matmul(scores,V)

            scores = scores.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)

            scores = self.w_0(scores)
            return scores

class SwiGLUFFN(nn.Module):
    def __init__(self,d_model,init_method = "xavier_uniform",dropout = 0.1):
        super().__init__()
        self.init_method = init_method
        self.d_model = d_model
        self.layer1  = nn.Linear(self.d_model,self.d_model*4)
        self.layer2 = nn.Linear(self.d_model,self.d_model*4)
        self.layer3 = nn.Linear(self.d_model*4,self.d_model)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.layer1,self.layer2,self.layer3]:
            if self.init_method == "xaivier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight)
            elif self.init_method =="xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.constant_(layer.bias,0.0)

    def forward(self,x):
     gate = F.silu(self.layer2)
     activated = self.layer1 * gate
     return self.layer3(activated)

class TransformerBlock(nn.Module):
    def __init__(self,d_model=768,num_q_heads=8,num_kv_heads=2,dropout=0.1,init_method="xavier_uniform",use_flash=True):
        super().__init__()
        self.attention = GroupedQueryAttention(d_model=d_model,num_q_heads=num_q_heads,num_kv_heads=num_kv_heads,init_method=init_method,use_flash=True)
        self.ffn = SwiGLUFFN(d_model=d_model,init_method=init_method,dropout=dropout)
        self.norm1 = RMSNorM(d_model)
        self.norm2 = RMSNorM(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        batch_size,seq_length,d_model = x.shape
        mask = look_ahead_mask(seq_length)

        x = x + self.dropout(self.attention(self.norm1(x),mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x



class Transformer(nn.Module):
    def __init__(self, num_q_heads=12,num_kv_heads=2, d_model=768, num_layers=12, 
                 dropout=0.1, vocab_size=50000, max_seq_length=512,use_flash=True):
        super().__init__()
        self.num_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

        self.token_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_embeddings = nn.Embedding(self.max_seq_length, self.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                num_q_heads=num_q_heads,
                num_kv_heads = num_kv_heads, 
                dropout=dropout, 
                init_method="xavier_uniform",
                use_flash = use_flash
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorM(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance (module, nn.Linear):
            nn.init.normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias,0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight)


    def forward(self,idx,targets=None,mask=None):
        batch_size,seq_length = idx.shape

        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequnce {seq_length} is greater than Maximum {self.max_seq_length}"
                )

        pos = torch.arange(0,seq_length,dtype=torch.long,device=idx.device)

        pos_emb = self.positional_embeddings(pos)
        pos_emb = pos_emb.unsqueeze(0)

        token_emb = self.token_embeddings(idx)

        x = pos_emb + token_emb

        for i, block in enumerate(self.blocks):
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B,T,C = logits.shape

            logits_flat = logits.view(B*T,C)
            targets_flat = targets.view(B*T)

            loss = F.cross_entropy(logits_flat,targets_flat)
        return logits,loss

    def generate(self,idx,max_new_tokens=100,temperature=1.0,top_k=None):
        idx = idx.clone()
        for _ in range(max_new_tokens):
            if idx.size(1)>self.max_seq_length:
                idx = idx[:,-self.max_seq_length:]

            logits,_ = self(idx)
            logits = logits[:,-1,:]/temperature

            if top_k is not None:
                v,_ = torch.topk(logits,top_k)
                logits[logits<v[:,[-1]]] = float("-inf")

            probs = F.softmax(logits,dim=-1)
            next_idx = torch.multinomial(probs,num_samples=1)
            idx = torch.cat([idx,next_idx],dim=1)
        return idx

model = Transformer(num_q_heads=12,num_kv_heads=4,d_model=768,num_layers=12,dropout=0.1,vocab_size=50000,max_seq_length=512,use_flash=True)

parameters = sum(p.numel() for p in model.parameters())

print(f"Total Parameters: {parameters:,}")
