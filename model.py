import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_frequencies(dim,seq_length,base=10000):
    theta = 1.0 / (base ** torch.arange(0,dim,2).float()/dim)

    positions = torch.arange(seq_length)

    angles = torch.outer(positions,theta)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos , sin

 # Cos has shape (seq,dim///2) sin too ...so i need to match this when applying rope

def apply_rope(Q,K,cos,sin):

    batch,heads,seq,dim = Q.shape 

    Q_ = Q.reshape(batch,heads,seq,dim//2,2)
    K_ = K.reshape(batch,heads,seq,dim//2,2)

    Q1,Q2 = Q_[...,0],Q_[...,1]
    K1,K2 = K_[...,0],K_[...,1]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    Q_rotated = torch.stack([
        Q1*cos - Q2 * sin,
        Q2* sin + Q1*cos
    ],dim=-1)

    K_rotated = torch.stack([
        K1 * cos - K2 * sin,
        K2 * sin + K1 * cos
    ],dim=-1)

    Q = Q_rotated.flatten(start_dim=-2)
    K = K_rotated.flatten(start_dim=-2)

    return Q,K
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
        rms = torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)

        normalized = x/rms

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

        self.kv_dim = (d_model//num_q_heads) * num_kv_heads

        self.w_0 = nn.Linear(d_model,self.d_model)
        self.w_k = nn.Linear(d_model,self.kv_dim)
        self.w_q = nn.Linear(d_model,self.d_model)
        self.w_v = nn.Linear(d_model,self.kv_dim)

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

        repeat_factor = self.num_q_heads // self.num_kv_heads

        # Q Shape so i can remember when passing RoPE (batch_size,num_h,seq,dim_head)

        K = K.repeat_interleave(repeat_factor,dim=1) 

        V = V.repeat_interleave(repeat_factor,dim=1) 

        cos,sin = get_frequencies(self.d_k,seq_length) # Always use d_k != d_model

        Q,K = apply_rope(Q,K,cos.to(x.device),sin.to(x.device))

        if self.use_flash:
            scores = F.scaled_dot_product_attention(
                Q,K,V,
                attn_mask=None,
                dropout_p = 0.0,
                is_causal = True
            )
            scores = scores.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

            scores = self.w_0(scores)

            return scores


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
            if self.init_method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight)
            elif self.init_method =="xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias,0.0)

    def forward(self,x):
     gate = F.silu(self.layer2(x))
     activated = self.layer1(x) * gate
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

        token_emb = self.token_embeddings(idx)

        x =token_emb

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


# A little Testing
test_input = torch.randint(0, 50000, (2, 32))
try:
    output, loss = model(test_input, test_input)
    print("Forward pass works!")
except Exception as e:
    print(f"Error: {e}")