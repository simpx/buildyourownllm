import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time

torch.manual_seed(42)

prompts = ["春江", "往事"] # 推理的输入prompts
max_new_token = 100 # 推理生成的最大tokens数量

max_iters = 5000 # 训练的最大迭代次数
eval_iters = 100 # 评估的迭代次数
eval_interval = 200 # 评估的间隔
batch_size = 32 # 每个批次的大小
block_size = 8 # 每个序列的最大长度
learning_rate = 1e-2 # 学习率
n_embed = 32 # 嵌入层的维度
tain_data_ratio = 0.9 # 训练数据占数据集的比例，剩下的是验证数据

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]
    
    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos[i] for i in l])
    
class BabyGPT(nn.Module):

    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 嵌入层，把token映射到n_embd维空间
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性层，把n_embd维空间映射到vocab_size维空间，

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx) # 获得token的嵌入表示 (B,T,n_embd)
        logits = self.lm_head(tok_emb) # 通过线性层，把embedding结果重新映射回vocab_size维空间 (B,T,vocab_size)

        if targets is None: # 推理场景，不需要计算损失值
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 把(B,T,C)的形状转换为(B*T,C)，因为交叉熵损失函数第一个参数只接受二维输入。这个操作并没有丢失信息
            targets = targets.view(B*T) # 把(B,T)的形状转换为(B*T)，因为交叉熵损失函数第二个参数只接受一维输入。这个操作并没有丢失信息
            loss = F.cross_entropy(logits, targets) # 计算交叉熵损失
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # logits的形状是(B,T,vocab_size)，每一个token都计算了下一个token的概率
            logits = logits[:, -1, :] # 实际上我们只需要最后一个token算出来的值
            probs = F.softmax(logits, dim=-1) # 使用softmax函数算概率分布，这里dim=-1表示对最后一个维度进行softmax
            idx_next = torch.multinomial(probs, num_samples=1) # 根据概率分布随机采样，这里num_samples=1表示采样一个token
            idx = torch.cat((idx, idx_next), dim=1) # 把采样的token拼接到序列后面
        return idx
tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size
def count_parameters(model: nn.Module):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # size in MB
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Parameter Size: {param_size:.2f} MB")
    
    # 打印每层参数量
    print("\nParameters by layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")

# 在你的代码中添加：
model = BabyGPT(vocab_size, n_embed).to(device)
count_parameters(model)

print(vocab_size)