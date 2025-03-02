import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import argparse

# 超参数
batch_size = 64 # 一批包含的文本序列个数
block_size = 256 # 一个文本序列包含的字符个数
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # 进一步降低了学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_head = 6
n_layer = 6
dropout = 0.2
n_embed = 384 # 不太明白，感觉就是随便定义的一个纬度值
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# 数据准备
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 词汇表直接来自于text，包含符号、换行、空格
chars = sorted(list(set(text)))
vocab_size = len(chars)

# decode、encode函数，在序号和字符间转换
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 数据loader

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 前90%的数据当作训练数据，后10%的数据当作校验数据
train_data = data[:n]
val_data = data[n:]


# 获得一份随机的训练数据或者校验数据
# 格式为(Batch_size, Block_size)
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 获得batch_size个随机数
    x = torch.stack([data[i:i+block_size] for i in ix]) # 用随机数获得batch_size组数据，每组有block_size长
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # y是x中每个token往后偏移1的新数据
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):
    """ 
    one head of self-attension 
    原理见gpt-dev.ipynb
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # pytorch里的module都会被自动当作layer来处理，用register_buffer后，这里就是一个普通的变量
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # 这里的C是n_embed
        k = self.key(x) # (B, T, head_size) 
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) = (B, T, T)，最后呈上C**-0.5避免softmax过于稀疏
        #print(wei.shape)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 上三角都是-inf，下三角是k y的点积
        #print(wei.shape)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        #print(wei.shape)
        v = self.value(x) # (B, T, head_size)
        #print(x.shape, wei.shape, v.shape)
        out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        # 至此，返回了一个(B, T, head_size) 的tensor，里面包含的信息：
        # 1. token本身的
        # 2. 这个token之前的所有token的
        # 这就是attension
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # 增加一个投影，通过linear实现。为什么要增加这一层？
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedFoward(nn.Module):
    '''
    这里包含一个线性变化层，给模型增加了表达能力，虽然我也不知道为什么
    然后增加了一个激活函数，把负值变为0了
    just magic
    # 为什么叫 FeedForward 层？
    前馈神经网络（FeedForward Neural Network）：
    在前馈神经网络中，数据从输入层通过隐藏层传递到输出层，每一层的输出作为下一层的输入，没有循环或反馈连接。这个过程被称为前馈（FeedForward）。
    FeedForward 层是前馈神经网络的基本构建块，通常由一个线性变换和一个非线性激活函数组成。
    增加模型的表达能力：
    线性变换（nn.Linear）可以对输入进行加权求和，但它本质上是线性的，无法捕捉到复杂的非线性关系。
    激活函数（如 nn.ReLU）引入了非线性，使得模型能够表示更复杂的函数关系，从而增加模型的表达能力。
    '''
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # 论文里说要用4倍
            nn.ReLU(), # 把负值变为0，正直不变的激活函数
            nn.Linear(4 * n_embed, n_embed), # 增加了一个projection layer。这里似乎有一个好处，就是允许前面的layer用更多的维度
            # karpathy说，dropout层可以放在残差连接之前，而这个而这里的输出就是打算用在残差连接的输入的
            # dropout会随机丢掉一些特征，并且缩放没丢掉的特征
            # dropout=0.5
            # >>> x
            # tensor([[ 1.,  2.,  3.,  4.,  5.],
            #         [ 6.,  7.,  8.,  9., 10.]])
            # >>> output
            # tensor([[ 2.,  0.,  0.,  8., 10.],
            #         [ 0.,  0., 16., 18., 20.]])
            nn.Dropout(dropout),
        )
    def forward(self, x):
        # 这是一个token level的操作
        return self.net(x)

class Block(nn.Module):
    """
    transformer block
    包含多头自注意力 + feedforward
    """
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        '''
        （Layer Normalization，层归一化）是一种归一化技术，用于神经网络中，以提高训练的稳定性和速度。它在每一层的输入上进行归一化处理，使得每一层的输入具有零均值和单位方差。

        通俗解释
        在神经网络中，输入数据的分布可能会在训练过程中发生变化，这种现象被称为“内部协变量偏移”（Internal Covariate Shift）。这种变化会导致训练过程变得不稳定，学习率需要非常小，训练时间变长。
        
        tensor([[1., 2., 3.],
                [4., 5., 6.]])
        层归一化后变成：
        tensor([[-1.2247,  0.0000,  1.2247],
                [-1.2247,  0.0000,  1.2247]], grad_fn=<NativeLayerNormBackward0>)
        '''
        
        self.ln1 = nn.LayerNorm(n_embed) # layer norm layer
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        '''
        在深度学习和神经网络中，"residual connection"（残差连接）是一种技术，用于缓解深层网络中的梯度消失问题，并加速训练过程。它最早由 He et al. 在 ResNet（Residual Networks）中提出。

        残差连接的基本概念
        残差连接的基本思想是通过引入一个跳跃连接（skip connection），将输入直接添加到输出，从而形成一个残差块（residual block）。具体来说，假设输入为 ( x )，经过若干层变换后的输出为 ( F(x) )，那么残差块的输出为：

        [ y = F(x) + x ]

        这种结构允许梯度直接通过跳跃连接传播，从而缓解梯度消失问题，并使得网络更容易训练。

        说白了，就是算出参数后，再加上原来的x

        下面的加法是为了“残差连接”，而ln是层归一化
        '''
        x = x + self.sa(self.ln1(x)) # 使用了残差连接，我估计作用是，保留原来的x信息，避免梯度消失
        x = x + self.ffwd(self.ln2(x))
        return x

    
# 模型
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # 这里就是把token，映射成了n_embed维的向量
        self.postion_embedding_table = nn.Embedding(block_size, n_embed) # 建设一个“位置”映射关系
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # language model head的缩写，为了作最后一层，用来把模型输出成词汇表大小
    
    def forward(self, idx, targets=None):
        B, T = idx.shape # B是batch size，T是block size
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # 获得词向量，因为输入是(B,T)，因此输出是(B,T,C)
        # torch.arange(T)就是生成[0, 1, 2, ..., T-1]，也就是这些token的“位置序号”
        # 输出了序号的向量
        # 有意思的是，此时pos_emb只是序号的向量。按道理说每一次pos_emb的值都是固定的
        # 如果只是推理而不是训练的话，这个值明显可以被缓存
        pos_emb = self.postion_embedding_table(torch.arange(T, device=idx.device))
        # 神奇的加法，这一加，就让x同时被token和位置向量影响了
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # 输入idx，比如(B, 1)，续写最多max_new_tokens个字符
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 推理
            # 由于我们有位置嵌入，而且最多只支持0~block size的输入，因此这里得限制idx的输入
            # 虽然限制了输入，但实际上对一个bigramlanguageModel来说没差别，反而性能变好了
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # 只需要取最后一个值当作预测值。我觉得这是很不高效的，因为我们用了linear作为最后一层，做了很多无谓的计算
            logits = logits[:, -1, :] # becomes (B, C)
            # 使用softmax做概率分布
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 根据概率分布随机采样，值为1
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) 随机采样函数，模型输出的随机性来自这里
            # append结果到idx后面去
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# 训练

model = BigramLanguageModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# 计算当前损失值
# 计算eval_iters次训练数据的损失值的平均值，再计算eval_iters次校验数据的损失值的平均值
# 都返回出来
@torch.no_grad() # no_grad要求pytorch不要自动求导
def estimate_loss():
    out = {}
    model.eval() # 让模型内部的特定层都使用评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_time = time.time()
for iter in range(max_iters):
   

    # 隔一段时间，输出当前的损失值
    if iter % eval_interval == 0:
        losses = estimate_loss()
        elapsed_time = time.time() - start_time
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, elapsed time {elapsed_time:.2f}s, remaining batches {max_iters - iter}")

    # sample一份数据
    xb, yb = get_batch('train')

    # 评估损失值，算梯度、调参
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter == max_iters - 1:
        losses = estimate_loss()
        elapsed_time = time.time() - start_time
        print(f"finally step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, elapsed time {elapsed_time:.2f}s, remaining batches {max_iters - iter}")

# 推理

context = torch.zeros((1, 1), dtype=torch.long, device=device) # 输入了一段
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# 保存模型参数到文件
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# 解析命令行参数
parser = argparse.ArgumentParser(description='Save model parameters')
parser.add_argument('--filename', type=str, default='model_params.pth', help='Filename to save the model parameters')
args = parser.parse_args()

# 保存模型参数
save_model(m, args.filename)