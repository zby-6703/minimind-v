import math, torch, torch.nn.functional as F  # 导入数学库、PyTorch 主库，以及常用的函数式接口 F（如 softmax、cross_entropy）。
from torch import nn  # nn 里包含神经网络常用模块，比如 Linear、Embedding、Dropout、Module。
from transformers.activations import ACT2FN  # Transformers 提供的“激活函数名字 -> 函数”的映射表，比如 'silu'。
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig  # 继承这些类后，模型能更好接入 Hugging Face 的保存、加载和生成接口。
from transformers.modeling_outputs import MoeCausalLMOutputWithPast  # Hugging Face 定义的输出结构，适合带 MoE 和 past_key_values 的因果语言模型。

# 初学者阅读提示：
# - Python 里 `#` 后面的内容是注释，不会被程序执行，只是写给人看的说明。
# - `class Xxx(...)` 是定义一个类；类可以理解成“带有数据和功能的模板”。
# - `def xxx(...)` 是定义函数或方法；写在类里面的函数通常叫“方法”。
# - `self.xxx` 表示“当前这个对象自己的 xxx 属性”，模型参数和配置通常都挂在 self 上。
# - `True` / `False` 是布尔值，表示开关；`None` 表示“没有值”。
# - 深度学习代码里经常写张量形状，如 `[batch, seq_len, hidden_size]`，意思是一个多维数组的三个维度。

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):  # 定义模型配置类，保存所有超参数，并兼容 Hugging Face 的配置系统。
    model_type = "minimind"  # Hugging Face 用这个字符串识别模型类型。

    # `__init__` 是 Python 类的构造函数：创建 MiniMindConfig(...) 对象时会自动运行。
    # 形如 `hidden_size=768` 的参数表示“默认值”：用户不传 hidden_size 时，就用 768。
    # `**kwargs` 是 Python 的固定写法，意思是“接收额外的命名参数，并打包成一个字典 dict”。
    # 例如 MiniMindConfig(vocab_size=10000, dropout=0.1) 中，没有写在参数列表里的 vocab_size/dropout 会进入 kwargs。
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):  # 初始化配置；未显式传入的参数从 kwargs 或默认值读取。
        # `super()` 表示父类 PretrainedConfig；`**kwargs` 这里表示“把 kwargs 字典重新展开成命名参数传给父类”。
        super().__init__(**kwargs)  # 先让父类 PretrainedConfig 处理通用配置，例如 tokenizer 相关字段。
        # 下面这些 `self.xxx = ...` 都是在对象里保存配置；之后模型会通过 config.xxx 读取这些值。
        # 这些默认值不是 Python 规定，而是模型作者选择的一组“超参数默认值”；大多数都可以通过 kwargs 覆盖。
        self.hidden_size = hidden_size  # 每个 token 的隐藏向量维度，也就是模型内部表示的宽度。
        self.num_hidden_layers = num_hidden_layers  # Transformer Block 的层数，层数越多通常表达能力越强。
        self.use_moe = use_moe  # 是否启用 MoE（Mixture of Experts，混合专家）前馈网络。
        # `kwargs.get("dropout", 0.0)` 是字典取值：如果 kwargs 里有 "dropout" 就用它，否则用默认值 0.0。
        # 之后很多行都用这个模式：既允许外部覆盖参数，又给出一个可运行的默认值。
        self.dropout = kwargs.get("dropout", 0.0)  # Dropout 概率；训练时随机丢弃部分神经元以减少过拟合。
        self.vocab_size = kwargs.get("vocab_size", 6400)  # 词表大小，即模型能识别多少种 token id。
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # 序列开始 token 的 id，BOS = beginning of sequence。
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # 序列结束 token 的 id，EOS = end of sequence。
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否优先使用 PyTorch 的高效注意力实现。
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)  # Query 注意力头数量，多头注意力会把表示拆成多个子空间并行学习。
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # Key/Value 头数量；小于 Query 头数时就是 GQA，可节省显存和计算。
        # `//` 是整数除法，例如 768 // 8 = 96；这里要求每个 head 的维度是整数。
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 每个注意力头的维度，通常等于 hidden_size / head 数。
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # 前馈网络使用的激活函数名称。
        # 这行初看很绕，可以拆开理解：
        # 1. `math.pi` 是 Python math 模块里的圆周率 π，约等于 3.14159。
        # 2. `hidden_size * math.pi` 表示把 MLP 中间层设成 hidden_size 的约 3.14 倍，这是该项目采用的经验设计。
        # 3. `/ 64` 先看需要多少个 64；`math.ceil(...)` 是“向上取整”，例如 ceil(37.2)=38。
        # 4. 最后 `* 64` 把结果变回维度，所以整体效果是“向上取到最接近的 64 的倍数”。
        #    这样做常见于深度学习工程，因为 64 的倍数通常更适合 GPU/矩阵乘法计算。
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)  # 前馈网络中间层维度，向上取到 64 的倍数便于计算优化。
        # `32768` 大约是 32K token 上下文；这里不是训练样本数量，而是模型能准备的位置编码长度。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 模型预计算的最大位置长度，也就是最长上下文长度上限。
        # `1e-6` 是科学计数法，等于 0.000001；小数值常用于防止除以 0。
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 中防止除零的小数值。
        # `1e6` 等于 1,000,000；RoPE 里这个值会影响不同位置的旋转频率。
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # RoPE 旋转位置编码的基频参数，影响长上下文表现。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # 是否让输入 embedding 和输出 lm_head 共享同一份权重。
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)  # 推理时是否使用 RoPE 缩放以支持更长上下文。
        # `{...}` 是字典；`"beta_fast": 32` 表示 key 是字符串 "beta_fast"，value 是数字 32。
        # `A if 条件 else B` 是 Python 的条件表达式：条件为真用 A，否则用 B。
        # 所以下面这段表示：如果 inference_rope_scaling=True，就使用这个字典；否则 rope_scaling=None。
        self.rope_scaling = {  # 如果启用长上下文推理，这里配置 YaRN 风格的 RoPE 缩放参数。
            "beta_fast": 32,  # 高频部分的变化边界参数。
            "beta_slow": 1,  # 低频部分的变化边界参数。
            "factor": 16,  # 位置编码扩展倍数。
            "original_max_position_embeddings": 2048,  # 训练时原始上下文长度，用来判断是否需要扩展。
            "attention_factor": 1.0,  # 对 cos/sin 位置编码整体乘的缩放系数。
            "type": "yarn"  # 标明使用的 RoPE 缩放类型。
        } if self.inference_rope_scaling else None  # 未启用长上下文缩放时，rope_scaling 为空。
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)  # MoE 中专家网络的总数量。
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每个 token 会被路由到几个专家。
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # MoE 专家内部前馈网络的中间层维度。
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)  # 是否把选中的 top-k 专家概率重新归一化为和为 1。
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # MoE 路由辅助损失系数，用来鼓励专家负载更均衡。

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):  # RMSNorm 是一种归一化层，作用类似 LayerNorm，但计算更简单。
    def __init__(self, dim: int, eps: float = 1e-5):  # dim 是最后一维的大小，eps 用于数值稳定。
        super().__init__()  # 初始化 nn.Module 的内部状态。
        self.eps = eps  # 保存 eps，归一化时会加到分母里防止除以 0。
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放参数，初始为 1，形状是 [dim]。

    def norm(self, x):  # 只做 RMS 归一化，不包含可学习的 weight 缩放。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # x 除以均方根；rsqrt 表示 1 / sqrt(...)。

    def forward(self, x):  # 前向传播：输入张量 x，输出归一化后的张量。
        return (self.weight * self.norm(x.float())).type_as(x)  # 先转 float 提高稳定性，再转回原 dtype，最后乘可学习权重。

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):  # 预计算 RoPE 需要的 cos/sin 表。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0  # 生成每个偶数维度对应的旋转频率，并初始化注意力缩放系数。
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (  # 从配置中取出 YaRN 缩放所需的参数。
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),  # 原始上下文长度和扩展倍数。
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)  # 高频/低频边界和整体注意力缩放。
        )
        if end / orig_max > 1.0:  # 只有目标长度超过原始训练长度时，才真正调整 RoPE 频率。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))  # 根据 beta 反推出频率维度边界。
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)  # 计算需要平滑过渡的维度区间。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)  # 生成 0 到 1 的线性过渡系数。
            freqs = freqs * (1 - ramp + ramp / factor)  # 对不同频率维度做平滑缩放，低维保持更多原始信息，高维扩展更明显。
    t = torch.arange(end, device=freqs.device)  # `torch.arange(end)` 会生成 [0, 1, 2, ..., end-1] 这样的 1D 张量。
    freqs = torch.outer(t, freqs).float()  # `torch.outer(a, b)` 做外积，得到“每个位置 x 每个频率”的二维表。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # `torch.cos` 逐元素取余弦；`torch.cat` 是按最后一维拼接。
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor  # `torch.sin` 逐元素取正弦；cos/sin 表大小要和 head_dim 对齐。
    return freqs_cos, freqs_sin  # 返回预计算好的 cos 和 sin，后续每层注意力会复用。

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):  # 把 RoPE 旋转位置编码应用到 query 和 key 上。
    def rotate_half(x):  # 定义一个内部函数：把向量后一半取负并放到前面，实现二维旋转的一部分。
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)  # 最后一维拆成两半，组合成旋转后的形状。

    # `unsqueeze(dim)` 的意思是在指定位置“插入一个长度为 1 的维度”，方便和 q/k 做广播相乘。
    # 这里的广播（broadcast）是 PyTorch 的自动对齐机制：小张量会被看作“按规则复制”成大张量来运算。    
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)  # 对 query 应用旋转位置编码，并保持原来的 dtype。
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)  # 对 key 应用同样的位置旋转，让注意力感知相对位置。
    return q_embed, k_embed  # 返回加入位置信息后的 query 和 key。

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:  # 把 key/value 头复制到和 query 头数量一致，用于  。
    bs, slen, num_key_value_heads, head_dim = x.shape  # 读取张量形状：batch、序列长度、kv 头数、每头维度。
    if n_rep == 1:  # 如果 kv 头数已经等于 query 头数，就不用复制。
        return x  # 直接返回原张量。
    # `None` 在索引里等价于插入一个新维度，和 `unsqueeze` 很像。
    # `expand(...)` 不是实际拷贝内存，而是用“视图 + 广播”的方式让张量看起来被复制了。
    # `reshape(...)` 再把形状整理回正常的四维/三维结构。
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))  # 增加复制维度、广播复制，再合并成更多的头。

class Attention(nn.Module):  # 多头自注意力模块，是 Transformer 的核心计算之一。
    def __init__(self, config: MiniMindConfig):  # 根据配置创建注意力层需要的线性层和参数。
        super().__init__()  # 初始化 nn.Module。
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads  # 如果没指定 kv 头数，就让它等于 query 头数。
        self.n_local_heads = config.num_attention_heads  # 保存 query 注意力头数量。
        self.n_local_kv_heads = self.num_key_value_heads  # 保存 key/value 注意力头数量。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 kv 头需要复制几份才能匹配 query 头数。
        self.head_dim = config.head_dim  # 每个注意力头的向量维度。
        self.is_causal = True  # 因果语言模型只能看当前位置及之前的 token，不能偷看未来。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)  # 把 hidden_states 投影成 query。
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 把 hidden_states 投影成 key。
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 把 hidden_states 投影成 value。
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)  # 把多头注意力输出合并回 hidden_size。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 对每个 query 头做 RMSNorm，提升训练稳定性。
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 对每个 key 头做 RMSNorm，提升训练稳定性。
        self.attn_dropout = nn.Dropout(config.dropout)  # 注意力权重上的 dropout。
        self.resid_dropout = nn.Dropout(config.dropout)  # 注意力输出上的 dropout，之后会走残差连接。
        self.dropout = config.dropout  # 保存 dropout 概率，flash attention 调用时会用到。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn  # 判断当前 PyTorch 是否支持高效注意力，并且配置允许使用。

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):  # 注意力前向传播，输入 x 形状通常是 [batch, seq_len, hidden_size]。
        bsz, seq_len, _ = x.shape  # 读取 batch 大小和当前输入序列长度。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # 同一份输入分别投影成 query、key、value。
        # `view(...)` 是“重塑形状”，前提是底层内存连续；这里把最后一维拆成“头数 x 每头维度”。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # 把 query 拆成多头形状 [batch, seq, q_heads, head_dim]。
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # 把 key 拆成多头形状 [batch, seq, kv_heads, head_dim]。
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # 把 value 拆成多头形状 [batch, seq, kv_heads, head_dim]。
        xq, xk = self.q_norm(xq), self.k_norm(xk)  # 对 query 和 key 做归一化，value 不做。
        cos, sin = position_embeddings  # 取出当前序列位置对应的 RoPE cos/sin 表。
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 给 query/key 加入旋转位置编码。
        if past_key_value is not None:  # 如果推理时传入了历史 key/value，就复用之前 token 的缓存。
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 沿序列长度维度拼接历史 key 和当前 key。
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 沿序列长度维度拼接历史 value 和当前 value。
        past_kv = (xk, xv) if use_cache else None  # 如果开启缓存，就把新的完整 key/value 返回给下一步生成使用。
        # `transpose(1, 2)` 交换第 1 和第 2 个维度，把张量从 [batch, seq, heads, dim] 变成 [batch, heads, seq, dim]。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))  # 调整成注意力计算需要的 [batch, heads, seq, head_dim]。
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):  # 条件允许时使用 PyTorch 内置高效 attention。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)  # 直接计算缩放点积注意力，训练时才启用 dropout。
        else:  # 如果不能用 flash attention，就手动实现注意力计算。
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # query 和 key 点积得到注意力分数，并除以 sqrt(head_dim) 稳定数值。
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)  # 加上上三角 mask，让当前位置看不到未来位置。
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9  # 把 padding 位置的分数压到极小，softmax 后接近 0。
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv  # softmax 得到注意力权重，再加权求和 value。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # `reshape(..., -1)` 里的 -1 表示“让 PyTorch 自动推断这一维的大小”。
        output = self.resid_dropout(self.o_proj(output))  # 输出投影回 hidden_size，并做 dropout。
        return output, past_kv  # 返回注意力结果，以及可选的 key/value 缓存。

class FeedForward(nn.Module):  # 普通前馈网络，也叫 MLP，是 Transformer block 中注意力之后的非线性变换。
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):  # intermediate_size 可覆盖默认中间层大小。
        super().__init__()  # 初始化 nn.Module。
        # `a or b` 是 Python 的短路写法：如果 a 为“真值”就返回 a，否则返回 b。
        # 这里的意思是：如果外部传了 intermediate_size 就用它；没传就退回 config.intermediate_size。
        intermediate_size = intermediate_size or config.intermediate_size  # 如果没传入中间层大小，就使用配置里的默认值。
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 门控分支，把 hidden_size 升维到 intermediate_size。
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # 降维分支，把 intermediate_size 映射回 hidden_size。
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 另一条升维分支，用来和激活后的门控分支逐元素相乘。
        self.act_fn = ACT2FN[config.hidden_act]  # 根据字符串选择激活函数，例如 silu。

    def forward(self, x):  # 前馈网络的前向传播。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))  # SwiGLU 风格：激活(gate) 与 up 分支相乘，再降维。

class MOEFeedForward(nn.Module):  # MoE 版本的前馈网络：多个专家 MLP，由路由器决定每个 token 走哪些专家。
    def __init__(self, config: MiniMindConfig):  # 根据配置创建路由器和专家列表。
        super().__init__()  # 初始化 nn.Module。
        self.config = config  # 保存配置，forward 时会用到专家数量、top-k 等参数。
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # 路由器：给每个 token 输出 num_experts 个专家分数。
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])  # 创建多个专家，每个专家都是一个 FeedForward。
        self.act_fn = ACT2FN[config.hidden_act]  # 保存激活函数；这个变量当前没有直接使用，但保留了配置对应关系。

    def forward(self, x):  # MoE 前向传播，输入 x 形状是 [batch, seq_len, hidden_dim]。
        batch_size, seq_len, hidden_dim = x.shape  # 读取输入的三个维度。
        x_flat = x.view(-1, hidden_dim)  # 把 batch 和 seq_len 合并，方便按 token 路由；`-1` 表示这一维由 PyTorch 自动算出来。
        scores = F.softmax(self.gate(x_flat), dim=-1)  # 路由器输出专家概率，每个 token 对所有专家的概率和为 1。
        # `torch.topk` 会返回两个东西：最大的值（weight）和对应下标（idx）。
        # 例如从 4 个专家里选 1 个，就相当于“这个 token 主要交给哪个专家处理”。
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)  # 为每个 token 选择概率最高的 top-k 个专家。
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)  # 重新归一化 top-k 概率，避免权重和不是 1。
        y = torch.zeros_like(x_flat)  # 创建输出缓冲区，形状和 x_flat 一样，后面把专家结果加进去。
        for i, expert in enumerate(self.experts):  # 逐个遍历专家网络。
            mask = (topk_idx == i)  # 找出哪些 token 的 top-k 专家里包含当前专家 i。
            if mask.any():  # 如果当前专家至少被一个 token 选中，就真正计算它。
                # `mask.any(dim=-1)` 表示“这一行里是否至少有一个 True”。
                # `nonzero()` 会把 True 的位置找出来；`flatten()` 再把结果压成一维。
                token_idx = mask.any(dim=-1).nonzero().flatten()  # 取出需要送入当前专家的 token 下标。
                weight = topk_weight[mask].view(-1, 1)  # 取出这些 token 分配给当前专家的权重，并变成列向量。
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))  # 专家输出乘路由权重后，加回对应 token 的输出位置。
            elif self.training:  # 训练时如果某个专家没有被选中，也让它参与计算图。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())  # 乘以 0 不改变数值，但能避免分布式训练中未使用参数的问题。
        if self.training and self.config.router_aux_loss_coef > 0:  # 训练时可计算路由辅助损失，帮助专家负载均衡。
            # `F.one_hot(...)` 会把整数下标变成 one-hot 向量，例如 2 -> [0, 0, 1, 0]。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)  # 统计每个专家被选中的平均比例。
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef  # 负载比例和平均概率相乘，作为辅助损失。
        else:  # 推理或辅助损失系数为 0 时，不额外惩罚路由。
            self.aux_loss = scores.new_zeros(1).squeeze()  # 创建一个和 scores 同设备同 dtype 的 0 标量。
        return y.view(batch_size, seq_len, hidden_dim)  # 把扁平化输出还原成 [batch, seq_len, hidden_dim]。

class MiniMindBlock(nn.Module):  # 一个 Transformer Block，包含注意力、前馈网络和两处归一化。
    def __init__(self, layer_id: int, config: MiniMindConfig):  # layer_id 是层编号，这里没有使用，但保留接口方便扩展。
        super().__init__()  # 初始化 nn.Module。
        self.self_attn = Attention(config)  # 创建自注意力层。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力前的 RMSNorm。
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 前馈网络前的 RMSNorm。
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)  # 根据配置选择普通 MLP 或 MoE MLP。

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):  # 一个 block 的前向传播。
        residual = hidden_states  # 保存残差分支，稍后把注意力输出加回原输入。
        # 这里采用 Pre-Norm 结构：先归一化，再进入注意力/MLP；这种结构在深层 Transformer 里更稳定。
        hidden_states, present_key_value = self.self_attn(  # 对归一化后的 hidden_states 做自注意力。
            self.input_layernorm(hidden_states), position_embeddings,  # 先做 RMSNorm，再传入 RoPE 位置编码。
            past_key_value, use_cache, attention_mask  # 传入可选 KV 缓存、是否返回缓存，以及 padding mask。
        )
        hidden_states += residual  # 残差连接：注意力输出 + block 输入，帮助深层网络稳定训练。
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # 第二个残差连接：MLP(归一化后的状态) + 当前状态。
        return hidden_states, present_key_value  # 返回新的隐藏状态，以及当前层的 KV 缓存。

class MiniMindModel(nn.Module):  # 不含最终词表分类头的 Transformer 主体。
    def __init__(self, config: MiniMindConfig):  # 根据配置搭建 embedding、多个 block、最终 norm 和 RoPE 缓冲区。
        super().__init__()  # 初始化 nn.Module。
        self.config = config  # 保存配置对象，forward 和生成时会用到。
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers  # 保存词表大小和层数。
        # `nn.Embedding(词表大小, 向量维度)` 可以理解成一张可学习的查表矩阵。
        # 输入 token id=5 时，它会取出第 5 行向量作为这个 token 的表示。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # token embedding：把 token id 映射成 hidden_size 维向量。
        self.dropout = nn.Dropout(config.dropout)  # embedding 后的 dropout。
        # `nn.ModuleList` 是 PyTorch 专用列表；把子模块放进去后，PyTorch 才能正确找到它们的参数。
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])  # 堆叠多个 Transformer block。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 所有 block 之后的最终 RMSNorm。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)  # 预计算所有位置的 RoPE cos/sin。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 注册 cos 表为 buffer；不会训练，保存模型时也不持久化。
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)  # 注册 sin 表为 buffer；移动模型到 GPU 时会一起移动。

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):  # 主模型前向传播，输入 token ids，输出隐藏状态和缓存。
        # `input_ids` 是 tokenizer 之后的整数矩阵，例如一句话会先变成 [1, 245, 89, 2] 这样的 token id。
        batch_size, seq_length = input_ids.shape  # input_ids 形状是 [batch, seq_len]。
        if hasattr(past_key_values, 'layers'): past_key_values = None  # `hasattr(obj, "layers")` 检查对象是否有 layers 属性；这里用于兼容某些新版缓存格式。
        past_key_values = past_key_values or [None] * len(self.layers)  # `[None] * 层数` 会创建一个列表，表示每层都暂无缓存。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 如果有历史缓存，当前位置从历史长度之后开始。
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # token id -> embedding 向量，并做 dropout。
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:  # 某些 meta-device 初始化场景下 buffer 可能丢失，这里检测并重建。
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)  # 重新计算 RoPE cos/sin。
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)  # 把重建后的表移动到当前输入所在设备。
        # `a:b` 是 Python 切片，表示从下标 a 取到 b-1；这里按当前序列位置截取 RoPE 表。
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])  # 截取当前 token 对应的位置编码。
        presents = []  # 用来收集每一层返回的 KV 缓存。
        # `zip(a, b)` 会把两个列表按位置配对，例如第 0 层配第 0 层缓存，第 1 层配第 1 层缓存。
        for layer, past_key_value in zip(self.layers, past_key_values):  # 逐层通过 Transformer block，同时取出对应层的历史缓存。
            hidden_states, present = layer(  # 当前层输出新的 hidden_states 和当前层缓存。
                hidden_states,  # 当前 token 表示，形状 [batch, seq, hidden]。
                position_embeddings,  # 当前序列位置的 RoPE cos/sin。
                past_key_value=past_key_value,  # 当前层历史 key/value，推理加速用。
                use_cache=use_cache,  # 是否返回新的 key/value 缓存。
                attention_mask=attention_mask  # padding mask，避免模型关注 padding token。
            )
            presents.append(present)  # 保存当前层的缓存；如果 use_cache=False，这里会保存 None。
        hidden_states = self.norm(hidden_states)  # 所有层结束后做最终归一化。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())  # 汇总所有 MoE 层的辅助损失；没有 MoE 时为 0。
        return hidden_states, presents, aux_loss  # 返回最终隐藏状态、每层 KV 缓存、MoE 辅助损失。

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):  # 带语言模型头的完整因果语言模型，可用于训练和生成。
    config_class = MiniMindConfig  # 告诉 Hugging Face 这个模型对应的配置类。
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 标记 lm_head 和 embedding 可以共享权重。

    def __init__(self, config: MiniMindConfig = None):  # 初始化完整语言模型；如果没传配置就用默认配置。
        # `config or MiniMindConfig()` 的意思是：如果传入了 config 就用传入的，否则新建默认配置。
        self.config = config or MiniMindConfig()  # 保存配置对象，None 时创建默认 MiniMindConfig。
        super().__init__(self.config)  # 初始化 PreTrainedModel，注册配置和 HF 相关机制。
        self.model = MiniMindModel(self.config)  # 创建 Transformer 主体。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 输出层：把 hidden_size 映射到 vocab_size，得到每个 token 的 logits。
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight  # 如果启用权重共享，让输入词向量和输出分类权重指向同一参数。
        self.post_init()  # Hugging Face 的初始化收尾流程，比如初始化权重。

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):  # 训练/推理前向传播，返回 logits、loss、缓存等。
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)  # 先通过 Transformer 主体得到隐藏状态。
        # `logits_to_keep=0` 时，slice(-0, None) 等价于 slice(0, None)，也就是保留全部位置。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  # 控制只计算最后若干位置的 logits，可节省推理开销。
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # 对选中的位置做词表分类，输出形状 [batch, kept_seq, vocab_size]。
        loss = None  # 默认没有损失；只有传入 labels 时才计算训练 loss。
        if labels is not None:  # 训练时传入 labels，模型会计算 next-token prediction 损失。
            # 因果语言模型的训练目标是“用前面的 token 预测下一个 token”：
            # 如果输入是 [我, 爱, 学习]，第 0 个位置预测“爱”，第 1 个位置预测“学习”。
            # 所以 logits 去掉最后一个位置，labels 去掉第一个位置，让预测和答案错开一位对齐。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()  # x 是第 t 个位置预测，y 是第 t+1 个真实 token。
            # `contiguous()` 是让张量内存变连续，很多 `.view(...)` 操作要求这个条件。
            # `ignore_index=-100` 是 PyTorch/NLP 里常见约定：标签为 -100 的位置不参与 loss。
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)  # 交叉熵损失；label 为 -100 的位置会被忽略。
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)  # 用标准结构返回所有结果。
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()  # 生成时不需要梯度，关闭梯度可节省显存并加速。
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):  # 自定义文本生成函数。
        # 这里的生成是“自回归生成”：模型一次只预测下一个 token，再把这个 token 接回输入，继续预测下一个。
        # 参数直觉：
        # - max_new_tokens：最多生成多少个新 token。
        # - temperature：温度，越低越保守，越高越随机。
        # - top_k：只从概率最高的 k 个候选里选。
        # - top_p：只从累计概率达到 p 的候选集合里选，也叫 nucleus sampling。
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)  # 读取输入 token，并按 num_return_sequences 复制多份。
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None  # 如果有 attention_mask，也同步复制多份。
        past_key_values = kwargs.pop("past_key_values", None)  # 读取外部传入的 KV 缓存；通常第一次生成时为空。
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)  # 标记每条序列是否已经生成 EOS。
        if streamer: streamer.put(input_ids.cpu())  # 如果传入 streamer，先把原始输入 token 推送出去。
        for _ in range(max_new_tokens):  # 最多生成 max_new_tokens 个新 token。
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0  # 如果有缓存，past_len 表示已经缓存的历史 token 数。
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)  # 只把未缓存的新 token 输入模型，节省重复计算。
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None  # 为即将追加的新 token 扩展一位 mask。
            logits = outputs.logits[:, -1, :] / temperature  # `[:, -1, :]` 取每条序列最后一个位置，因为生成时只关心下一个 token。
            if repetition_penalty != 1.0:  # 如果设置了重复惩罚，就降低已经出现过的 token 再次被选中的概率。
                for i in range(input_ids.shape[0]):  # 对 batch 中每条序列分别处理。
                    seen = torch.unique(input_ids[i])  # 找出这条序列里已经出现过的 token。
                    score = logits[i, seen]  # 取出这些已出现 token 当前的 logits。
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)  # 正分数除以惩罚系数，负分数乘以惩罚系数。
            if top_k > 0:  # top-k 采样：只保留概率最高的 k 个 token。
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')  # 低于第 k 大 logit 的候选全部屏蔽。
            if top_p < 1.0:  # top-p 采样：保留累计概率达到 p 的最小候选集合。
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 按 logit 从大到小排序，同时保存原始 token 下标。
                # `torch.softmax` 把 logits 变成概率；`torch.cumsum` 计算累计和，用来找到累计概率超过 top_p 的位置。
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p  # 找出累计概率超过 top_p 的位置。
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0  # 至少保留概率最高的第一个 token，并把截断边界右移一位。
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')  # 把排序后的 mask 映射回原词表位置，并屏蔽被丢弃的 token。
            # `torch.multinomial` 按概率随机抽样；`argmax` 直接选最大概率。前者更有变化，后者更稳定但可能更死板。
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)  # 采样模式随机抽取 token，非采样模式选择最大概率 token。
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)  # 已结束的序列继续填 EOS，保持 batch 长度一致。
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # 把新生成的 token 拼接到序列末尾。
            past_key_values = outputs.past_key_values if use_cache else None  # 更新 KV 缓存，下轮只需计算新 token。
            if streamer: streamer.put(next_token.cpu())  # 如果有 streamer，就把刚生成的新 token 推送出去。
            if eos_token_id is not None:  # 如果设置了 EOS，就检查是否应该停止。
                finished |= next_token.squeeze(-1).eq(eos_token_id)  # 当前生成 EOS 的序列标记为已完成。
                if finished.all(): break  # 如果 batch 内所有序列都结束，提前停止生成。
        if streamer: streamer.end()  # 通知 streamer 生成结束。
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}  # 调试或续写场景下，可同时返回生成结果和 KV 缓存。
        return input_ids  # 默认只返回完整 token 序列。
