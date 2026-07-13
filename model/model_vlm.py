import os  # 导入 Python 标准库 os，用来检查文件或目录是否存在。
import torch  # 导入 PyTorch 主库，后面会用它处理张量、关闭梯度、拼接张量等。
import warnings  # 导入 warnings 模块，用来控制 Python 的警告信息显示。
from .model_minimind import *  # 导入同目录下基础语言模型的所有公开名字，例如 MiniMindConfig、MiniMindForCausalLM、MOEFeedForward、F 等。
from typing import Optional, Tuple, List, Union  # 导入类型标注工具；这些标注主要帮助读代码和 IDE 提示，不会改变运行逻辑。
from torch import nn  # 从 torch 中导入 nn 模块；nn 里包含神经网络层，如 Linear、LayerNorm、Module。
from transformers import SiglipImageProcessor, SiglipVisionModel  # 导入 SigLIP 的图片预处理器和视觉模型，用来把图片转成视觉特征。
from transformers.modeling_outputs import MoeCausalLMOutputWithPast  # 导入 Hugging Face 标准输出结构，里面可同时放 loss、logits、缓存等。

warnings.filterwarnings('ignore')  # 关闭警告显示，让运行日志更干净；注意这不会修复问题，只是不把 warning 打出来。


# VLM 是 Vision-Language Model（视觉语言模型）的缩写。
# 这个配置类在 MiniMindConfig 的基础上，额外加入“图片相关”的配置。
class VLMConfig(MiniMindConfig):  # 继承基础 MiniMindConfig，表示它拥有语言模型配置的全部字段，同时增加视觉字段。
    model_type = "minimind-v"  # Hugging Face 用这个字符串识别模型类型；这里的 -v 可以理解成 vision 版本。

    # image_special_token：文本里用来占位图片的位置，例如 "<|image_pad|>"。
    # image_ids：image_special_token 被 tokenizer 编码后对应的 token id 列表；这里默认 [12]。
    # **kwargs：接收其他配置参数，例如 hidden_size、num_hidden_layers、image_hidden_size 等。
    # 小提醒：image_ids=[12] 使用了列表作为默认值，通常 Python 不推荐这样写；但本文件没有修改它，所以不会引入额外影响。
    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):  # 初始化视觉语言模型配置。
        self.image_special_token = image_special_token  # 保存图片占位符字符串；训练数据/提示词里会用它表示“这里有图片特征”。
        self.image_ids = image_ids  # 保存图片占位符对应的 token id；后面会靠这个 id 找到文本序列中应该插入图片向量的位置。
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)  # 视觉编码器输出的特征维度；SigLIP base 常见隐藏维度是 768。
        self.image_token_len = kwargs.get("image_token_len", 64)  # 每张图片最终对应多少个“图片 token”；例如 64 表示一张图占 64 个连续位置。
        super().__init__(**kwargs)  # 调用父类 MiniMindConfig，继续初始化语言模型的配置，如 hidden_size、层数、词表大小等。


# MM 可以理解为 MultiModal（多模态）。
# VisionProjector 的任务：把视觉模型输出的向量维度，变成语言模型内部 hidden_size 的维度。
# 例如视觉特征是 768 维，而语言模型 hidden_size 也是/可能是另一个维度，就需要用投影层对齐。
class MMVisionProjector(nn.Module):  # 继承 nn.Module，表示这是一个可训练的 PyTorch 神经网络模块。
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):  # in_dim 是输入维度，out_dim 是输出维度；后两个参数当前没有实际使用。
        super().__init__()  # 初始化 nn.Module 的内部机制；自定义网络模块里通常都要先调用它。
        self.mlp = nn.Sequential(  # nn.Sequential 会把多个层按顺序串起来，输入会依次经过下面这些层。
            nn.LayerNorm(in_dim),  # 对视觉特征最后一维做 LayerNorm，让输入分布更稳定。
            nn.Linear(in_dim, out_dim),  # 第一层线性变换：把视觉隐藏维度 in_dim 映射到语言模型隐藏维度 out_dim。
            nn.GELU(),  # GELU 是常见激活函数，引入非线性，让投影层不只是简单矩阵乘法。
            nn.Linear(out_dim, out_dim),  # 第二层线性变换：继续在语言模型隐藏空间里调整图片特征。
        )

    def forward(self, x):  # 定义前向传播；x 通常是视觉编码器输出，形状类似 [batch, image_tokens, image_hidden_size]。
        return self.mlp(x)  # 让 x 依次经过 LayerNorm、Linear、GELU、Linear，输出形状最后一维变成 out_dim。


# 继承自语言模型：MiniMindVLM = MiniMindForCausalLM + 视觉编码器 + 视觉投影层。
# 直觉上，它先把文字 token 转成 hidden_states，再把图片特征替换进图片占位符对应的位置，然后继续走语言模型。
class MiniMindVLM(MiniMindForCausalLM):  # 继承 MiniMindForCausalLM，复用原来的 Transformer 主体、lm_head 和生成逻辑。
    config_class = VLMConfig  # 告诉 Hugging Face：这个模型默认使用 VLMConfig 作为配置类。

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):  # 初始化 VLM；vision_model_path 是本地视觉模型目录。
        self.config = config or VLMConfig()  # 如果外部传入 config 就用它；否则创建一份默认 VLMConfig。
        super().__init__(self.config)  # 调用父类 MiniMindForCausalLM，创建语言模型主体 self.model 和输出头 self.lm_head。
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)  # 加载视觉编码器和图片预处理器；失败时会得到 None。
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)  # 创建视觉投影层，把图片特征维度对齐到语言模型 hidden_size。

    @staticmethod  # 静态方法不依赖 self；调用时可以写 MiniMindVLM.get_vision_model(...)。
    def get_vision_model(model_path: str):  # 从本地路径加载 SigLIP 视觉模型和对应的图片处理器。
        from transformers import logging as hf_logging  # 在函数内部导入 transformers 日志工具，避免在文件加载时额外处理日志。
        hf_logging.set_verbosity_error()  # 只显示 error 级别日志，减少加载模型时的普通提示。
        if not os.path.exists(model_path):  # 如果本地模型目录不存在，就无法加载视觉模型。
            return None, None  # 返回两个 None：第一个代表没有视觉编码器，第二个代表没有图片处理器。
        try:  # try/except 用来捕获加载模型时可能出现的错误，避免程序直接崩溃。
            model = SiglipVisionModel.from_pretrained(model_path)  # 从本地目录读取 SigLIP 视觉模型权重和配置。
        except (RuntimeError, ValueError):  # 如果权重格式、配置或运行时状态有问题，from_pretrained 可能抛出这些错误。
            return None, None  # 加载失败时也返回 None，调用方可以据此判断视觉模型不可用。
        processor = SiglipImageProcessor.from_pretrained(model_path)  # 加载图片预处理器，例如 resize、normalize、转 tensor 等规则。
        # 冻结 vision_encoder 的所有参数：训练 VLM 时通常只训练投影层和语言模型，不更新视觉编码器。
        for param in model.parameters():  # 遍历视觉模型里的每一个参数张量。
            param.requires_grad = False  # requires_grad=False 表示反向传播时不为这个参数计算梯度，也不会更新它。
        return model.eval(), processor  # model.eval() 切换到推理模式，关闭 dropout 等训练行为；同时返回预处理器。

    @staticmethod  # 这个方法也不需要访问 self，所以定义成静态方法。
    def image2tensor(image, processor):  # 把一张 PIL 图片转换成视觉模型能接收的张量输入。
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')  # 如果图片带透明通道，就转成普通 RGB；视觉模型通常期望 3 通道输入。
        inputs = processor(images=image, return_tensors="pt")  # 使用 SigLIP 的 processor 做 resize/normalize，并返回 PyTorch tensor 格式。
        return inputs  # 返回通常类似字典的 BatchFeature，例如包含 pixel_values，后续会送进 vision_encoder。

    @staticmethod  # 不依赖实例状态，只依赖传入的 image_inputs 和 vision_model。
    def get_image_embeddings(image_inputs, vision_model):  # 用视觉编码器把图片张量转成图片 token 向量。
        if hasattr(image_inputs, 'keys'):  # 判断 image_inputs 是否像字典一样有 keys；processor 输出通常就是这种结构。
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}  # 去掉多余的第 1 维，避免形状从 [B,1,C,H,W] 变成视觉模型不期望的形式。
        with torch.no_grad():  # 关闭梯度计算；视觉编码器被冻结了，不需要为它保存反向传播中间结果，能省显存。
            if hasattr(image_inputs, 'keys'):  # 如果输入是字典/BatchFeature，就用 ** 展开成关键字参数传给视觉模型。
                outputs = vision_model(**image_inputs, interpolate_pos_encoding=True)  # 前向通过 SigLIP；interpolate_pos_encoding=True 允许位置编码适配不同图片尺寸。
            else:  # 如果输入不是字典，而是直接的 pixel_values 张量，就走这个分支。
                outputs = vision_model(pixel_values=image_inputs, interpolate_pos_encoding=True)  # 显式把张量作为 pixel_values 传给视觉模型。
        return outputs.last_hidden_state  # 返回最后一层视觉隐藏状态，形状通常是 [batch, image_token_len, image_hidden_size]。

    @torch.compiler.disable  # 禁用 torch.compile 对这个函数的编译；这里有 Python 循环和动态拼接，编译收益小且可能不稳定。
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):  # 把投影后的图片特征插入/替换到文本 hidden_states 中。
        if vision_tensors is None or not self.config.image_ids:  # 如果没有图片特征，或配置里没有图片 token id，就不用做任何替换。
            return h  # 直接返回原始文本 hidden_states。
        marker, vf = self.config.image_ids[0], vision_tensors  # marker 是图片占位 token id；vf 是 vision features，也就是图片特征张量。
        if vf.dim() == 3:  # 如果 vf 形状是 [batch, image_tokens, hidden]，说明每个样本只有一张图。
            vf = vf.unsqueeze(1)  # 插入一个“图片数量”维度，变成 [batch, 1, image_tokens, hidden]，方便统一处理单图/多图。
        out = []  # 用来保存每个 batch 样本替换后的 hidden_states。
        for b in range(h.size(0)):  # 遍历 batch 中的每个样本；h.size(0) 就是 batch_size。
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0  # hb 是当前样本的隐藏状态；seq 是 token id 列表；k 是第几张图；i 是当前扫描位置。
            while i < len(seq):  # 从左到右扫描当前文本序列里的 token。
                if seq[i] == marker:  # 如果当前位置是图片占位 token，就说明这里应该放图片特征。
                    start = i  # 记录这一段连续图片占位 token 的起始位置。
                    while i < len(seq) and seq[i] == marker:  # 继续向右找，直到这一段连续图片占位 token 结束。
                        i += 1  # 每遇到一个图片占位 token，就把扫描位置向右移动一格。
                    if k < vf.size(1):  # 确认当前样本还有第 k 张图片特征可用，避免图片数量不够时报错。
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]  # 用第 k 张图片的前 N 个视觉 token 替换这段占位 token，并截断到原序列长度。
                        k += 1  # 当前图片已经用掉，下一段图片占位 token 对应下一张图。
                else:  # 如果当前位置不是图片占位 token，就继续看下一个 token。
                    i += 1  # 普通文本 token 不需要替换，扫描位置右移一格。
            out.append(hb)  # 把当前样本替换完成后的 hidden_states 放入列表。
        return torch.stack(out)  # 把 batch 内所有样本重新堆叠成一个张量，形状回到 [batch, seq_len, hidden_size]。

    def forward(self,  # 定义 VLM 的前向传播；训练和推理都会调用这个函数。
                input_ids: Optional[torch.Tensor] = None,  # 文本 token id，形状通常是 [batch, seq_len]。
                attention_mask: Optional[torch.Tensor] = None,  # 注意力 mask，通常 1 表示真实 token，0 表示 padding。
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # 生成时的 KV 缓存，用来避免重复计算历史 token。
                use_cache: bool = False,  # 是否返回新的 KV 缓存；自回归生成时一般会开启。
                logits_to_keep: Union[int, torch.Tensor] = 0,  # 控制保留哪些位置的 logits；推理时只算最后位置可以省计算。
                labels: Optional[torch.Tensor] = None,  # 训练标签；传入后会计算 next-token prediction 的交叉熵 loss。
                pixel_values: Optional[torch.FloatTensor] = None,  # 图片张量或 processor 输出；没有图片时可以为 None。
                **args):  # 接收额外参数，保持和 Hugging Face/父类接口兼容；本函数内部没有直接使用 args。
        batch_size, seq_length = input_ids.shape  # 读取输入文本形状；batch_size 是样本数，seq_length 是每条样本的 token 数。
        if hasattr(past_key_values, 'layers'): past_key_values = None  # 兼容某些新版缓存对象；如果它有 layers 属性，这里简单地丢弃并重新按列表逻辑处理。
        past_key_values = past_key_values or [None] * len(self.model.layers)  # 如果没有缓存，就为每一层准备一个 None；层数来自语言模型主体。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 如果有缓存，start_pos 是历史序列长度；否则从 0 开始。

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))  # 把 token id 查表成向量，并做 dropout，形状变为 [batch, seq_len, hidden_size]。

        # 只有在“传入了图片”并且“当前是序列开头”时才注入图片特征。
        # start_pos == 0 很重要：生成第 2、3、4... 个 token 时，历史图片特征已经在 KV 缓存里了，不需要重复编码图片。
        if pixel_values is not None and start_pos == 0:  # 判断本次前向是否需要处理图片输入。
            if hasattr(pixel_values, 'keys'):  # 如果 pixel_values 像字典一样，通常说明它来自 processor，例如 {'pixel_values': tensor}。
                sample_val = next(iter(pixel_values.values()))  # 取字典里的第一个张量，用它判断当前图片输入的维度形状。
                if sample_val.ndim == 5:  # 5 维通常表示多图批次，例如 [batch, num_images, channels, height, width]。
                    bs, num = sample_val.shape[:2]  # 读取 batch 大小和每个样本的图片数量。
                    vision_tensors = self.vision_proj(MiniMindVLM.get_image_embeddings({k: v.flatten(0, 1) for k, v in pixel_values.items()}, self.vision_encoder))  # 先把 batch 和图片数量合并，统一送进视觉编码器，再投影到语言 hidden_size。
                    vision_tensors = vision_tensors.view(bs, num, vision_tensors.shape[1], -1)  # 把合并过的维度还原成 [batch, num_images, image_tokens, hidden_size]。
                else:  # 如果不是 5 维，就按普通单图或已整理好的 batch 输入处理。
                    vision_tensors = self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder))  # 提取图片特征并投影，通常得到 [batch, image_tokens, hidden_size]。
            else:  # 如果 pixel_values 不是字典，而是直接的 PyTorch 张量，就走张量形状处理逻辑。
                if len(pixel_values.shape) == 6:  # 6 维时可能多了一层长度为 1 的维度，例如 [batch, num_images, 1, C, H, W]。
                    pixel_values = pixel_values.squeeze(2)  # 去掉第 2 维的单例维度，变成更标准的 [batch, num_images, C, H, W]。
                bs, num, c, im_h, im_w = pixel_values.shape  # 读取图片张量形状；c 是通道数，im_h/im_w 是图片高宽。
                vision_tensors = torch.stack([self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)) for i in range(num)], dim=1)  # 逐张图片提取特征并投影，再按图片数量维度堆叠。
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])  # 把图片特征替换进 <|image_pad|> 对应的 hidden_states 位置。

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        # 这段是容错逻辑：有些 transformers/meta-device 初始化流程可能让 RoPE buffer 变成异常的 0，需要重算。
        if self.model.freqs_cos[0, 0] == 0:  # 检查 RoPE 的 cos 表开头是否异常为 0；正常 cos(0) 应该是 1。
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)  # 重新预计算 RoPE 需要的 cos/sin 表。
            self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)  # 把新表移动到 hidden_states 所在设备，例如 CPU 或 GPU。
        position_embeddings = (  # 准备当前这段序列要用的位置编码。
            self.model.freqs_cos[start_pos:start_pos + seq_length],  # 取从 start_pos 开始、长度为 seq_length 的 cos 表。
            self.model.freqs_sin[start_pos:start_pos + seq_length]  # 取同样位置范围的 sin 表。
        )

        presents = []  # 用来收集每一层 Transformer 返回的 KV 缓存。
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):  # 逐层遍历 Transformer block，并取出对应层的历史缓存。
            hidden_states, present = layer(  # 当前层接收 hidden_states，输出更新后的 hidden_states 和当前层缓存。
                hidden_states,  # 当前 token 表示，形状一般是 [batch, seq_len, hidden_size]。
                position_embeddings,  # 当前序列位置对应的 RoPE cos/sin，用于注意力里的位置编码。
                past_key_value=past_key_value,  # 当前层历史 key/value；没有缓存时为 None。
                use_cache=use_cache,  # 是否让当前层返回新的 key/value 缓存。
                attention_mask=attention_mask  # 注意力 mask，用于屏蔽 padding token。
            )
            presents.append(present)  # 保存当前层的缓存；如果 use_cache=False，present 通常是 None。

        hidden_states = self.model.norm(hidden_states)  # 所有 Transformer 层结束后做最终 RMSNorm，得到更稳定的隐藏状态。

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())  # 如果模型使用 MoE，就汇总每层的路由辅助损失；不用 MoE 时结果是 0。
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # dummy gradient for DDP：乘 0 不改变数值，但让分布式训练知道 vision_proj 参数在计算图里。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  # 如果 logits_to_keep 是整数，就只保留最后若干个位置；如果是张量/切片，则直接使用。
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # 把隐藏状态映射到词表大小，得到每个候选 token 的分数 logits。

        loss = None  # 默认不计算 loss；推理/生成时通常只需要 logits。
        if labels is not None:  # 训练时传入 labels，才计算语言模型的交叉熵损失。
            shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个预测位置；第 t 个位置的输出用来预测第 t+1 个 token。
            shift_labels = labels[..., 1:].contiguous()  # 去掉第一个标签位置，让标签与 shift_logits 错开一位对齐。
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)  # 计算交叉熵；标签为 -100 的位置会被忽略。

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)  # 打包成 Hugging Face 风格输出，方便训练和生成代码读取字段。
        return output  # 返回包含 loss、aux_loss、logits、KV 缓存、隐藏状态的结果对象。

    def generate(self, *args, num_return_sequences=1, **kwargs):  # 包装父类 generate，让多返回序列时图片输入也能同步复制。
        if num_return_sequences > 1 and 'pixel_values' in kwargs:  # 如果一次要生成多条回答，并且输入里包含图片，就需要把图片也复制同样份数。
            pv = kwargs['pixel_values']  # 取出图片输入；可能是字典/BatchFeature，也可能是普通张量。
            if hasattr(pv, 'keys'):  # 如果图片输入是字典，就要对字典里的每个张量分别复制。
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}  # 在 batch 维重复图片张量，其余维度保持不变。
            else:  # 如果图片输入本身就是张量，就直接重复这个张量。
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))  # 只复制 batch 维；例如 [B,C,H,W] 变成 [B*num_return_sequences,C,H,W]。
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)  # 调用父类生成逻辑，真正执行自回归文本生成。
