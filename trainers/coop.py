import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    # "EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def forward_with_ctx(self, ctx):
        # ctx = ctx  # [n_ctx, ctx_dim] 或 [n_cls, n_ctx, ctx_dim]
        if ctx.dim() == 2:
            # 如果ctx是2D，扩展为3D
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # 确保维度匹配
        prefix = self.token_prefix  # [n_cls, 1, dim]
        suffix = self.token_suffix  # [n_cls, *, dim]


        if self.class_token_position == "end":
            # 确保所有张量在第一维（batch/class维度）上匹配
            if ctx.size(0) != prefix.size(0):
                ctx = ctx.expand(prefix.size(0), -1, -1)

            prompts = torch.cat(
                [
                    prefix,  # [n_cls, 1, dim]
                    ctx,  # [n_cls, n_ctx, dim]
                    suffix,  # [n_cls, *, dim]
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class AdaptiveTemperature(nn.Module):
    """
    自适应温度调节器模块，管理温度参数和相关的运行统计量
    仿照BatchNorm实现，包含可学习的缩放和平移参数
    """

    def __init__(self, feature_dim=1, momentum=0.1, eps=1e-5):
        super().__init__()

        self.log_temperature = nn.Parameter(torch.zeros(1))
        # 可学习的缩放和平移参数
        self.weight = nn.Parameter(torch.ones(feature_dim) * 0.3)  # gamma
        self.bias = nn.Parameter(torch.zeros(feature_dim))  # beta

        # self.weight_logits = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        # self.weight_adjustedLogits = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        # 注册缓冲区用于存储运行统计量
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_var', torch.ones(feature_dim))

        # 移动平均动量参数和数值稳定性参数
        self.momentum = momentum
        self.eps = eps

    def forward(self, logits, update_stats=True):
        """
        对logits进行Z-score标准化，然后应用可学习的缩放和平移

        Args:
            logits: 输入的logits张量，形状为[batch_size, num_classes]
            update_stats: 是否更新运行统计量

        Returns:
            标准化并经过缩放和平移后的logits
        """
        # 获取logits的形状信息
        if logits.dim() == 2:
            # 如果是二维张量 [batch_size, num_classes]
            batch_size, num_classes = logits.size()
            # 沿着类别维度计算统计量
            reduce_dim = 1
        else:
            # 处理其他维度情况
            reduce_dim = -1

        if self.training and update_stats:
            # 计算当前batch的统计量
            current_mean = logits.mean(dim=reduce_dim, keepdim=True)
            current_var = logits.var(dim=reduce_dim, keepdim=True)

            # 更新移动平均
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean.mean(0)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var.mean(0)

            # 使用当前batch的统计量进行标准化
            mean = current_mean
            var = current_var
        else:
            # 测试时使用累积的统计量
            mean = self.running_mean.view(1, -1)
            var = self.running_var.view(1, -1)

        # 添加epsilon避免除零
        std = (var + self.eps).sqrt()
        normalized = (logits - mean) / std

        # 应用可学习的缩放和平移参数
        if logits.dim() == 2:
            # 对于二维输入，权重和偏置需要正确广播
            return self.weight.view(1, -1) * normalized + self.bias.view(1, -1)
        else:
            # 对于其他维度，适当调整权重和偏置的形状
            return self.weight * normalized + self.bias



class addNoiseCosineLoss(nn.Module):
    def __init__(self, noise_scale=0.01, top_k_ratio=0.5):
        super().__init__()
        self.noise_scale = noise_scale
        self.top_k_ratio = top_k_ratio

    def add_topk_noise(self, features):
        batch_size, feature_dim = features.shape
        k = int(feature_dim * self.top_k_ratio)

        # 找到top-k索引
        _, top_indices = torch.topk(torch.abs(features), k, dim=1)  # [batch_size, k]

        # 使用gather提取重要特征
        important_features = torch.gather(features, 1, top_indices)  # [batch_size, k]

        # 添加噪声
        noise = torch.randn_like(important_features) * self.noise_scale
        noisy_important = important_features + noise

        # 使用scatter_替换回去
        noisy_features = features.clone()
        noisy_features.scatter_(1, top_indices, noisy_important)

        return noisy_features

    def forward(self, student_features, teacher_features):
        total_loss = 0

        # 原始特征的余弦相似度损失
        cos = F.cosine_similarity(student_features, teacher_features, dim=1)
        loss = 1.0 - torch.mean(cos)
        total_loss += loss

        # 对重要特征添加噪声后的余弦相似度损失
        s_features = self.add_topk_noise(student_features)
        t_features = self.add_topk_noise(teacher_features)
        cos = F.cosine_similarity(s_features, t_features, dim=1)
        loss = 1.0 - torch.mean(cos)
        total_loss += loss

        return total_loss / 2.0


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # 使用改进后的AdaptiveTemperature模块，传入类别数量作为feature_dim

        num_classes = len(classnames)
        self.temp_adjuster = AdaptiveTemperature(feature_dim=num_classes, momentum=0.01)


        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()
        clip_model =clip_model.cuda()

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.original_text_features = text_features


        self.addNoiseCosineLoss= addNoiseCosineLoss(noise_scale=0.0001,top_k_ratio=0.3)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.current_image_features = image_features
        self.current_text_features = text_features

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        return logits

    def get_distill_loss(self, text_features):
        """基于标准化logits的蒸馏损失"""
        # 计算教师模型和学生模型的logits
        image_features = self.current_image_features
        logit_scale = self.logit_scale.exp()

        teacher_text_features = F.normalize(self.original_text_features, dim=-1)
        student_text_features = F.normalize(text_features, dim=-1)

        teacher_logits = logit_scale * image_features @ teacher_text_features.t()
        student_logits = logit_scale * image_features @ student_text_features.t()


        # 对教师和学生的logits进行标准化
        std_teacher_logits = self.temp_adjuster(teacher_logits, update_stats=True)
        std_student_logits = self.temp_adjuster(student_logits, update_stats=True)

        # 计算软标签
        soft_teacher = F.softmax(std_teacher_logits, dim=-1)
        log_soft_student = F.log_softmax(std_student_logits, dim=-1)

        # 计算KL散度损失
        distill_loss = F.kl_div(
            log_soft_student,
            soft_teacher,
            reduction='batchmean'
        )

        distill_loss_kgcoop = self.addNoiseCosineLoss(student_text_features, teacher_text_features)

        kgcoop_weight=0.5
        return (1-kgcoop_weight)*distill_loss + kgcoop_weight*distill_loss_kgcoop



@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "temp_adjuster" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        # 创建包含prompt_learner和temp_adjuster参数的优化器
        params_to_optimize = [
            {'params': self.model.prompt_learner.parameters()},
            {'params': self.model.temp_adjuster.parameters()}
        ]
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim = build_optimizer(params_to_optimize, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("temp_adjuster", self.model.temp_adjuster, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        output = self.model(image)
        text_features = self.model.current_text_features
        distill_loss = self.model.get_distill_loss(text_features)
        loss = F.cross_entropy(output, label)
        distill_loss_weight = 12.0
        loss_total = loss + distill_loss*distill_loss_weight
        if torch.isnan(loss_total):
            print(f"Warning: Loss is NaN! loss={loss}, distill_loss={distill_loss}")

        self.model_backward_and_update(loss_total)

        loss_summary = {
            "loss": loss.item(),
            "distill_loss": distill_loss.item()*distill_loss_weight,
            "loss_total": loss_total.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # 特殊处理temp_adjuster
            if name == "temp_adjuster":
                print(f"正在加载温度调节器参数，处理可能的类别数不匹配...")
                # 获取当前类别数和加载的状态字典中的类别数
                current_classes = self._models[name].weight.size(0)
                loaded_classes = state_dict['weight'].size(0)

                if current_classes != loaded_classes:
                    print(f"类别数不匹配: 当前有 {current_classes} 类, 加载的状态有 {loaded_classes} 类")

                    # 创建新的温度调节器
                    momentum = self._models[name].momentum
                    eps = self._models[name].eps
                    new_temp_adjuster = AdaptiveTemperature(feature_dim=current_classes, momentum=momentum, eps=eps)

                    # 如果当前类别数小于加载的类别数，可以加载部分参数
                    if current_classes < loaded_classes:
                        # 加载前current_classes个参数
                        new_state_dict = {}
                        new_state_dict['weight'] = state_dict['weight'][:current_classes]
                        new_state_dict['bias'] = state_dict['bias'][:current_classes]
                        new_state_dict['running_mean'] = state_dict['running_mean'][:current_classes]
                        new_state_dict['running_var'] = state_dict['running_var'][:current_classes]
                        new_state_dict['log_temperature'] = state_dict['log_temperature']

                        # 加载部分参数
                        new_temp_adjuster.load_state_dict(new_state_dict)
                        print("已加载部分温度调节器参数")

                    # 替换原来的温度调节器
                    self._models[name] = new_temp_adjuster.cuda()
                    self.model.temp_adjuster = new_temp_adjuster.cuda()
                    print("已替换温度调节器")
                    continue

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
