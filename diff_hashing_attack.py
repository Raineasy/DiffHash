from typing import Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, load_database_hash, aggregate_attention, show_cross_attention
from distances import LpDistance
import torch.nn.functional as F
import torch.nn as nn
from TGA import *
from HashingDataset import *
import os
import lpips
from pytorch_msssim import ssim

def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================
    """
    batch_size = 1

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    def ca_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor

            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_)
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image
    
def soft_sign(x, temperature=1.0):
    return torch.tanh(x / temperature)


def compute_hash_distance(hash1, hash2):
    """计算两个哈希码之间的距离"""
    return torch.sum(torch.abs(hash1 - hash2))

def compute_hash_loss(current_hash, target_hash):
    # 二值化目标哈希
    target_binary = torch.sign(target_hash)
    
    # 直接最小化与目标二值哈希的距离
    binary_loss = F.mse_loss(current_hash, target_binary)
    
    # 添加二值化约束
    quant_loss = torch.mean(torch.abs(torch.abs(current_hash) - 1))
    
    return binary_loss, quant_loss

def diffattack(
    model,
    prompts,
    controller,
    target_prompt,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    image=None,
    model_name="inception",
    save_path=None,
    res=224,
    start_step=15,
    iterations=30,
    verbose=True,
    topN=1,
    args=None
):
    print(f'prompt presentation: {prompts}')

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    height = width = res

    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    target_vector = " ".join([target_prompt[i] for i in range(0, topN)])
    print("Generated prompt vector: ", target_vector)

    # ---------------- 逆向扩散，获取初始 latent ---------------- #
    latent, inversion_latents = ddim_reverse_sample(
        image, prompts, model,
        num_inference_steps,
        0, 
        res=height
    )
    inversion_latents = inversion_latents[::-1]

    init_prompt = [prompts[0]]
    batch_size = len(init_prompt)
    latent = inversion_latents[start_step - 1]

    # ========== 准备 无条件 和 有条件 的文本嵌入 ========== #
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        init_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)

    # ------------------- 优化 uncond_embeddings ------------------- #
    uncond_embeddings.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeddings], lr=5e-2)  # 可适当再提高
    loss_func = torch.nn.MSELoss()

    context = torch.cat([uncond_embeddings, text_embeddings])

    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
        for _ in range(10 + 2 * ind):
            out_latents = diffusion_step(model, latents, context, t, guidance_scale)
            optimizer.zero_grad()
            loss = loss_func(out_latents, inversion_latents[start_step - 1 + ind + 1])
            loss.backward()
            optimizer.step()
            context = torch.cat([uncond_embeddings, text_embeddings])

        with torch.no_grad():
            latents = diffusion_step(model, latents, context, t, guidance_scale).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    uncond_embeddings.requires_grad_(False)

    # ------------------- 注册注意力控制 ------------------- #
    register_attention_control(model, controller)

    batch_size = 3
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    context = [torch.cat(i) for i in context]

    original_latent = latent.clone()
    latent.requires_grad_(True)

    init_image = preprocess(image, res)
    
    apply_mask = args.is_apply_mask
    if apply_mask:
        init_mask = None
    else:
        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    # 加载哈希模型
    hash_model = torch.load(args.hash_model)
    hash_model = hash_model.cuda()
    hash_model.eval()

    # 如果有训练过的 text_to_hash 模型则加载
    text_guided_attack = TextGuidedAttack(
        hash_model=hash_model,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        hash_size = 16
    ).to(model.device)  

    print("Starting training text-to-hash mapping...")
    trained_model_path = 'best_text_to_hash_model_CSQ_NUS-WIDE_5000_16.pth'
    if os.path.exists(trained_model_path):
        print(f"Loading newly trained model from {trained_model_path}")
        text_guided_attack.text_to_hash.load_state_dict(
            torch.load(trained_model_path)
        )
    else:
        print(f"Warning: Could not find newly trained model at {trained_model_path}")
        if os.path.exists(args.text_to_hash_model_path):
            print(f"Loading pre-trained model from {args.text_to_hash_model_path}")
            text_guided_attack.text_to_hash.load_state_dict(
                torch.load(args.text_to_hash_model_path)
            )
        else:
            print("Warning: No model found, using initialized weights")

    # 初始化优化器
    optimizer = optim.AdamW([latent], lr=1e-3)
    
    # 记录最佳的hamming loss用于判断是否开始图像修复
    best_hamming_loss = float('inf')
    hamming_threshold = 0.4  # 设置阈值
    
    # 标记是否进入图像修复阶段
    in_restoration_phase = False
    
    for iter_idx in range(iterations):
        controller.loss = 0
        controller.reset()
        optimizer.zero_grad()

        # 动态调整权重和学习率
        if in_restoration_phase:
            # 已经达到目标hamming loss，专注于图像修复
            attack_weight = 2     # 大幅降低攻击权重
            image_quality_weight = 200.0  # 大幅提高图像质量权重
            ssim_weight = 100.0
            tv_weight = 50.0
            current_lr = 5e-3       # 使用较小的学习率避免破坏已达到的攻击效果
            print("进入修复阶段")
        else:
            # 还未达到目标，保持攻击为主
            attack_weight = 15.0
            image_quality_weight = 1.0
            ssim_weight = 0.5
            tv_weight = 0.1
            current_lr = 5e-2

        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # 扩散前向过程
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        # 计算注意力
        before_attention_map = aggregate_attention(
            prompts, controller, args.res // 32, ("up", "down"), True, 0, is_cpu=False
        )
        after_attention_map = aggregate_attention(
            prompts, controller, args.res // 32, ("up", "down"), True, 0, is_cpu=False
        )
        sequence_length = before_attention_map.size(-1)
        before_true_label_attention_map = before_attention_map[:, :, 1: sequence_length - 1]
        after_true_label_attention_map = after_attention_map[:, :, 1: sequence_length - 1]

        # 目标文本向量
        target_vec_str = " ".join([target_prompt[i] for i in range(0, topN)])
        text_input = model.tokenizer(
            [target_vec_str],
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        target_vector = model.text_encoder(text_input.input_ids)[0]

        # 解码图像
        out_latents = model.vae.decode(1 / 0.18215 * latents)['sample'][1:]
        out_image = (out_latents / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image.sub(mean).div(std)  # 预处理
        out_image = out_image.permute(0, 3, 1, 2)

        current_hash = hash_model(out_image)
        target_hash = text_guided_attack.text_to_hash(target_vector)
        if len(target_hash.shape) > 2:
            target_hash = target_hash.squeeze(0)

        # 计算各种损失
        hash_distance = F.mse_loss(current_hash, target_hash)
        cos_sim = F.cosine_similarity(current_hash, target_hash, dim=1).mean()
        similarity_loss = 1 - cos_sim
        binary_loss = torch.mean(torch.abs(torch.abs(current_hash) - 1))

        # Hamming Loss
        binary_current = torch.sign(current_hash)
        binary_target = torch.sign(target_hash)
        hamming_loss = torch.mean(torch.abs(binary_current - binary_target))

        # =========== 额外示例：bitwise hinge-like 惩罚(可选) =========== #
        # 若二者符号不一致，则惩罚。可加 margin(1.0)等。越小越好
        # hinge_bit_loss = mean( max( 0, margin - (binary_current * binary_target) ) )
        # 当二者同号时 (binary_current * binary_target = +1)，无需惩罚
        # 当异号时 (binary_current * binary_target = -1)，loss = margin - (-1) = margin+1
        # 如果希望对错误 bit 做强烈惩罚，可把 margin设更大或乘系数。
        margin = 1.0
        bit_prod = binary_current * binary_target  # 同号 = +1, 异号 = -1
        hinge_mask = (margin - bit_prod).clamp(min=0)  # max(0, margin - prod)
        hinge_bit_loss = hinge_mask.mean()
                # 更新最佳hamming loss并检查是否进入修复阶段
        if hamming_loss.item() < best_hamming_loss:
            best_hamming_loss = hamming_loss.item()
            
        if best_hamming_loss < hamming_threshold and not in_restoration_phase:
            print(f"\n[!] Achieved target hamming loss ({best_hamming_loss:.4f}), switching to restoration phase")
            in_restoration_phase = True
            
        # 图像质量损失
        l2_loss_val = F.mse_loss(out_image, init_image)
        ssim_loss_val = 1 - ssim(out_image, init_image)
        
        def total_variation_loss(image):
            diff_h = image[:, :, :, :-1] - image[:, :, :, 1:]
            diff_v = image[:, :, :-1, :] - image[:, :, 1:, :]
            return torch.mean(diff_h**2) + torch.mean(diff_v**2)

        tv_loss_val = total_variation_loss(out_latents)

        # 综合损失
        # 在这里把 hinge_bit_loss 也与 attack_weight 相乘，以强化单 bit 的惩罚
        loss = (
            0*attack_weight * (
                hash_distance * args.l2_weight +
                similarity_loss * args.similarity_loss_weight +
                binary_loss * args.quant_weight +
                hamming_loss * args.hamming_weight+
                hinge_bit_loss * 15
            ) +
            image_quality_weight * l2_loss_val +
            ssim_weight * ssim_loss_val +
            tv_weight * tv_loss_val +
            controller.loss * (args.self_attn_loss_weight * (2 if in_restoration_phase else 25)) +
            after_true_label_attention_map.var() * (args.cross_attn_loss_weight * (2 if in_restoration_phase else 8))
        )

        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_([latent], max_norm=1.0)
        optimizer.step()

        # 打印详细信息
        if verbose:
            phase_str = "RESTORATION" if in_restoration_phase else "ATTACK"
            print(f"\n[Iter {iter_idx+1}/{iterations}] [{phase_str}]"
                  f" Loss: {loss.item():.4f} |"
                  f" Hamming: {hamming_loss.item():.4f} |"
                  f" Best Hamming: {best_hamming_loss:.4f} |"
                  f" HashDist: {hash_distance.item():.4f}")
            print(f" L2: {l2_loss_val.item():.4f} | SSIM: {ssim_loss_val.item():.4f} |"
                  f" TV: {tv_loss_val.item():.4f}")

    # 最终一次前向，得到最后的图像
    with torch.no_grad():
        controller.loss = 0
        controller.reset()
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

    print("Attack finished. Please check final image and hash distance.")


    out_image = model.vae.decode(1 / 0.18215 * latents.detach())['sample'][1:] * init_mask + (
            1 - init_mask) * init_image
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)

    """
            ==========================================
            ============= Visualization ==============
            ==========================================
    """

    # Decode the final latent into an image
    image = latent2image(model.vae, latents.detach())

    # Preprocess the original image and perturbed image for visualization
    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    perturbed = image[1:].astype(np.float32) / 255 * init_mask.squeeze().unsqueeze(-1).cpu().numpy() + \
                (1 - init_mask.squeeze().unsqueeze(-1).cpu().numpy()) * real
    perturbed_image = (perturbed * 255).astype(np.uint8)

    # 修改保存图像的部分
    if save_path:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        # 保存对比图像（原始 vs 扰动）
        view_images(np.concatenate([real, perturbed]) * 255, show=False,
                    save_path=f"{save_path}_comparison.png")
        
        # 保存对抗样本
        view_images(perturbed_image, show=False, 
                    save_path=f"{save_path}_adversarial.png")
        
        # 保存相对差异图
        diff = perturbed - real
        diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255
        view_images(diff.clip(0, 255), show=False,
                    save_path=f"{save_path}_diff_relative.png")
        
        # 保存绝对差异图
        diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
        view_images(diff.clip(0, 255), show=False,
                    save_path=f"{save_path}_diff_absolute.png")

    # Assuming you have a reset_attention_control function to reset attention mechanisms
    reset_attention_control(model)

    if current_hash.shape != target_hash.shape:
        print(f"Shape mismatch: current_hash {current_hash.shape}, target_hash {target_hash.shape}")
    
    # 在保存最终的二值化哈希码之前，确保目录存在
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        # 确保图像格式正确
        if not isinstance(out_image, torch.Tensor):
            out_image = torch.tensor(out_image).to(model.device)
        if len(out_image.shape) == 3:
            out_image = out_image.unsqueeze(0)
        
        # 计算最终的哈希码
        final_hash = hash_model(out_image)
        final_binary = torch.sign(final_hash)
        print("Final binary hash:", final_binary.cpu().numpy())
        
        # 确保保存的数据是有效的
        if final_binary is not None and final_binary.numel() > 0:
            try:
                # 确保目录存在
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                
                # 修改保存路径，确保与加载时的路径一致
                save_path_hash = os.path.join(save_dir, "adversarial_retrieval", "final_hash.pt")
                os.makedirs(os.path.dirname(save_path_hash), exist_ok=True)
                
                # 保存哈希码
                binary_hash = ((final_binary + 1) / 2).cpu().numpy().astype(np.int32).flatten()
                torch.save(binary_hash, save_path_hash)
                
                print(f"Successfully saved hash to {save_path_hash}")
                print(f"Saved hash value: {binary_hash}")
                
                # 验证保存的文件
                if os.path.exists(save_path_hash) and os.path.getsize(save_path_hash) > 0:
                    loaded_hash = torch.load(save_path_hash)
                    print(f"Verified saved hash: {loaded_hash}")
                else:
                    print(f"Warning: Hash file verification failed")
            except Exception as e:
                print(f"Error saving hash: {e}")
        else:
            print("Error: Final binary hash is empty or None.")

    return perturbed_image[0], final_binary.cpu()
