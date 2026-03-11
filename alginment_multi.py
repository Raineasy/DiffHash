from typing import Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from TGA import *
from HashingDataset import *
import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from attentionControl import AttentionControlEdit
import diff_hashing_attack
from PIL import Image
import numpy as np
import os

import random

import argparse


# Argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default="/root/autodl-tmp/DiffAttack/output", type=str,
                    help='Where to save the adversarial examples, and other results')
parser.add_argument('--images_root', default="/root/autodl-tmp/DiffAttack/imagenet-compatible/images1", type=str,
                    help='The clean images root directory')
parser.add_argument('--prompt_path', default="/root/autodl-tmp/DiffAttack/demo/target.txt", type=str,
                    help='The path to the prompt file containing text descriptions for the attack')
parser.add_argument('--target_path', default="/root/autodl-tmp/DiffAttack/prompts/target_prompts2.txt", type=str,
                    help='The path to the prompt file containing text descriptions for the attack')
parser.add_argument('--is_test', default=False, type=bool,
                    help='Whether to test the robustness of the generated adversarial examples')
parser.add_argument('--pretrained_diffusion_path',
                    default="/root/autodl-tmp/DiffAttack/stable",
                    type=str,
                    help='Change the path to `stabilityai/stable-diffusion-2-base` if want to use the pretrained model')
parser.add_argument('--result_save_dir', default="/root/autodl-tmp/DiffAttack/result_output", type=str,
                    help='Where to save the adversarial examples, and other results')
parser.add_argument('--diffusion_steps', default=20, type=int, help='Total DDIM sampling steps')
parser.add_argument('--start_step', default=15, type=int, help='Which DDIM step to start the attack')
parser.add_argument('--iterations', default=30, type=int, help='Iterations of optimizing the adv_image')
parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')
parser.add_argument('--model_name', default="inception", type=str,
                    help='The surrogate model from which the adversarial examples are crafted')


parser.add_argument('--guidance', default=2.5, type=float, help='guidance scale of diffusion models')
parser.add_argument('--attack_loss_weight', default=20, type=int, help='attack loss weight factor')
parser.add_argument('--cross_attn_loss_weight', default=10000, type=int, help='cross attention loss weight factor')
parser.add_argument('--self_attn_loss_weight', default=100, type=int, help='self attention loss weight factor')
parser.add_argument('--attn_decay_start', default=0.3, type=float,
                    help='When to start decaying attention weights (as fraction of total iterations)')
parser.add_argument('--min_attn_weight', default=0.3, type=float,
                    help='Minimum attention weight after decay')
parser.add_argument('--moving_avg_alpha', default=0.9, type=float,
                    help='Moving average coefficient for loss balancing')
parser.add_argument('--hash_model', type=str, default="/root/autodl-tmp/DiffAttack/hashing_model/NUS-WIDE_CSQ_ResNet50_16.pth" ,
                    help='hash model path')    

parser.add_argument('--retrieval_loss_weight', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--database_hash_path', type=str, default="/root/autodl-tmp/DiffAttack/data_path/database_code_NUS-WIDE_CSQ_ResNet50_16.txt",
                    help='Path to the pre-computed database hashes (.txt file)')
parser.add_argument('--text_to_hash_model_path', type=str, default="/root/autodl-tmp/DiffAttack/best_text_to_hash_model.pth",
                    help='Path to the pre-trained text-to-hash model weights')
parser.add_argument('--similarity_loss_weight', type=float, default=2.0,
                    help='Weight for direct similarity loss with target hash')
parser.add_argument('--pos_size', type=int, default=75,
                    help='Number of positive samples for retrieval')
parser.add_argument('--is_apply_mask', default=False, type=bool,
                    help='Whether to leverage pseudo mask for better imperceptibility (See Appendix D)')
parser.add_argument('--lr', type=float, default=1e-1, help='Initial learning rate')
parser.add_argument('--l2_weight', type=float, default=2.5, help='Weight for L2 distance loss')
parser.add_argument('--adv_weight', type=float, default=0.1, help='Weight for adversarial loss')
parser.add_argument('--quant_weight', type=float, default=1.0, help='Weight for quantization loss')
parser.add_argument('--balance_weight', type=float, default=0.1, help='Weight for balance loss')
parser.add_argument('--hamming_weight', type=float, default=2.5, help='Weight for Hamming distance loss')
parser.add_argument('--initial_lr', type=float, default=1e-2, help='Initial learning rate')
parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate')
parser.add_argument('--binary_weight', type=float, default=2.0, help='Weight for binary hash loss')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
parser.add_argument('--patch_size', type=int, default=32,
                    help='Size of patches for region importance analysis')
parser.add_argument('--hash_weight', type=float, default=1.0,
                    help='Weight for hash loss')
parser.add_argument('--recon_weight', type=float, default=0.5,
                    help='Weight for reconstruction loss')
parser.add_argument('--attn_weight', type=float, default=0.3,
                    help='Weight for attention loss')
parser.add_argument('--save_frequency', type=int, default=10,
                    help='Frequency of saving intermediate results')

def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(42)


def run_diffusion_attack1(image, prompts, diffusion_model, target_prompts, diffusion_steps, guidance=2.5,
                         self_replace_steps=1., save_dir=r"C:\Users\PC\Desktop\output", res=224,
                         model_name="inception", start_step=15, iterations=30, args=None):
    
    # 确保 prompts 是列表类型
    if isinstance(prompts, str):
        prompts = [prompts]
    
    controller = AttentionControlEdit(diffusion_steps, self_replace_steps, args.res)

    adv_image = diffattack1(model=diffusion_model, prompts=prompts, controller=controller,target_prompt=target_prompts,
                                                                  num_inference_steps=diffusion_steps,
                                                                  guidance_scale=guidance,
                                                                  image=image,
                                                                  save_path=save_dir, res=res, model_name=model_name,
                                                                  start_step=start_step,
                                                                  iterations=iterations, args=args)


    return adv_image
def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0



def diffattack1(
        model,
        prompts,
        controller,
        target_prompt,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        image=None,
        model_name="inception",
        save_path=r"/root/autodl-tmp/DiffAttack/output/123",
        res=224,
        start_step=15,
        iterations=30,
        verbose=True,
        topN=1,
        args=None
):
    
    print(f'prompt presentation: {prompts}')

    #print("prompts:", prompts.shape)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

   

    hash_model = torch.load(args.hash_model)
    hash_model = hash_model.cuda()
    hash_model.eval()
    dataset = HashingDataset(annotations_file=r"/root/autodl-tmp/DiffAttack/data/NUS-WIDE/annotations_5000_train1.json", image_root=r"/root/autodl-tmp/DiffAttack/data/NUS-WIDE")
    
    text_guided_attack = TextGuidedAttack(
        hash_model=hash_model,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        hash_size=16
    ).to(model.device)  

    # 训练模型
    print("Starting training text-to-hash mapping...")
    train_text_to_hash_mapping(
        text_guided_attack.text_to_hash,
        hash_model,
        model.text_encoder,
        model.tokenizer,
        dataset
    )

    # 加载刚刚训练好的模型
    trained_model_path = 'best_text_to_hash_model_CSQ_NUS-WIDE_5000_16.pth'
    if os.path.exists(trained_model_path):
        print(f"Loading newly trained model from {trained_model_path}")
        text_guided_attack.text_to_hash.load_state_dict(
            torch.load(trained_model_path)
        )
    else:
        print(f"Warning: Could not find newly trained model at {trained_model_path}")
        # 如果找不到新训练的模型，尝试加载预训练模型
        if os.path.exists(args.text_to_hash_model_path):
            print(f"Loading pre-trained model from {args.text_to_hash_model_path}")
            text_guided_attack.text_to_hash.load_state_dict(
                torch.load(args.text_to_hash_model_path)
            )
        else:
            print("Warning: No model found, using initialized weights")

    # 打印模型状态确认
    print("Current model state:")
    for name, param in text_guided_attack.text_to_hash.named_parameters():
        print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

if __name__ == "__main__":
    args = parser.parse_args()
    assert args.res % 32 == 0 and args.res >= 96, "Please ensure the input resolution be a multiple of 32 and also >= 96."

    guidance = args.guidance
    diffusion_steps = args.diffusion_steps  # Total DDIM sampling steps.
    start_step = args.start_step  # Which DDIM step to start the attack.
    iterations = args.iterations  # Iterations of optimizing the adv_image.
    res = args.res  # Input image resized resolution.
    model_name = args.model_name  # The surrogate model from which the adversarial examples are crafted.

    # Load pretrained diffusion model
    pretrained_diffusion_path = args.pretrained_diffusion_path
    ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path, local_files_only=True).to('cuda:0')
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    ###############################################################我改改改改##########################################################################################
    # Load the clip model which belong to SD
    print("Loaded the pretrained clip belong to SD to get the semantic guidance")
    text_encoder = ldm_stable.text_encoder
    tokenizer = ldm_stable.tokenizer

    prompts_path = args.prompt_path
    with open(prompts_path, 'r') as f:
        description = [line.strip().strip('"') for line in f.readlines()]
    print("Read the prompts of the original attack image:")
    print(description)


    target_path = args.target_path
    with open(target_path, 'r')as f:
        target_description = [line.strip().strip('"') for line in f.readlines()]
    print("Read the prompts of the target attack image:")
    print(target_description)



 ###############################################################我改改改改##########################################################################################
    

    # Read prompt file
    # with open(args.prompt_path, 'r') as file:
    #     prompt = file.read().strip()
    # print(f"Prompt: {prompt}")
    image_path = '/root/autodl-tmp/DiffAttack/demo/0558_132524885.jpg'
    tmp_image = Image.open(image_path).convert('RGB')
    tmp_image.save(os.path.join(args.save_dir, str(1).rjust(4, '0') + "_originImage.png"))

    _ = run_diffusion_attack1(tmp_image, description,
                                            ldm_stable,
                                            target_description,
                                            diffusion_steps,
                                            guidance=guidance,
                                            res=res,
                                            model_name=model_name,
                                            start_step=start_step,
                                            iterations=iterations,
                                            save_dir=os.path.join(args.save_dir, str(2).rjust(4, '0')),
                                            args=args)