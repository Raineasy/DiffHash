import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from attentionControl import AttentionControlEdit
import diff_hashing_attack
from PIL import Image
import numpy as np
import os
from cMap import *
import random
import json
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms

# Argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default="/root/autodl-tmp/DiffAttack/output_16", type=str,
                    help='Where to save the adversarial examples, and other results')
parser.add_argument('--images_root', default="/root/autodl-tmp/DiffAttack/data/NUS-WIDE/", type=str,
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
parser.add_argument('--top_k', type=int, default=5000)
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
# 在 parser 参数中添加数据集相关的参数
parser.add_argument('--dataset', type=str, default='nuswide', choices=['flickr25k', 'mscoco', 'nuswide'],
                    help='Choose dataset from: flickr25k, mscoco, nuswide')

# 为不同数据集添加对应的路径和配置
parser.add_argument('--flickr25k_root', type=str, default='/root/autodl-tmp/DiffAttack/data/FLICKR-25K',
                    help='Root directory for Flickr25K dataset')
parser.add_argument('--mscoco_root', type=str, default='/root/autodl-tmp/DiffAttack/data/MSCOCO',
                    help='Root directory for MSCOCO dataset')
parser.add_argument('--nuswide_root', type=str, default='/root/autodl-tmp/DiffAttack/data/NUS-WIDE',
                    help='Root directory for NUS-WIDE dataset')
parser.add_argument('--database_label_path', type=str, default="/root/autodl-tmp/DiffAttack/data/NUS-WIDE/database_label.txt",
                    help='Path to the database labels')
parser.add_argument('--annotations_file', default="/root/autodl-tmp/DiffAttack/data/NUS-WIDE/annotations_2100_new.json", type=str,
                    help='Path to the annotation file for the original images')
parser.add_argument('--target_annotations_file', default="/root/autodl-tmp/DiffAttack/data/NUS-WIDE/annotations_2100_target.json", type=str,
                    help='Path to the target annotation file containing target descriptions and labels')

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


def run_diffusion_attack(image, prompts, diffusion_model, target_prompts, diffusion_steps, guidance=2.5,
                         start_step=15, iterations=30, args=None, save_path=None):
    """Run diffusion attack and return adversarial image and hash"""
    if isinstance(prompts, list) and len(prompts) > 1:
        # 如果有多个提示词，将它们合并成一个字符串
        prompt = ' '.join(prompts)
        prompts = [prompt]
    
    if isinstance(target_prompts, list) and len(target_prompts) > 1:
        target_prompt = ' '.join(target_prompts)
        target_prompts = [target_prompt]
    
    #print(f"Using source prompt: {prompts}")
    #print(f"Using target prompt: {target_prompts}")
    controller = AttentionControlEdit(diffusion_steps, 1.0, args.res)
    adv_image, adv_hash = diff_hashing_attack.diffattack(
        model=diffusion_model,
        prompts=prompts,
        controller=controller,
        target_prompt=target_prompts,
                                                                  num_inference_steps=diffusion_steps,
                                                                  guidance_scale=guidance,
                                                                  image=image,
        res=args.res,
        model_name=args.model_name,
                                                                  start_step=start_step,
        iterations=iterations,
        args=args,
        save_path=save_path
    )
    return adv_image, adv_hash
def evaluate_transfer(args, adv_images, query_labels, target_labels):
    """评估迁移攻击效果"""
    transfer_results = {}
    
    # 定义要测试的模型
    transfer_models = {
        16: ['DPSH', 'HashNet'],
        32: ['CSQ'],
        64: ['CSQ']
    }
    
    models_to_test = transfer_models.get(args.bit, ['CSQ'])
    
    for model_name in models_to_test:
        try:
            # 加载预训练的二值哈希码和标签
            model_dir = f'/root/autodl-tmp/DiffAttack/save/{model_name}/{args.data}_{args.bit}bits/'
            database_binary = np.load(os.path.join(model_dir, "trn_binary.npy"))
            database_label = np.load(os.path.join(model_dir, "trn_label.npy"))
            
            # 加载模型
            model_path = f'{args.model_path}{args.data}_{model_name}_{args.net}_{args.bit}.pth'
            if not os.path.exists(model_path):
                print(f"Warning: Model {model_path} not found, skipping...")
                continue
                
            print(f"\nTesting transfer attack on {model_name}...")
            
            # 加载迁移模型
            transfer_model = torch.load(model_path).cuda()
            transfer_model.eval()
            
            # 计算对抗样本的哈希码
            adv_binary = []
            with torch.no_grad():
                for adv_img in adv_images:
                    if isinstance(adv_img, np.ndarray):
                        adv_img = torch.from_numpy(adv_img).float().cuda()
                    hash_code = transfer_model(adv_img.unsqueeze(0))
                    binary_code = (torch.sign(hash_code).cpu().numpy() + 1) / 2
                    adv_binary.append(binary_code)
            adv_binary = np.vstack(adv_binary)
            
            # 计算tMap
            tmap = compute_tmap(
                qB=adv_binary,
                database_hash=database_binary,
                queryL=target_labels,
                database_label=database_label,
                topk=args.top_k
            )
            
            # 计算原始图片的tMap
            original_binary = []
            with torch.no_grad():
                for img in args.original_images:
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img).float().cuda()
                    hash_code = transfer_model(img.unsqueeze(0))
                    binary_code = (torch.sign(hash_code).cpu().numpy() + 1) / 2
                    original_binary.append(binary_code)
            original_binary = np.vstack(original_binary)
            
            original_tmap = compute_tmap(
                qB=original_binary,
                database_hash=database_binary,
                queryL=query_labels,
                database_label=database_label,
                topk=args.top_k
            )
            
            transfer_results[model_name] = {
                'adv_tmap': tmap,
                'original_tmap': original_tmap
            }
            
            print(f'{model_name} Transfer Attack Results:')
            print(f'Original T-Map: {original_tmap:.4f}')
            print(f'Adversarial T-Map: {tmap:.4f}')
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return transfer_results

def save_retrieval_results(query_image, retrieved_images, save_path):
    """
    将查询图像和检索到的图像拼接保存
    Args:
        query_image: 查询图像 (PIL Image 或 numpy array)
        retrieved_images: 检索到的图像列表 [PIL Image]
        save_path: 保存路径
    """
    # 确保查询图像是PIL Image格式
    if isinstance(query_image, np.ndarray):
        query_image = Image.fromarray((query_image * 255).astype(np.uint8))
    
    # 统一图像大小
    size = (224, 224)
    query_image = query_image.resize(size)
    retrieved_images = [img.resize(size) for img in retrieved_images]
    
    # 创建拼接画布 (1行6列)
    width = size[0] * 6
    height = size[1]
    canvas = Image.new('RGB', (width, height))
    
    # 放置查询图像
    canvas.paste(query_image, (0, 0))
    
    # 放置检索图像
    for i, img in enumerate(retrieved_images[:5]):  # 只取前5张
        canvas.paste(img, ((i + 1) * size[0], 0))
    
    # 保存结果
    canvas.save(save_path)

def compute_tmap(qB, database_hash, queryL, database_label, topk):
    """
    计算检索性能的tMap指标
    Args:
        qB: 查询哈希码 (n_query, hash_dim)
        database_hash: 数据库哈希码 (n_database, hash_dim)
        queryL: 查询标签 (n_query, n_classes)
        database_label: 数据库标签 (n_database, n_classes)
        topk: 取前k个检索结果计算指标
    """
    print("Calculating tMap...")
    
    # 确保输入是numpy数组
    if isinstance(qB, torch.Tensor):
        qB = qB.cpu().numpy()
    if isinstance(database_hash, torch.Tensor):
        database_hash = database_hash.cpu().numpy()
    if isinstance(queryL, torch.Tensor):
        queryL = queryL.cpu().numpy()
    if isinstance(database_label, torch.Tensor):
        database_label = database_label.cpu().numpy()
    
    # 确保标签是二维的
    if len(queryL.shape) == 1:
        queryL = queryL.reshape(-1, 1)
    if len(database_label.shape) == 1:
        database_label = database_label.reshape(-1, 1)
    
    num_query = qB.shape[0]
    tmap = 0
    retrieved_indices_list = []  # 存储每个查询的检索结果索引
    
    for i in tqdm(range(num_query)):
        # 计算汉明距离
        hamming_dist = CalcHammingDist(qB[i:i+1], database_hash)
        
        # 获取距离最近的topk个样本的索引
        ind = np.argsort(hamming_dist)[0][:topk]
        retrieved_indices_list.append(ind[:5])  # 只保存前5个索引
        
        # 获取这些样本的标签
        retrieved_labels = database_label[ind]
        
        # 计算与查询标签的匹配情况
        matches = np.sum(np.logical_and(queryL[i], retrieved_labels), axis=1) > 0
        
        # 计算相关样本的位置
        positions = np.where(matches)[0] + 1
        
        if len(positions) > 0:
            # 计算平均精度
            precisions = np.arange(1, len(positions) + 1) / positions
            ap = np.mean(precisions)
            tmap += ap
    
    if num_query > 0:
        tmap = tmap / num_query
        
    return tmap, retrieved_indices_list

def main(args):
    # Load invalid images list
    invalid_images = set()
    try:
        with open("invalid_images.txt", 'r') as f:
            invalid_images = set(line.strip() for line in f)
        print(f"Loaded {len(invalid_images)} invalid images to skip")
    except FileNotFoundError:
        print("No invalid_images.txt found, will process all images")

    # Load pretrained diffusion model
    print("Loading pretrained Stable Diffusion model...")
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_diffusion_path, local_files_only=True).to('cuda:0')
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    # Load hash model
    print("Loading hash model...")
    hash_model = torch.load(args.hash_model).to('cuda:0')
    hash_model.eval()

    # Load database hash codes and labels
    print("Loading database hash codes and labels...")
    database_hash = np.loadtxt(args.database_hash_path, dtype=float)
    database_label = np.loadtxt(args.database_label_path, dtype=int)
    database_hash = (database_hash + 1) / 2  # 转换到[0,1]范围

    # Load original and target annotations
    print("Loading annotations...")
    with open(args.annotations_file, 'r') as f:
        all_annotations = json.load(f)
    
    with open(args.target_annotations_file, 'r') as f:
        all_target_annotations = json.load(f)

    # 同时过滤source和target annotations，保持一一对应关系
    filtered_pairs = [
        (source_ann, target_ann) 
        for source_ann, target_ann in zip(all_annotations, all_target_annotations)
        if source_ann['image'] not in invalid_images
    ]
    
    # 解压过滤后的配对
    annotations, target_annotations = zip(*filtered_pairs)
    annotations = list(annotations)
    target_annotations = list(target_annotations)
    
    print(f"Filtered out {len(all_annotations) - len(annotations)} invalid images")
    print(f"Remaining pairs: {len(annotations)}")

    # Initialize storage for adversarial hashes and labels
    qB, qL = [], []
    original_qB = []
    original_qL = []  # 添加原始图片的存储列表

    print("\nStarting adversarial attacks...")
    for idx, annotation in enumerate(tqdm(annotations)):
        # Create unique save directory for this sample
        image_name = os.path.splitext(os.path.basename(annotation['image']))[0]
        sample_save_dir = os.path.join(args.save_dir, f"{image_name}")
        os.makedirs(sample_save_dir, exist_ok=True)

        # Load original image and description
        image_path = annotation['image']
        prompts = annotation['text']
        if isinstance(prompts, str):
            prompts = [prompts]
        image_path = os.path.join(args.images_root, image_path)
        image = Image.open(image_path).convert('RGB')

        # 计算原始图片的hash
        transform = transforms.Compose([
            transforms.Resize((args.res, args.res)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0).to('cuda:0')
        with torch.no_grad():
            original_hash = hash_model(img_tensor)
        original_binary_hash = ((original_hash.cpu() + 1) / 2).numpy()
        original_label = annotation['labels']

        # 保存原始图片的hash和label
        original_qB.append(original_binary_hash)
        original_qL.append(original_label)

        # Load target description and label
        target_prompts = target_annotations[idx]['text']
        if isinstance(target_prompts, str):
            target_prompts = [target_prompts]
        target_label = target_annotations[idx]['labels']

        # Perform adversarial attack
        adv_image, adv_hash = run_diffusion_attack(
            image=image,
            prompts=prompts,
            diffusion_model=ldm_stable,
            target_prompts=target_prompts,
            diffusion_steps=args.diffusion_steps,
            guidance=args.guidance,
            start_step=args.start_step,
            iterations=args.iterations,
            args=args,
            save_path=os.path.join(sample_save_dir, f"{image_name}")
        )
        
        # Convert hash to binary format
        binary_hash = ((adv_hash.cpu() + 1) / 2).numpy()

        # Append adversarial hash and target label
        qB.append(binary_hash)
        qL.append(target_label)

        # 计算并保存原始图片的tMap
        original_tmap, original_retrieved_indices = compute_tmap(
            original_binary_hash.reshape(1, -1),
            database_hash,
            np.array(original_label).reshape(1, -1),
            database_label,
            args.top_k
        )

        # Save individual tMap and retrieval results
        individual_tmap, retrieved_indices = compute_tmap(
            binary_hash.reshape(1, -1), 
            database_hash, 
            np.array(target_label).reshape(1, -1), 
            database_label, 
            args.top_k
        )
        
        # 获取检索到的图像
        retrieved_images = []
        original_retrieved_images = []
        # 读取数据库图像列表
        database_img_file = os.path.join(args.images_root, "database_img.txt")
        if os.path.exists(database_img_file):
            with open(database_img_file, 'r') as f:
                database_img_list = [line.strip() for line in f.readlines()]
            
            # 获取对抗样本的检索结果
            for idx in retrieved_indices[0][:5]:
                if idx < len(database_img_list):
                    img_path = os.path.join(args.images_root, database_img_list[idx])
                    if os.path.exists(img_path):
                        retrieved_images.append(Image.open(img_path).convert('RGB'))
                    else:
                        print(f"Warning: Image not found at {img_path}")
                else:
                    print(f"Warning: Index {idx} out of range")
            
            # 获取原始图片的检索结果
            for idx in original_retrieved_indices[0][:5]:
                if idx < len(database_img_list):
                    img_path = os.path.join(args.images_root, database_img_list[idx])
                    if os.path.exists(img_path):
                        original_retrieved_images.append(Image.open(img_path).convert('RGB'))
                    else:
                        print(f"Warning: Image not found at {img_path}")
                else:
                    print(f"Warning: Index {idx} out of range")
        
        # 保存对抗样本的检索结果
        if retrieved_images:
            # 确保adv_image是PIL Image格式
            if isinstance(adv_image, np.ndarray):
                adv_image_pil = Image.fromarray((adv_image * 255).astype(np.uint8))
        else:
                adv_image_pil = adv_image
                
            save_retrieval_results(
                adv_image_pil,  # 使用转换后的PIL Image
                retrieved_images,
                os.path.join(sample_save_dir, f"{image_name}_adv_retrieval_results.png")
            )
        
        # 保存原始图片的检索结果
        if original_retrieved_images:
            save_retrieval_results(
                image,  # 原始查询图像已经是PIL Image格式
                original_retrieved_images,
                os.path.join(sample_save_dir, f"{image_name}_original_retrieval_results.png")
            )
        
        with open(os.path.join(sample_save_dir, "tmap_result.txt"), 'w') as f:
            f.write(f"Image: {image_name}\n")
            f.write(f"Original text: {prompts}\n")
            f.write(f"Target text: {target_prompts}\n")
            f.write(f"Original labels: {original_label}\n")
            f.write(f"Target labels: {target_label}\n")
            f.write(f"Original tMap (Top-{args.top_k}): {original_tmap:.4f}\n")
            f.write(f"Adversarial tMap (Top-{args.top_k}): {individual_tmap:.4f}\n")

    # Convert lists to numpy arrays
    qB = np.vstack(qB)
    qL = np.array(qL)
    original_qB = np.vstack(original_qB)
    original_qL = np.array(original_qL)

    print("\nComputing overall retrieval metrics...")
    print("Query hash shape:", qB.shape)
    print("Query label shape:", qL.shape)
    print("Database hash shape:", database_hash.shape)
    print("Database label shape:", database_label.shape)

    # Compute overall tMap for both original and adversarial images
    overall_original_tmap, _ = compute_tmap(original_qB, database_hash, original_qL, database_label, args.top_k)
    overall_adv_tmap, _ = compute_tmap(qB, database_hash, qL, database_label, args.top_k)

    # Save overall results
    result_save_path = os.path.join(args.save_dir, "overall_tmap_results.txt")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(result_save_path, 'w') as f:
        f.write("============= Overall Results =============\n")
        f.write(f"Number of processed images: {len(annotations)}\n")
        f.write(f"Overall Original tMap (Top-{args.top_k}): {overall_original_tmap:.4f}\n")
        f.write(f"Overall Adversarial tMap (Top-{args.top_k}): {overall_adv_tmap:.4f}\n")
        f.write("\n============= Detailed Results =============\n")
        
        # 添加每个样本的详细信息
        for idx, (orig_ann, target_ann) in enumerate(zip(annotations, target_annotations)):
            f.write(f"\nSample {idx+1}:\n")
            f.write(f"Image: {os.path.basename(orig_ann['image'])}\n")
            f.write(f"Original text: {orig_ann['text']}\n")
            f.write(f"Target text: {target_ann['text']}\n")
            
            sample_adv_tmap, _ = compute_tmap(
                qB[idx:idx+1],
                database_hash,
                np.array(target_ann['labels']).reshape(1, -1),
                database_label,
                args.top_k
            )
            f.write(f"Adversarial tMap: {sample_adv_tmap:.4f}\n")
            f.write("-" * 50 + "\n")

    print(f"Overall Original tMap (Top-{args.top_k}): {overall_original_tmap:.4f}")
    print(f"Overall Adversarial tMap (Top-{args.top_k}): {overall_adv_tmap:.4f}")
    print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)