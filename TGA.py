from typing import Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from HashingDataset import HashingDataset

class TextGuidedAttack(torch.nn.Module):
    def __init__(self, hash_model, text_encoder, tokenizer, hash_size=16, temperature=0.1):
        """
        Parameters
        ----------
        hash_model : 哈希检索模型路径或模型对象
        text_encoder : 文本编码器 (CLIP或Stable Diffusion的text encoder)
        tokenizer : 分词器
        hash_size : 哈希码长度
        temperature : 温度参数
        """
        super(TextGuidedAttack, self).__init__()
        
        # 如果hash_model是字符串(路径)，则加载模型
        if isinstance(hash_model, str):
            print(f"Loading hash model from {hash_model}")
            self.hash_model = torch.load(hash_model)
        else:
            self.hash_model = hash_model
            
        # 确保模型在正确的设备上
        self.hash_model = self.hash_model.cuda()
        self.hash_model.eval()  # 设置为评估模式

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.hash_size = hash_size
        self.temperature = temperature
        
        # 添加文本到哈希空间的映射层
        self.text_to_hash = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, hash_size),
            nn.Tanh()
        ).cuda()
        
    def get_text_hash(self, text_description):
        """
        将文本描述转换为目标哈希码
        """
        # 如果输入是单个字符串，转换为列表
        if isinstance(text_description, str):
            text_description = [text_description]
        
        tokens = self.tokenizer(
            text_description,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.text_encoder.device)
        
        with torch.no_grad():
            text_features = self.text_encoder(tokens.input_ids)[0].mean(dim=1)
        
        text_hash = self.text_to_hash(text_features)
        return text_hash
    
    def compute_hash_loss(self, image, target_text):
        """
        计算基于哈希的对抗损失
        """
        # 确保输入图像是3通道的
        if image.shape[1] == 4:
            image = image[:, :3, :, :]
        
        # 确保模型处于评估模式
        self.hash_model.eval()
        
        with torch.no_grad():
            # 获取图像的哈希码
            image_hash = self.hash_model(image)  # [batch_size, hash_size]
            
            # 如果target_text是字符串，换为列表
            if isinstance(target_text, str):
                target_text = [target_text] * image.shape[0]
            
            # 获取目标文本的哈希码
            target_hash = self.get_text_hash(target_text)  # [batch_size, hash_size]
            
            # 确保批次大小匹配
            if target_hash.shape[0] != image_hash.shape[0]:
                target_hash = target_hash[:image_hash.shape[0]]
        
        # 计算汉明距离损失
        hamming_distance = torch.abs(image_hash - target_hash).mean()
        
        # 计算方向一致性损失
        direction_loss = 1 - F.cosine_similarity(image_hash, target_hash).mean()
        
        return hamming_distance, direction_loss
    
    def forward(self, image, target_text, original_image=None):
        """计算总的攻击损失"""
        # 确保输入图像是3通道的
        if image.shape[1] == 4:
            image = image[:, :3, :, :]
    
        # 1. 哈希空间对齐损失
        hamming_loss, direction_loss = self.compute_hash_loss(image, target_text)
    
        # 2. 特征空间对齐损失
        with torch.no_grad():
            # 使用模型的特征提取层获取图像特征
            x = image
            for layer in self.hash_model.feature_layers:
                x = layer(x)
            image_features = x.view(x.size(0), -1)  # [batch_size, 2048]
            
            # 获取文本特征
            if isinstance(target_text, str):
                target_text = [target_text] * image.shape[0]
            elif isinstance(target_text, list) and len(target_text) != image.shape[0]:
                if len(target_text) > image.shape[0]:
                    target_text = target_text[:image.shape[0]]
                else:
                    target_text = target_text * (image.shape[0] // len(target_text)) + \
                                 target_text[:(image.shape[0] % len(target_text))]
            
            tokens = self.tokenizer(
                target_text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.text_encoder.device)
            
            text_features = self.text_encoder(tokens.input_ids)[0]
            text_features = text_features.mean(dim=1)
            
            # 使用实例级的text_projection而不是每次创建新的
            if not hasattr(self, 'text_projection'):
                self.text_projection = nn.Linear(1024, 2048).to(image_features.device)
            text_features = self.text_projection(text_features)
            
            # 确保批次大小匹配
            if text_features.shape[0] != image_features.shape[0]:
                if text_features.shape[0] > image_features.shape[0]:
                    text_features = text_features[:image_features.shape[0]]
                else:
                    text_features = text_features.repeat(image_features.shape[0] // text_features.shape[0] + 1, 1)
                    text_features = text_features[:image_features.shape[0]]
        
        # 计算特征空间的损失，使用余弦相似度
        feature_sim = F.cosine_similarity(image_features, text_features).mean()
        feature_loss = 1 - feature_sim
        
        # 3. 扰动约束
        perturbation_loss = 0
        if original_image is not None:
            if original_image.shape[1] == 4:
                original_image = original_image[:, :3, :, :]
            # 使用L2范数和相对扰动
            perturbation = torch.norm(image - original_image, p=2, dim=(1,2,3))
            perturbation_loss = torch.mean(perturbation) / torch.norm(original_image, p=2, dim=(1,2,3)).mean()
        
        # 4. 添加图像质量损失
        smoothness_loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
                         torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        
        # 调整损失权重
        total_loss = (
            hamming_loss * 2.0 +          # 增加哈希损失的权重
            direction_loss * 1.0 +         # 保持方向损失不变
            feature_loss * 0.5 +           # 降低特征损失的权重
            perturbation_loss * 0.3 +      # 增加扰动约束的权重
            smoothness_loss * 0.1          # 添加平滑损失
        )
        
        # 移除梯度裁剪部分，因为它会导致第二次反向传播
        # 改为返回梯度范数，让外部函数处理梯度裁剪
        grad_norm = None
        if total_loss.requires_grad:
            grads = torch.autograd.grad(total_loss, image, create_graph=True, retain_graph=True)[0]
            grad_norm = torch.norm(grads)
        
        return total_loss, {
            'hamming_loss': hamming_loss.item(),
            'direction_loss': direction_loss.item(),
            'feature_loss': feature_loss.item(),
            'perturbation_loss': perturbation_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'grad_norm': grad_norm.item() if grad_norm is not None else None
        }

def train_text_to_hash_mapping(text_to_hash_model, hash_model, text_encoder, tokenizer, dataset, num_epochs=80):
    """
    训练文本到哈希空间的映射，使用汉明距离作为主要损失
    """
    # 1. 优化器和学习率设置
    optimizer = optim.AdamW(text_to_hash_model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    def compute_hamming_distance(x, y):
        """计算批次间的汉明距离"""
        # 将连续值转换为二值
        x_binary = (x > 0).float()
        y_binary = (y > 0).float()
        # 计算汉明距离
        return torch.mean(torch.abs(x_binary - y_binary))
    
    for epoch in range(num_epochs):
        total_loss = 0
        text_to_hash_model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            texts = batch['text']
            image_paths = batch['image']
            
            # 图像处理
            images = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
                    ])(img)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
                    
            if not images:
                continue
                
            images = torch.stack(images).cuda()
            
            # 文本处理
            tokens = tokenizer(
                texts, 
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(text_encoder.device)
            
            with torch.no_grad():
                text_features = text_encoder(tokens.input_ids)[0]
                text_features = text_features.mean(dim=1)
                target_hash = hash_model(images)
            
            # 前向传播
            predicted_hash = text_to_hash_model(text_features)
            
            # 计算损失
            # 1. 汉明距离损失
            hamming_loss = compute_hamming_distance(predicted_hash, target_hash)
            
            # 2. 方向一致性损失（余弦相似度）
            direction_loss = 1 - F.cosine_similarity(predicted_hash, target_hash).mean()
            
            # 3. 量化损失（鼓励输出接近二值）
            quantization_loss = torch.mean(torch.abs(torch.abs(predicted_hash) - 1.0))
            
            # 4. 平衡性损失（鼓励0和1的平衡）
            balance_loss = torch.abs(predicted_hash.mean())
            
            # 总损失
            loss = (hamming_loss * 2.0 +           # 主要关注汉明距离
                   direction_loss * 0.5 +          # 保持方向一致性
                   quantization_loss * 0.3 +       # 鼓励二值化
                   balance_loss * 0.2)             # 保持平衡
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_to_hash_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 损失统计
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
                      f'Total Loss: {avg_loss:.4f}\n'
                      f'Hamming Loss: {hamming_loss.item():.4f}, '
                      f'Direction Loss: {direction_loss.item():.4f}\n'
                      f'Quantization Loss: {quantization_loss.item():.4f}, '
                      f'Balance Loss: {balance_loss.item():.4f}')
        
        # 每轮结束后的处理
        avg_epoch_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}')
        
        scheduler.step()
        
        # 早停和模型保存
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(text_to_hash_model.state_dict(), 'best_text_to_hash_model_CSQ_NUS-WIDE_5000_16.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

