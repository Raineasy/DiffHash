import numpy as np
import torch
from PIL import Image
import cv2
from typing import Tuple
import os

import torch
import torch.nn.functional as F


def aggregate_attention(prompts, attention_store, res: int, from_where, is_cross: bool, select: int, is_cpu=True):
    """
    聚合注意力图，确保所有注意力图具有相同的分辨率。
    """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2

    for location in from_where:
        key = f"{location}_{'cross' if is_cross else 'self'}"
        if key not in attention_maps:
           # print(f"Warning: No attention maps found for key: {key}")
            continue

        for i, item in enumerate(attention_maps[key]):
            # 获取注意力图的实际分辨率
            if item.dim() == 3:  # [batch, pixels, tokens]
                actual_res = int(np.sqrt(item.shape[1]))
                # 重塑注意力图为正方形
                cross_maps = item.reshape(len(prompts), actual_res, actual_res, -1)[select]
            else:  # 如果已经是4D张量 [batch, height, width, tokens]
                actual_res = item.shape[1]
                cross_maps = item[select]

            # 如果分辨率与目标不匹配，则调整
            if actual_res != res:
                # 确保输入是4D张量 [batch, channels, height, width]
                cross_maps = cross_maps.permute(2, 0, 1).unsqueeze(0)
                cross_maps = F.interpolate(cross_maps, size=(res, res), mode="bilinear", align_corners=False)
                cross_maps = cross_maps.squeeze(0).permute(1, 2, 0)
                #print(f"Adjusted attention map {i} from {actual_res}x{actual_res} to {res}x{res}")

            # 确保注意力图形状匹配目标分辨率
            if cross_maps.shape[:2] != (res, res):
                raise RuntimeError(
                    f"Attention map {i} shape mismatch: expected {res}x{res}, got {cross_maps.shape[:2]}."
                )

            # 确保所有注意力图的最后一个维度大小一致
            if cross_maps.shape[-1] != 77:
                #print(f"Adjusting attention map {i} last dimension from {cross_maps.shape[-1]} to 77")
                if cross_maps.shape[-1] < 77:
                    # 如果当前维度小于77，填充0
                    padding = 77 - cross_maps.shape[-1]
                    cross_maps = F.pad(cross_maps, (0, padding))
                else:
                    # 如果当前维度大于77，裁剪
                    cross_maps = cross_maps[:, :, :77]

            out.append(cross_maps)

    if len(out) == 0:
        raise ValueError("No valid attention maps were aggregated. Check input parameters and attention store.")

    # 拼接并平均化
    out = torch.stack(out, dim=0)  # [num_maps, height, width, tokens]
    out = out.mean(0)  # [height, width, tokens]
    return out.cpu() if is_cpu else out


def show_cross_attention(prompts, tokenizer, attention_store, res: int, from_where, select: int = 0, save_path=None):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.detach().cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    view_images(np.stack(images, axis=0), save_path=save_path)


def show_self_attention_comp(prompts, attention_store, res: int, from_where,
                             max_com=7, select: int = 0, save_path=None):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    view_images(np.concatenate(images, axis=1), save_path=save_path)


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, save_path=None, show=False):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if show:
        pil_img.show()
    if save_path is not None:
        pil_img.save(save_path)
        
def load_database_hash(hash_model, database_hash_path=None, database_dataset=None):
    """
    加载数据库的哈希码
    
    Args:
        hash_model: 哈希模型（用于获��设备信息）
        database_hash_path: 预计算的哈希码路径 (.txt文件)
        database_dataset: 包含哈希码的数据集对象
    
    Returns:
        database_hash: 数据库中所有图像的哈希码 [N, hash_size]
    """
    if database_hash_path and os.path.exists(database_hash_path):
        print(f"Loading database hashes from: {database_hash_path}")
        # 读取txt文件中的哈希码
        hash_codes = []
        with open(database_hash_path, 'r') as f:
            for line in f:
                # 假设每行是空格分隔的浮点数
                values = [float(x) for x in line.strip().split()]
                # 转换为tensor
                hash_code = torch.tensor(values, dtype=torch.float)
                # 二值化处理
                hash_code = (hash_code >= 0).float()
                hash_codes.append(hash_code)
        
        # 将所有哈希码堆叠成一个tensor
        database_hash = torch.stack(hash_codes)
        
        # 将0/1转换为-1/1（如果需要）
        database_hash = 2 * database_hash - 1
        
    elif database_dataset is not None:
        print("Loading database hashes from dataset")
        database_hash = database_dataset.get_all_hashes()
    else:
        raise ValueError("Either database_hash_path or database_dataset must be provided")
    
    print(f"Loaded {len(database_hash)} database hashes with shape: {database_hash.shape}")
    return database_hash.cuda()


def check_attention_shape(attention_map, expected_res):
    """检查注意力图的形状是否正确"""
    if attention_map.dim() == 3:  # [batch, pixels, tokens]
        actual_res = int(np.sqrt(attention_map.shape[1]))
        if actual_res ** 2 != attention_map.shape[1]:
            raise ValueError(f"Invalid attention map shape: {attention_map.shape}")
        return actual_res
    elif attention_map.dim() == 4:  # [batch, height, width, tokens]
        return attention_map.shape[1]
    else:
        raise ValueError(f"Unexpected attention map dimensions: {attention_map.dim()}")


# class SmoothAP(torch.nn.Module):
#     def __init__(self, anneal):
#         super(SmoothAP, self).__init__()
#         self.anneal = anneal

#     def sigmoid(self, tensor, temp=0.1):
#         exponent = -tensor / temp
#         exponent = torch.clamp(exponent, min=-50, max=50)
#         y = 1.0 / (1.0 + torch.exp(exponent))
#         return y

#     def hmm(self, x, y, k):
#         """
#         Parameters
#         ----------
#         x : query features [2, 512]
#         y : retrieval set features [2, 77, 1024] 
#         k : feature dimension
#         """
#         # print("Debug - x shape:", x.shape)
#         # print("Debug - y shape:", y.shape)
#         # print("Debug - k:", k)
        
#         # 处理 y 的维度
#         if len(y.shape) == 3:
#             # 将 y 从 [2, 77, 1024] 转换为 [77, 512]
#             y = y.mean(0)  # [77, 1024]
#             y = y @ torch.randn(1024, k).to(y.device)  # [77, 512]
        
#         # 确保 x 是 2D
#         if len(x.shape) == 1:
#             x = x.unsqueeze(0)
        
#         # print("Debug - after processing:")
#         # print("x shape:", x.shape)
#         # print("y shape:", y.shape)
        
#         # 计算相似度矩阵
#         sim = torch.mm(x, y.permute(1, 0))  # [2, 77]
#         return ((k - sim) / 2)

#     def caculate_rank(self, x, y):
#         # 获取特征维度
#         k = x.shape[-1] if len(x.shape) > 1 else len(x)
        
#         # 计算相似度
#         query_sim = self.hmm(x, y, k)  # [2, 77]
        
#         # 获取实际的batch size
#         batch = query_sim.shape[0]
        
#         # 创建适当大小的mask
#         mask = (1 - torch.eye(batch)).cuda()
        
#         # 重塑相似度矩阵以进行排序计算
#         query_D = query_sim.unsqueeze(1) - query_sim.unsqueeze(0)  # [2, 2, 77]
        
#         # 应用sigmoid
#         sim_sg = self.sigmoid(query_D, temp=self.anneal)
        
#         # 应用mask
#         sim_sg = sim_sg * mask.unsqueeze(-1)
        
#         # 计算排名
#         all_rk = torch.sum(sim_sg, dim=1) + 1
        
#         return all_rk

#     def forward(self, query, database, pos_len):
#         """
#         query: [2, 512] 当前特征
#         database: [2, 77, 1024] 目标特征
#         pos_len: 正样本长度
#         """
#         all_rank = self.caculate_rank(query, database)
#         pos_rank = self.caculate_rank(query, database[:pos_len])
        
#         # 确保维度匹���
#         if len(pos_rank) > 0:
#             AP = torch.sum(pos_rank / all_rank[:len(pos_rank)]) / len(pos_rank)
#         else:
#             AP = torch.tensor(0.0).to(query.device)
            
#         return (1 - AP)