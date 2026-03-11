from torch.utils.data import Dataset
import json
import os

class HashingDataset(Dataset):
    def __init__(self, annotations_file, image_root):
        """
        初始化数据集
        Args:
            annotations_file: JSON文件路径，包含图像-文本对的注释
            image_root: 图像根目录
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 处理text字段：统一转换为字符串
        if isinstance(ann['text'], list):
            text = ' '.join(ann['text'])  # 将列表连接成一个字符串
        else:
            text = ann['text']  # 已经是字符串
        
        return {
            'image': os.path.join(self.image_root, ann['image']),
            'text': text,  # 确保返回的是字符串
            'labels': ann['labels']
        }