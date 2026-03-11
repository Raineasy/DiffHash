from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import torch
from torchvision import transforms

class TextImageHashDataset(Dataset):
    def __init__(self, image_root, annotation_file, transform=None):
        """
        文本-图像对数据集
        
        Parameters
        ----------
        image_root : str
            图像根目录
        annotation_file : str
            包含图像路径和对应文本描述的标注文件路径（JSON格式）
            格式: [{"image": "path/to/image.jpg", "text": "description"}, ...]
        transform : callable, optional
            图像预处理转换
        """
        self.image_root = image_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 加载标注文件
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 加载图像
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # 获取文本描述
        text = ann['text']
        
        return text, image

def create_dataloader(image_root, annotation_file, batch_size=32, num_workers=4):
    """
    创建数据加载器
    """
    dataset = TextImageHashDataset(image_root, annotation_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader 