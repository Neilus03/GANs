from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms

class EGD_GAN_Dataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.file_list = []
        self.label_list = []
        
        for label in os.listdir(root_folder):
            label_path = os.path.join(root_folder, label)
            if not os.path.isdir(label_path):
                continue
            
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                if not os.path.isfile(file_path):
                    continue
                
                self.file_list.append(file_path)
                try:
                    self.label_list.append(int(label))
                except ValueError:
                    print(f"Skipping folder {label} as it cannot be converted to an integer label.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path)

        label = torch.tensor(self.label_list[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = EGD_GAN_Dataset(root_folder="/home/ndelafuente/CVC/EGD_Barcelona/GANs/ACA-GAN/EGD-Barcelona/split_by_label/train", 
                              transform=transform)
    
    print(dataset[34])  # Output will be (tensor(image), tensor(label))
