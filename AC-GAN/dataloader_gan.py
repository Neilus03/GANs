from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms

class EGDDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.file_list = []
        self.label_list = []
        for label in os.listdir(root_folder):
            for filename in os.listdir(os.path.join(root_folder, label)):
                self.file_list.append(os.path.join(root_folder, label, filename))
                self.label_list.append(int(label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path)

        label = torch.tensor(self.label_list[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        
        # Flatten the image
        image = torch.flatten(image)

        return image, label


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = EGDDataset(root_folder="/content/drive/MyDrive/EGD-Barcelona/split_by_label/train", 
                     transform=transform)

    
    print(dataset[2])  # Output will be (image, final_label)
