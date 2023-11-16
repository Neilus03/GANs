# dataloader.py
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class_mapping = {'paella': 0, 'macaroni_and_cheese': 1, 'waffles': 2}

class CustomFoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images and labels
        for class_name, class_idx in class_mapping.items():
            class_dir = os.path.join(root_dir, 'images', class_name)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_food101_dataloader(batch_size, root_dir, transform):
    dataset = CustomFoodDataset(root_dir=root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((560, 640)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])

#EXAMPLE TESTING DATALOADER

# Create dataloader
root_dir = '/home/ndelafuente/CVC/GANs/ACA-GAN/data/food-101'
batch_size = 16
train_loader = get_food101_dataloader(batch_size=batch_size, root_dir=root_dir, transform=transform)

# Test dataloader
if __name__ == '__main__':
    for i, (img, label) in enumerate(train_loader):
        print(f"Batch {i+1}: {img.shape}, {label}")
        plt.imshow(img[0].permute(1, 2, 0))
        plt.show()
        if i == 10:
            break
