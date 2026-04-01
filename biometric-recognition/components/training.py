import torch
import torch.nn as nn
import logging
import argparse
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm 
from config_loader import get_combined_args


# Enhanced Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BiometricDataset(Dataset):
    def __init__(self, metadata_path):
        logger.info(f"Loading metadata from {metadata_path}...")
        # Added weights_only=False to address the Future-Warning
        self.data = torch.load(metadata_path, weights_only=False)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info(f"Dataset initialized with {len(self.data)} samples.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            iris = self.transform(Image.open(item['iris']).convert('RGB'))
            fp = self.transform(Image.open(item['fp']).convert('RGB'))
            return iris, fp, item['id']
        except Exception as e:
            logger.error(f"Error loading files for index {idx}: {e}")
            # Return a blank tensor if a file is corrupt to prevent crash
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), item['id']

class MultiModalModel(nn.Module):
    def __init__(self, num_classes=45):
        super().__init__()
        logger.info("Initializing Dual-Stream ResNet18 Architecture...")
        self.iris_net = models.resnet18(weights=None)
        self.iris_net.fc = nn.Identity()
        self.fp_net = models.resnet18(weights=None)
        self.fp_net.fc = nn.Identity()
        self.fc = nn.Linear(512 + 512, num_classes)

    def forward(self, iris, fp):
        x1 = self.iris_net(iris)
        x2 = self.fp_net(fp)
        return self.fc(torch.cat((x1, x2), dim=1))

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training hardware: {device.type.upper()}")

    dataset = BiometricDataset(args.metadata)
    # Ensure num_workers=0 for stable CPU training in Docker
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = MultiModalModel(num_classes=45).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Starting Training: {args.epochs} epochs, batch size {args.batch_size}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # tqdm Progress Bar
        pbar = tqdm(loader, unit="batch", desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (iris, fp, labels) in enumerate(pbar):
            iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(iris, fp)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            
            # Update progress bar suffix with current loss
            pbar.set_postfix(loss=f"{current_loss:.4f}")
            
            # Additional log every 50 batches for the Docker logs (non-interactive)
            if i % 50 == 0 and i > 0:
                logger.info(f"Epoch {epoch+1} | Batch {i}/{len(loader)} | Loss: {current_loss:.4f}")

        avg_loss = total_loss / len(loader)
        logger.info(f"--- Epoch {epoch+1} Result: Avg Loss {avg_loss:.4f} ---")

        # Save checkpoint after every epoch
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
        logger.info(f"Model checkpoint saved to {args.model_path}")

def training_init(
        ml_config,
        env,
        output_dir,
        input_dirs
):
    return

if __name__ == "__main__":
    args = get_combined_args("Training Step")
    
    train(args)