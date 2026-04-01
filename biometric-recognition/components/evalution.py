import torch
import torch.nn as nn
import logging
import argparse
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from components.training import MultiModalModel, BiometricDataset
from config_loader import get_combined_args


# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation Mode: Running on {device}")

    # 1. Load Validation Dataset
    if not os.path.exists(args.metadata):
        logger.error(f"Validation metadata not found at {args.metadata}. Did you run preprocessing with --split?")
        return
        
    dataset = BiometricDataset(args.metadata)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Load Model Architecture & Weights
    model = MultiModalModel(num_classes=45).to(device)
    
    if not os.path.exists(args.model_path):
        logger.error(f"Trained model weights not found at {args.model_path}")
        return

    logger.info(f"Loading weights: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Evaluation Metrics
    correct = 0
    total = 0
    inference_times = []

    logger.info(f"Starting inference on {len(dataset)} validation pairs...")
    
    with torch.no_grad():
        pbar = tqdm(loader, unit="batch", desc="Inference")
        for iris, fp, labels in pbar:
            iris, fp, labels = iris.to(device), fp.to(device), labels.to(device)
            
            # Start Timer for Latency Calculation
            start_time = time.time()
            
            outputs = model(iris, fp)
            
            # End Timer
            end_time = time.time()
            inference_times.append((end_time - start_time) / iris.size(0))
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 4. Final Report
    accuracy = 100 * correct / total
    avg_latency = (sum(inference_times) / len(inference_times)) * 1000 # to ms

    logger.info("--- EVALUATION REPORT ---")
    logger.info(f"Test Samples: {total}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Avg Latency per Identity: {avg_latency:.2f} ms")
    logger.info("-------------------------")

if __name__ == "__main__":
    args = get_combined_args("Evaluation Step")
    evaluate(args)