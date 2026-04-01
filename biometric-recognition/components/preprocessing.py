import os
import torch
import logging
import argparse
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess(raw_dir, output_dir, split_ratio=0.8):
    logger.info(f"Starting preprocessing from {raw_dir} with {split_ratio} split.")
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset = []
    val_dataset = []
    
    identities = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    for person_id in sorted(identities):
        person_path = os.path.join(raw_dir, person_id)
        
        # 1. Collect Iris
        iris_paths = []
        for side in ['left', 'right']:
            side_path = os.path.join(person_path, side)
            if os.path.exists(side_path):
                iris_paths.extend([os.path.join(side_path, f) for f in os.listdir(side_path) if f.lower().endswith('.bmp')])

        # 2. Collect Fingerprints
        fp_path = os.path.join(person_path, "Fingerprint")
        fp_paths = []
        if os.path.exists(fp_path):
            fp_paths = [os.path.join(fp_path, f) for f in os.listdir(fp_path) if f.lower().endswith('.bmp')]

        # 3. Create All Possible Pairs for this person
        person_pairs = []
        for iris in iris_paths:
            for fp in fp_paths:
                person_pairs.append({
                    'iris': iris, 
                    'fp': fp, 
                    'id': int(person_id) - 1
                })
        
        # 4. Shuffle and Split for THIS specific person
        random.shuffle(person_pairs)
        split_idx = int(len(person_pairs) * split_ratio)
        
        train_dataset.extend(person_pairs[:split_idx])
        val_dataset.extend(person_pairs[split_idx:])

    # Save two separate files
    torch.save(train_dataset, os.path.join(output_dir, "train_metadata.pth"))
    torch.save(val_dataset, os.path.join(output_dir, "val_metadata.pth"))
    
    logger.info(f"Preprocessing Complete:")
    logger.info(f"-> Training Pairs: {len(train_dataset)}")
    logger.info(f"-> Validation Pairs: {len(val_dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="/data/raw")
    parser.add_argument("--output_dir", type=str, default="/data/processed")
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()
    preprocess(args.raw_dir, args.output_dir, args.split)