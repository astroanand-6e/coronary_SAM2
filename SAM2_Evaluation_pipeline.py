# SAM2_Evaluation.py
import os
import random
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import json
# import argparse # Removed argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob # To find checkpoint files

# Import SAM2 modules
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: Could not import SAM2 modules.")
    print("Please make sure you have cloned the 'segment-anything-2' repository")
    print("and installed it (e.g., using 'pip install -e .' inside the repo).")
    exit()

# --- Seed Setting ---
def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

# --- Data Loading Helper ---
def create_data_list(df, image_dir, mask_dir):
    data = []
    for index, row in df.iterrows():
        image_name = row['image_id']
        mask_name = row['mask_id']
        img_path = os.path.join(image_dir, image_name)
        msk_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(img_path) and os.path.exists(msk_path):
             data.append({
                "image": img_path,
                "annotation": msk_path,
                "id": image_name # Store ID for potential reference
            })
        else:
             print(f"Warning: Skipping evaluation entry {index}. Image or mask file not found:")
             if not os.path.exists(img_path): print(f"  - Image: {img_path}")
             if not os.path.exists(msk_path): print(f"  - Mask: {msk_path}")
    return data

# --- Point Sampling Helper ---
def get_points(mask, num_points):
    points_list = []
    coords = np.argwhere(mask > 0) # Find coordinates of non-zero pixels in the mask

    if len(coords) == 0:
        return np.zeros((1, 0, 2), dtype=np.float32), np.zeros((1, 0), dtype=np.float32) # Return empty arrays for coords and labels

    if len(coords) < num_points:
        point_indices = np.arange(len(coords))
        num_points_to_sample = len(coords) # Adjust num_points if fewer available
    else:
        point_indices = np.random.choice(len(coords), num_points, replace=False)
        num_points_to_sample = num_points

    for idx in point_indices:
        yx = coords[idx]
        points_list.append([yx[1], yx[0]])  # Convert to x, y format

    points_np = np.array(points_list, dtype=np.float32)
    if points_np.ndim == 1 and points_np.shape[0] > 0: # Handle case where only one point is sampled
         points_np = points_np.reshape(1,-1)
    elif points_np.shape[0] == 0: # Handle no points found case explicitly
        points_np = np.zeros((0, 2), dtype=np.float32)


    # Add labels (always 1 for foreground points in this context)
    labels_np = np.ones(num_points_to_sample, dtype=np.float32)

    points_np = np.expand_dims(points_np, axis=0) # Add batch dimension -> (1, N_sampled, 2)
    labels_np = np.expand_dims(labels_np, axis=0) # Add batch dimension -> (1, N_sampled)


    return torch.tensor(points_np, dtype=torch.float32), torch.tensor(labels_np, dtype=torch.float32)


# --- Evaluation Function for a single model ---
def evaluate_model(predictor, test_dataloader, device, num_points):
    predictor.model.eval() # Set model to evaluation mode
    total_iou = 0.0
    total_dice = 0.0
    total_sensitivity = 0.0
    total_precision = 0.0
    total_accuracy = 0.0
    num_samples = 0

    with torch.no_grad():
        with tqdm(test_dataloader, desc="Evaluating", unit="batch") as tepoch:
            for i, (images, masks, _, image_ids) in enumerate(tepoch):
                images = images.to(device)
                masks = masks.to(device)

                batch_iou = 0.0
                batch_dice = 0.0
                batch_sensitivity = 0.0
                batch_precision = 0.0
                batch_accuracy = 0.0
                valid_items_in_batch = 0

                # Process only the actual number of items in this batch
                actual_batch_size = images.size(0)
                
                for j in range(actual_batch_size):
                    image_tensor = images[j]
                    gt_mask = masks[j]

                    # --- Sample points from GT mask ---
                    # Check if gt_mask is a tensor before converting
                    if isinstance(gt_mask, torch.Tensor):
                        gt_mask_np = gt_mask.squeeze().cpu().numpy().astype(np.uint8)
                    elif isinstance(gt_mask, np.ndarray):
                         # Assuming it's already squeezed or has the correct shape
                         gt_mask_np = gt_mask.astype(np.uint8)
                    else:
                         print(f"Warning: Unexpected type for gt_mask: {type(gt_mask)}. Skipping item.")
                         continue # Skip this item if type is unexpected

                    point_coords, point_labels = get_points(gt_mask_np, num_points)

                    if point_coords.shape[1] == 0:
                        # print(f"Warning: No points found in GT mask for item {j} in batch {i}. Skipping.") # Optional: Add more verbose logging
                        continue

                    # Set image for the predictor
                    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    predictor.set_image(image_np)

                    # --- Predict using the SAM2 API ---
                    # Rename the output variable to avoid conflict with GT masks
                    predicted_masks, scores, _ = predictor.predict(
                        point_coords=point_coords.cpu().numpy().squeeze(0),
                        point_labels=point_labels.cpu().numpy().squeeze(0),
                        multimask_output=True
                    )

                    # Select best mask (highest score) using the renamed variable
                    best_mask_idx = np.argmax(scores)
                    pred_mask = predicted_masks[best_mask_idx]

                    # Convert to tensor format
                    pred_mask_tensor = torch.tensor(pred_mask, dtype=torch.float32).unsqueeze(0).to(device)

                    # --- Calculate Metrics ---
                    pred_mask_binary = (pred_mask_tensor > 0.5).float()
                    gt_mask_single = gt_mask # gt_mask still refers to the correct GT mask from the batch

                    # Ensure masks are on CPU for numpy operations or use torch directly
                    pred_mask_binary_cpu = pred_mask_binary.squeeze().cpu()
                    
                    # Check if gt_mask_single is a tensor before converting
                    if isinstance(gt_mask_single, torch.Tensor):
                        gt_mask_cpu = gt_mask_single.squeeze().cpu()
                    elif isinstance(gt_mask_single, np.ndarray):
                        # If it's already a numpy array, make sure it's squeezed
                        gt_mask_cpu = torch.tensor(np.squeeze(gt_mask_single))
                    else:
                        print(f"Warning: Unexpected type for gt_mask_single: {type(gt_mask_single)}. Skipping item.")
                        continue
                    
                    # Calculate TP, FP, FN using PyTorch for potential speed/device consistency
                    tp = torch.sum(pred_mask_binary_cpu * gt_mask_cpu)
                    fp = torch.sum(pred_mask_binary_cpu * (1 - gt_mask_cpu))
                    fn = torch.sum((1 - pred_mask_binary_cpu) * gt_mask_cpu)
                    tn = torch.sum((1 - pred_mask_binary_cpu) * (1 - gt_mask_cpu))

                    epsilon = 1e-6

                    # IoU
                    iou = (tp + epsilon) / (tp + fp + fn + epsilon)

                    # Dice
                    dice = (2. * tp + epsilon) / (2. * tp + fp + fn + epsilon)

                    # Sensitivity (Recall)
                    sensitivity = (tp + epsilon) / (tp + fn + epsilon)

                    # Precision
                    precision = (tp + epsilon) / (tp + fp + epsilon)

                    # Accuracy
                    accuracy = (tp + tn + epsilon) / (tp + tn + fp + fn + epsilon)

                    # Accumulate metrics for the batch
                    batch_iou += iou.item()
                    batch_dice += dice.item()
                    batch_sensitivity += sensitivity.item()
                    batch_precision += precision.item()
                    batch_accuracy += accuracy.item()
                    valid_items_in_batch += 1

                if valid_items_in_batch > 0:
                    avg_batch_iou = batch_iou / valid_items_in_batch
                    avg_batch_dice = batch_dice / valid_items_in_batch
                    avg_batch_sens = batch_sensitivity / valid_items_in_batch
                    avg_batch_prec = batch_precision / valid_items_in_batch
                    avg_batch_acc = batch_accuracy / valid_items_in_batch

                    # Accumulate average batch metrics for the epoch
                    total_iou += avg_batch_iou
                    total_dice += avg_batch_dice
                    total_sensitivity += avg_batch_sens
                    total_precision += avg_batch_prec
                    total_accuracy += avg_batch_acc
                    num_samples += 1

                # Update tqdm description with running averages
                tepoch.set_postfix({
                    "avg_iou": total_iou / num_samples if num_samples > 0 else 0,
                    "avg_dice": total_dice / num_samples if num_samples > 0 else 0,
                    "avg_acc": total_accuracy / num_samples if num_samples > 0 else 0
                })

    # Calculate overall average metrics for the epoch
    avg_iou_epoch = total_iou / num_samples if num_samples > 0 else 0
    avg_dice_epoch = total_dice / num_samples if num_samples > 0 else 0
    avg_sens_epoch = total_sensitivity / num_samples if num_samples > 0 else 0
    avg_prec_epoch = total_precision / num_samples if num_samples > 0 else 0
    avg_acc_epoch = total_accuracy / num_samples if num_samples > 0 else 0

    return avg_iou_epoch, avg_dice_epoch, avg_sens_epoch, avg_prec_epoch, avg_acc_epoch

# --- Custom Dataset for Evaluation (Loads data on the fly) ---
# (Same as before - no changes needed here for adding metrics)
class CoronaryArteryEvalDataset(Dataset):
    def __init__(self, data, max_points=3): # max_points is not used here but kept for consistency
        self.data = data
        self.max_points = max_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        mask_path = item['annotation']
        image_id = item['id']

        Img = cv2.imread(image_path)
        if Img is None:
            print(f"Warning: Could not read image {image_path}, returning dummy data.")
            return torch.zeros(3, 1024, 1024), torch.zeros(1, 1024, 1024), torch.zeros(1, self.max_points, 3), "error"
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = Img.astype(np.float32) / 255.0

        ann_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ann_map is None:
            print(f"Warning: Could not read mask {mask_path}, returning dummy data.")
            return torch.zeros(3, 1024, 1024), torch.zeros(1, 1024, 1024), torch.zeros(1, self.max_points, 3), "error"

        ann_map = 255 - ann_map
        ann_map = ann_map.astype(np.float32) / 255.0

        target_size = 1024
        h, w = Img.shape[:2]
        if h != target_size or w != target_size:
             r = min(target_size / w, target_size / h)
             new_w, new_h = int(w * r), int(h * r)
             Img = cv2.resize(Img, (new_w, new_h))
             ann_map = cv2.resize(ann_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
             # Add padding here if needed to reach 1024x1024

        binary_mask = (ann_map > 0.5).astype(np.uint8)

        image = torch.tensor(Img.transpose((2, 0, 1)).copy())
        mask = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0)

        # Return image_id instead of dummy points
        return image, mask, torch.zeros(1), image_id # Return mask and image_id


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration Variables (Hardcoded) ---
    DATA_DIR = "/home/administrator/Dev/Anand/SAM2_CASBloDaM/Aug_dataset_cor_arteries"  # <<< CHANGE THIS to your dataset directory
    MODEL_CFG = "sam2_hiera_s.yaml"
    CHECKPOINT_PATH = "/home/administrator/Dev/Anand/SAM2_CASBloDaM/cor_fine_tuned_tiny_sam2_epoch_5.pt" # <<< CHANGE THIS to your checkpoint file
    RESULTS_FILE = "evaluation_results.csv" # Output file name
    NUM_POINTS = 5      # Number of prompt points
    BATCH_SIZE = 4      # Batch size for evaluation
    DEVICE = None       # Auto-detect device ('cuda', 'mps', 'cpu') or set manually e.g., "cuda:0"
    SEED = 42           # Random seed
    TEST_SIZE = 0.4     # Fraction of data for evaluation
    # --- End Configuration ---

    # parser = argparse.ArgumentParser(description="Evaluate a fine-tuned SAM2 model.") # Removed argparse
    # ... (Removed all parser.add_argument lines)
    # args = parser.parse_args() # Removed argparse

    set_seeds(SEED) # Use hardcoded SEED

    # --- Device Configuration ---

    
    if DEVICE: # Use hardcoded DEVICE
        device = torch.device(DEVICE)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and Prepare Data ---
    print("Loading and preparing evaluation data...")
    data_dir = DATA_DIR # Use hardcoded DATA_DIR
    image_dir = os.path.join(data_dir, "Augmented_image")
    mask_dir = os.path.join(data_dir, "Augmented_mask")
    csv_path = os.path.join(data_dir, "train.csv")

    if not os.path.exists(csv_path) or not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Error: Dataset directory structure is incorrect or train.csv is missing in {data_dir}.")
        exit()

    all_df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(all_df, test_size=TEST_SIZE, random_state=SEED) # Use hardcoded TEST_SIZE and SEED
    test_data_list = create_data_list(test_df, image_dir, mask_dir)
    print(f"Evaluation samples: {len(test_data_list)}")

    if not test_data_list:
        print("Error: No valid evaluation data found.")
        exit()

    test_dataset = CoronaryArteryEvalDataset(test_data_list)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # Use hardcoded BATCH_SIZE

    # --- Load Model ---
    print(f"Loading model config from {MODEL_CFG}") # Use hardcoded MODEL_CFG
    base_model = build_sam2(MODEL_CFG, checkpoint_path=None, device=device) # Use hardcoded MODEL_CFG

    ckpt_path = CHECKPOINT_PATH # Use hardcoded CHECKPOINT_PATH
    model_name = os.path.basename(ckpt_path)
    print(f"\n--- Evaluating {model_name} ---")

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        exit()

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in checkpoint:
             base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
             print("Warning: 'model_state_dict' key not found, attempting to load entire object.")
             base_model.load_state_dict(checkpoint)
        print(f"Loaded weights from {ckpt_path}")
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}.")
        exit() # Exit if the single specified checkpoint fails to load

    predictor = SAM2ImagePredictor(base_model)
    predictor.model.to(device)

    # --- Evaluate ---
    # Evaluate and get all metrics
    avg_iou, avg_dice, avg_sens, avg_prec, avg_acc = evaluate_model(
        predictor, test_dataloader, device, NUM_POINTS # Use hardcoded NUM_POINTS
    )
    print(f"  Avg IoU: {avg_iou:.6f}, Avg Dice: {avg_dice:.6f}, Avg Sensitivity: {avg_sens:.6f}, Avg Precision: {avg_prec:.6f}, Avg Accuracy: {avg_acc:.6f}")

    # --- Save and Report Results ---
    results = {
        "model_checkpoint": model_name,
        "average_iou": avg_iou,
        "average_dice": avg_dice,
        "average_sensitivity": avg_sens,
        "average_precision": avg_prec,
        "average_accuracy": avg_acc
    }

    results_df = pd.DataFrame([results])
    cols_order = ["model_checkpoint", "average_accuracy", "average_iou", "average_dice", "average_sensitivity", "average_precision"]
    results_df = results_df[cols_order]
    results_df.to_csv(RESULTS_FILE, index=False, float_format='%.6f') # Use hardcoded RESULTS_FILE
    print(f"\nEvaluation complete. Results saved to {RESULTS_FILE}") # Use hardcoded RESULTS_FILE

    print("\n--- Evaluation Summary ---")
    print(results_df.to_string(index=False, float_format='%.6f'))
    print("--------------------------")