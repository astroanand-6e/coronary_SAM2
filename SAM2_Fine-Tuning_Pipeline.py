# SAM2 Fine-Tuning Pipeline

# Imports 
import os 
import random
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.colors as mcolors

# Importing SAM2 modules (assuming 'segment-anything-2' repo is cloned and installable)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: Could not import SAM2 modules.")
    print("Please make sure you have cloned the 'segment-anything-2' repository")
    print("and installed it (e.g., using 'pip install -e .' inside the repo).")
    exit()


# ---- seed setting ----

def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)


# ---- Data Loading Helper ----

def create_data_list(df, image_dir, mask_dir):
    data=[]
    for index, row in df.iterrows():
        image_name = row['image_id']
        image_mask = row['mask_id']
        img_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(image_dir, image_mask)
        if os.path.exists(img_path) and os.path.exists(mask_path):
            data.append({
                "image": img_path,
                "mask": mask_path
            })
        else:
            print(f"WARNING!: Skipping entry {index}. Image or mask file not found: ")
            if not os.path.exists(img_path): print(f" -Image: {img_path}")
            if not os.path.exists(mask_path): print(f" -Mask: {mask_path}")
    return data

# ---- Custom Dataset Class ----
class CoronaryArteryDataset(Dataset):
    def __init__(self, data, transform=None, max_points=3):
        self.data = data
        self.transform = transform # Not used
        self.max_points = max_points

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        mask_path = item['annotation']

        # Load image and mask
        Img = cv2.imread(image_path)
        if Img is None:
            raise FileNotFoundError(f"Error: Could not read image file at path: {image_path}. File may be missing, incorrect path, or corrupted.")

        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = Img.astype(np.float32) / 255.0 # Normalize the image to [0, 1]

        ann_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ann_map is None:
             raise FileNotFoundError(f"Error: Could not read mask file at path: {mask_path}. File may be missing, incorrect path, or corrupted.")

        ann_map = 255 - ann_map # Assuming mask is black background, white foreground
        ann_map = ann_map.astype(np.float32) / 255.0 # Normalize the mask to [0, 1]

        # Resize images and masks if needed (Target size 1024x1024 assumed by SAM2)
        target_size = 1024
        if Img.shape[0] != target_size or Img.shape[1] != target_size:
             h, w = Img.shape[:2]
             r = min(target_size / w, target_size / h)
             new_w, new_h = int(w * r), int(h * r)
             Img_resized = cv2.resize(Img, (new_w, new_h))
             ann_map_resized = cv2.resize(ann_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

             # Pad to target size
             top_pad = (target_size - new_h) // 2
             bottom_pad = target_size - new_h - top_pad
             left_pad = (target_size - new_w) // 2
             right_pad = target_size - new_w - left_pad

             Img = np.pad(Img_resized, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0)
             ann_map = np.pad(ann_map_resized, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)
        # Else: Assume image is already 1024x1024

        # Ensure mask is binary (0 or 1) after resizing and normalization
        binary_mask = (ann_map > 0.5).astype(np.uint8) # Threshold to make binary

        # Generate points
        points_list = []
        center_y, center_x = binary_mask.shape[0] // 2, binary_mask.shape[1] // 2

        def find_nearest_white_pixel(x, y, direction, push_distance=10):
            start_x, start_y = x, y
            distance = 0
            found_x, found_y = None, None
            while 0 <= x < binary_mask.shape[1] and 0 <= y < binary_mask.shape[0]:
                if binary_mask[y, x] > 0:
                    found_x, found_y = x, y # Mark the first white pixel encountered
                    break
                x += direction[0]
                y += direction[1]
                distance += 1
            else: # Reached boundary without finding a white pixel
                return None, None

            # If found, push further into the mask
            x, y = found_x, found_y
            for _ in range(push_distance):
                 nx, ny = x + direction[0], y + direction[1] # Check next step
                 if not (0 <= nx < binary_mask.shape[1] and 0 <= ny < binary_mask.shape[0] and binary_mask[ny, nx] > 0):
                     # Stop if the next step is out of bounds or black
                     break
                 # Update only if the next step is valid
                 x, y = nx, ny
            return x, y

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Up, Down, Left, Right
        nearest_points = []

        for direction in directions:
            x, y = center_x, center_y
            nearest_x, nearest_y = find_nearest_white_pixel(x, y, direction)
            if nearest_x is not None and nearest_y is not None:
                nearest_points.append((nearest_x, nearest_y))

        # Sort points by distance to center (optional, might help consistency)
        nearest_points.sort(key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
        # Ensure we have at least one point if possible, but pad/truncate to max_points
        nearest_points = nearest_points[:self.max_points]

        # Handle case where no points are found
        if not nearest_points and np.any(binary_mask):
             # If mask exists but center radiating failed, sample random points on mask
             y_coords, x_coords = np.where(binary_mask > 0)
             if len(y_coords) > 0:
                 num_to_sample = min(self.max_points, len(y_coords))
                 indices = np.random.choice(len(y_coords), num_to_sample, replace=False)
                 for i in indices:
                     nearest_points.append((x_coords[i], y_coords[i]))

        for point in nearest_points:
            points_list.append([point[0], point[1]]) # Append x, y

        points_np = np.array(points_list, dtype=np.float32) if points_list else np.zeros((0, 2), dtype=np.float32)

        # Padding to ensure consistent number of points
        num_found_points = len(points_np)
        padding_needed = self.max_points - num_found_points
        if padding_needed > 0:
            # Pad with a value indicating invalid point, e.g., -1, or keep 0 and handle in loss
            padding_array = np.zeros((padding_needed, 2), dtype=np.float32)
            points_np = np.concatenate([points_np, padding_array], axis=0)

        # Add labels: 1 for foreground points found, -1 for padding points (or 0 if handled in loss)
        labels_np = np.ones(num_found_points, dtype=np.float32)
        if padding_needed > 0:
            padding_labels = -np.ones(padding_needed, dtype=np.float32) # Label padding points as -1 (ignore)
            labels_np = np.concatenate([labels_np, padding_labels], axis=0)

        # Combine points and labels, add batch dimension
        points_with_labels = np.concatenate([points_np, labels_np[:, None]], axis=1)
        points = np.expand_dims(points_with_labels, axis=0) # Shape: (1, max_points, 3)

        # Convert to tensors
        image = torch.tensor(Img.transpose((2, 0, 1)).copy())
        mask = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0) # Use binary_mask

        num_actual_masks = 1 if np.any(binary_mask > 0) else 0

        return image, mask, torch.tensor(points, dtype=torch.float32), num_actual_masks
    

# ---- Training Function ----
def train_epoch(predictor, train_dataloader, epoch, accumulation_steps, optimizer, scaler, device, num_epochs):
    predictor.model.train() # Set model to training mode
    # Ensure requires_grad is True for relevant parts
    # Use the correct attribute names for each component
    for param in predictor.model.sam_mask_decoder.parameters():
        param.requires_grad = True
    for param in predictor.model.sam_prompt_encoder.parameters():
        param.requires_grad = True
    # Keep image encoder frozen - note the attribute is 'image_encoder', not 'sam_image_encoder'
    for param in predictor.model.image_encoder.parameters():
        param.requires_grad = False

    total_loss = 0.0
    total_iou = 0.0
    processed_batches = 0 # Count batches where optimizer step occurred

    # Use tqdm for progress bar
    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        optimizer.zero_grad(set_to_none=True) # Zero gradients at the start of the epoch accumulation cycle
        accumulated_items = 0 # Track items within an accumulation cycle
        accumulated_loss = 0.0
        accumulated_iou = 0.0

        for i, (images, masks, points, _) in enumerate(tepoch): # _ ignores num_masks if not needed here
            images = images.to(device)
            masks = masks.to(device) # Ground truth masks
            points = points.to(device) # Input points with labels

            batch_loss = 0.0
            batch_iou = 0.0
            valid_items_in_batch = 0

            # Process each item in the batch individually for prediction
            for j in range(images.size(0)):
                image_tensor = images[j] # Shape (C, H, W)
                gt_mask = masks[j]      # Shape (1, H, W)
                point_coords = points[j, :, :, :2] # Shape (1, N, 2) - Includes padding
                point_labels = points[j, :, :, 2] # Shape (1, N) - Includes padding labels (-1)

                # Filter out padding points based on label (-1 assumed for padding)
                valid_points_mask = (point_labels[0] != -1)
                if not torch.any(valid_points_mask):
                    # print(f"Warning: No valid points found for item {j} in batch {i}. Skipping.")
                    continue # Skip if no valid points

                valid_point_coords = point_coords[0][valid_points_mask].unsqueeze(0) # Shape (1, N_valid, 2)
                valid_point_labels = point_labels[0][valid_points_mask].unsqueeze(0) # Shape (1, N_valid)

                # Set image for the predictor (needs numpy HWC uint8)
                image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                try:
                    predictor.set_image(image_np) # Takes HWC numpy array
                except Exception as e:
                    print(f"Error setting image for predictor in batch {i}, item {j}: {e}")
                    print(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")
                    continue # Skip this item

                # --- Mixed Precision Training ---
                with torch.autocast(device_type=str(device).split(':')[0], dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, enabled=scaler.is_enabled()):
                    # Get sparse and dense embeddings from the prompt encoder
                    try:
                        # Use attribute name without sam_ prefix
                        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                            points=(valid_point_coords, valid_point_labels), # Use only valid points
                            boxes=None,
                            masks=None
                        )
                    except Exception as e:
                         print(f"Error in prompt encoder for batch {i}, item {j}: {e}")
                         print(f"Valid points shape: {valid_point_coords.shape}, labels shape: {valid_point_labels.shape}")
                         continue # Skip this item


                    # Predict masks using the mask decoder
                    # Use attribute name without sam_ prefix
                    low_res_masks, pred_iou_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor.features["image_embed"][-1].unsqueeze(0), # Use precomputed features
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(), # Use attribute name without sam_ prefix
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True, # Get multiple masks if configured
                        repeat_image=False, # Process one image at a time
                        high_res_features=[feat_level[-1].unsqueeze(0) for feat_level in predictor.features["high_res_feats"]]
                    )

                    # Postprocess masks to original image size
                    pred_masks_high_res = predictor.model.postprocess_masks(
                        low_res_masks,
                        predictor.input_size,
                        predictor.original_size,
                    ) # Shape (B=1, N_masks, H, W)

                    # Select the best mask based on predicted IoU score
                    best_mask_idx = torch.argmax(pred_iou_scores[0])
                    pred_mask_sigmoid = torch.sigmoid(pred_masks_high_res[0, best_mask_idx, :, :]).unsqueeze(0) # Shape (1, H, W)

                    # --- Calculate Loss ---
                    gt_mask_single = gt_mask # Already (1, H, W)

                    # Binary Cross-Entropy Loss (Segmentation Loss)
                    seg_loss = torch.nn.functional.binary_cross_entropy(pred_mask_sigmoid, gt_mask_single)

                    # IoU Calculation (for IoU loss and monitoring)
                    pred_mask_binary = (pred_mask_sigmoid > 0.5).float()
                    intersection = torch.sum(pred_mask_binary * gt_mask_single, dim=(1, 2))
                    union = torch.sum(pred_mask_binary, dim=(1, 2)) + torch.sum(gt_mask_single, dim=(1, 2)) - intersection
                    iou = (intersection + 1e-6) / (union + 1e-6) # Add epsilon

                    # IoU Loss (Predicted IoU vs Calculated IoU) - Use score for the best mask
                    iou_loss = torch.nn.functional.mse_loss(pred_iou_scores[0, best_mask_idx], iou[0]) # Compare single score to single IoU

                    # Combined Loss (adjust weights as needed)
                    loss_item = seg_loss + 20.0 * iou_loss # Weight IoU loss more as in SAM paper

                # Scale loss for mixed precision and accumulate gradients
                # Normalize loss by accumulation steps before scaling
                scaler.scale(loss_item / accumulation_steps).backward()

                batch_loss += loss_item.item()
                batch_iou += iou[0].item() # Get the scalar IoU value
                valid_items_in_batch += 1

            # Accumulate metrics for averaging over the accumulation cycle
            if valid_items_in_batch > 0:
                accumulated_loss += batch_loss
                accumulated_iou += batch_iou
                accumulated_items += valid_items_in_batch

            # Update model weights after accumulation_steps batches or at the end of epoch
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                if accumulated_items > 0: # Only step if gradients were accumulated
                    # Unscale gradients before clipping and stepping
                    scaler.unscale_(optimizer)
                    # Optional gradient clipping
                    torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
                    # Optimizer step
                    scaler.step(optimizer)
                    # Update scaler for next iteration
                    scaler.update()
                    # Zero gradients for the next accumulation cycle
                    optimizer.zero_grad(set_to_none=True)

                    # Log average loss/iou for the completed accumulation cycle
                    avg_cycle_loss = accumulated_loss / accumulated_items
                    avg_cycle_iou = accumulated_iou / accumulated_items
                    total_loss += avg_cycle_loss # Accumulate cycle average loss for epoch average
                    total_iou += avg_cycle_iou   # Accumulate cycle average IoU for epoch average
                    processed_batches += 1        # Increment count of effective batches (optimizer steps)

                    # Reset accumulators for the next cycle
                    accumulated_loss = 0.0
                    accumulated_iou = 0.0
                    accumulated_items = 0

                    # Update tqdm postfix with the latest cycle averages
                    tepoch.set_postfix({
                        "loss": avg_cycle_loss, # Show current cycle average loss
                        "iou": avg_cycle_iou,    # Show current cycle average IoU
                        "lr": optimizer.param_groups[0]["lr"],
                    })
                else:
                     # If no valid items in the cycle, still need to zero grad if it was the end of the loader
                     if (i + 1) == len(train_dataloader):
                          optimizer.zero_grad(set_to_none=True)


    # Calculate epoch averages based on processed batches (optimizer steps)
    avg_loss_epoch = total_loss / processed_batches if processed_batches > 0 else 0.0
    avg_iou_epoch = total_iou / processed_batches if processed_batches > 0 else 0.0
    return avg_loss_epoch, avg_iou_epoch

# ---- Validation Function ----
@torch.no_grad() # Decorator for no_grad context
def validate_epoch(predictor, test_dataloader, epoch, device, num_epochs):
    predictor.model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_iou = 0.0
    processed_items = 0 # Count total valid items processed

    with tqdm(test_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        for i, (images, masks, points, _) in enumerate(tepoch):
            images = images.to(device)
            masks = masks.to(device)
            points = points.to(device)

            batch_loss = 0.0
            batch_iou = 0.0
            valid_items_in_batch = 0

            for j in range(images.size(0)):
                image_tensor = images[j]
                gt_mask = masks[j]
                point_coords = points[j, :, :, :2]
                point_labels = points[j, :, :, 2]

                valid_points_mask = (point_labels[0] != -1)
                if not torch.any(valid_points_mask):
                     continue # Skip if no valid points

                valid_point_coords = point_coords[0][valid_points_mask].unsqueeze(0)
                valid_point_labels = point_labels[0][valid_points_mask].unsqueeze(0)

                image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                try:
                    predictor.set_image(image_np)
                except Exception as e:
                     print(f"Error setting image for predictor in validation batch {i}, item {j}: {e}")
                     continue

                # Use autocast for potential speedup during validation, disable gradients
                with torch.autocast(device_type=str(device).split(':')[0], dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, enabled=str(device).startswith("cuda")):
                    try:
                        # Use attribute name without sam_ prefix
                        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                            points=(valid_point_coords, valid_point_labels), boxes=None, masks=None
                        )
                    except Exception as e:
                         print(f"Error in validation prompt encoder for batch {i}, item {j}: {e}")
                         continue

                    # Use attribute name without sam_ prefix
                    low_res_masks, pred_iou_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor.features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(), # Use attribute name without sam_ prefix
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=False,
                        high_res_features=[feat_level[-1].unsqueeze(0) for feat_level in predictor.features["high_res_feats"]]
                    )

                pred_masks_high_res = predictor.model.postprocess_masks(
                    low_res_masks,
                    predictor.input_size,
                    predictor.original_size,
                )
                best_mask_idx = torch.argmax(pred_iou_scores[0])
                pred_mask_sigmoid = torch.sigmoid(pred_masks_high_res[0, best_mask_idx, :, :]).unsqueeze(0)
                gt_mask_single = gt_mask

                # --- Calculate Loss and IoU ---
                seg_loss = torch.nn.functional.binary_cross_entropy(pred_mask_sigmoid, gt_mask_single)
                pred_mask_binary = (pred_mask_sigmoid > 0.5).float()
                intersection = torch.sum(pred_mask_binary * gt_mask_single, dim=(1, 2))
                union = torch.sum(pred_mask_binary, dim=(1, 2)) + torch.sum(gt_mask_single, dim=(1, 2)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                iou_loss = torch.nn.functional.mse_loss(pred_iou_scores[0, best_mask_idx], iou[0])
                loss_item = seg_loss + 20.0 * iou_loss

                batch_loss += loss_item.item()
                batch_iou += iou[0].item()
                valid_items_in_batch += 1

            if valid_items_in_batch > 0:
                avg_batch_loss = batch_loss / valid_items_in_batch
                avg_batch_iou = batch_iou / valid_items_in_batch
                total_loss += batch_loss # Accumulate total loss across all items
                total_iou += batch_iou   # Accumulate total iou across all items
                processed_items += valid_items_in_batch # Count processed items

                tepoch.set_postfix({"loss": avg_batch_loss, "iou": avg_batch_iou})
            else:
                 tepoch.set_postfix({"loss": 0, "iou": 0})


    # Calculate epoch averages based on total processed items
    avg_loss_epoch = total_loss / processed_items if processed_items > 0 else 0.0
    avg_iou_epoch = total_iou / processed_items if processed_items > 0 else 0.0
    return avg_loss_epoch, avg_iou_epoch

# ---- Main Execution Block ----
if __name__ == "__main__":

    # --- Configuration / Hyperparameters ---
    # !! IMPORTANT: Update DATA_DIR and CHECKPOINT with your actual paths !!
    DATA_DIR = "/home/administrator/Dev/Anand/SAM2_CASBloDaM/Aug_dataset_cor_arteries" 
    CHECKPOINT = "/home/administrator/Dev/Anand/SAM2_CASBloDaM/sam2_hiera_small.pt"    

    # --- Model and Paths ---
    MODEL_CFG = "sam2_hiera_s.yaml" 
    OUTPUT_DIR = "checkpoints_sam2_tiny_coronary"
    HISTORY_DIR = "history_sam2_tiny_coronary"
    MODEL_NAME_PREFIX = "sam2_small_coronary_tuned"

    # --- Training Parameters ---
    EPOCHS = 50              # Number of training epochs
    BATCH_SIZE = 8                # Batch size (adjust based on GPU memory)
    LEARNING_RATE = 4e-4          # Learning rate (often lower for fine-tuning)
    WEIGHT_DECAY = 0.01           # Weight decay for AdamW
    ACCUMULATION_STEPS = 4        # Gradient accumulation steps (effective batch size = BATCH_SIZE * ACCUMULATION_STEPS)
    MAX_POINTS = 3                # Number of prompt points per sample
    TEST_SIZE = 0.15              # Fraction of data for validation (e.g., 15%)
    SEED = 42                     # Random seed for reproducibility

    # --- Device Configuration ---
    # Set to None for auto-detect, or specify "cuda:0", "mps", "cpu"
    FORCE_DEVICE = None
    # ----------------------------------------

    set_seeds(SEED)

    # --- Determine Device ---

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") #Print the used device to ensure the model is initialized on the correct device


    # --- Create Output Directories ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # --- Load Data ---
    print("Loading data...")
    image_dir = os.path.join(DATA_DIR, "Augmented_image")
    mask_dir = os.path.join(DATA_DIR, "Augmented_mask")
    csv_path = os.path.join(DATA_DIR, "train.csv")

    if not os.path.exists(DATA_DIR) or not os.path.isdir(DATA_DIR):
         print(f"Error: DATA_DIR not found or is not a directory: {DATA_DIR}")
         print("Please update the DATA_DIR variable in the script.")
         exit()
    if not os.path.exists(csv_path):
         print(f"Error: train.csv not found in {DATA_DIR}")
         exit()
    if not os.path.exists(image_dir):
         print(f"Error: Image directory not found: {image_dir}")
         exit()
    if not os.path.exists(mask_dir):
         print(f"Error: Mask directory not found: {mask_dir}")
         exit()



    all_df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(all_df, test_size=TEST_SIZE, random_state=SEED)

    train_data = create_data_list(train_df, image_dir, mask_dir)
    test_data = create_data_list(test_df, image_dir, mask_dir)
    print(f"Training samples: {len(train_data)}, Validation samples: {len(test_data)}")

    if not train_data or not test_data:
        print("Error: No valid training or testing data found. Check file paths and CSV content.")
        exit()

    # --- Create Datasets and Dataloaders ---
    print("Creating datasets and dataloaders...")
    train_dataset = CoronaryArteryDataset(train_data, max_points=MAX_POINTS)
    test_dataset = CoronaryArteryDataset(test_data, max_points=MAX_POINTS)

    # Adjust num_workers based on your system's capabilities
    num_workers = 2 if os.name == 'posix' else 0 # Generally > 0 works better on Linux/Mac
    print(f"Using {num_workers} workers for DataLoaders.")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if str(device) != 'cpu' else False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if str(device) != 'cpu' else False)

    # --- Initialize Model and Predictor ---
    print("Initializing SAM2 model...")
    # Ensure the config path is correct relative to where you run the script or use an absolute path
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # --- Configure Model for Training ---
    print("Configuring model for fine-tuning (freezing image encoder)...")
    # Use the correct attribute names for each component
    for name, param in predictor.model.named_parameters():
        if name.startswith("image_encoder"):  # Without 'sam_' prefix
            param.requires_grad_(False)
        # Check parameter names using the prefix from the dump
        elif name.startswith("sam_prompt_encoder") or name.startswith("sam_mask_decoder"):
            param.requires_grad_(True)
        else:
             param.requires_grad_(False) # Freeze other potential parts

    # Verify which parts are trainable (optional)
    # print("Trainable parameters:")
    # total_trainable_params = 0
    # for name, param in predictor.model.named_parameters():
    #     if param.requires_grad:
    #         print(f"  {name} - {param.numel()}")
    #         total_trainable_params += param.numel()
    # print(f"Total trainable parameters: {total_trainable_params}")


    predictor.model.to(device)

    # --- Optimizer and Scheduler ---
    print("Setting up optimizer and scheduler...")
    # Explicitly collect parameters that should be trainable based on their names
    trainable_params = []
    print("Identifying trainable parameters:")
    for name, param in predictor.model.named_parameters():
        # Check parameter names using the correct prefixes
        if name.startswith("sam_prompt_encoder") or name.startswith("sam_mask_decoder"):
            if param.requires_grad: # Ensure requires_grad was set correctly earlier
                trainable_params.append(param)
                # print(f"  Trainable: {name}") # Uncomment for verbose logging
            else:
                # This warning should hopefully not appear now
                print(f"  Warning: Parameter '{name}' was expected to be trainable but requires_grad is False.")
        elif param.requires_grad:
             # This case indicates unexpected trainable parameters
             print(f"  Warning: Parameter '{name}' is trainable but was not expected to be (frozen?).")

    # Check if any trainable parameters were found
    if not trainable_params:
        print("\nError: No trainable parameters were found for the optimizer!")
        print("Please check the model configuration section and ensure the parameter names")
        print("('prompt_encoder', 'mask_decoder') correctly match the model structure")
        print("and that their requires_grad attribute is being set to True.")
        print("\nListing all model parameters and their requires_grad status:")
        for name, param in predictor.model.named_parameters():
            print(f"  - {name}: requires_grad={param.requires_grad}")
        exit() # Exit if no parameters to train

    print(f"Found {len(trainable_params)} parameter groups/tensors to train.")

    optimizer = torch.optim.AdamW(
        params=trainable_params, # Pass the collected list
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', # Reduce LR when validation loss plateaus
        factor=0.5,
        patience=3, # Reduce LR more quickly if validation loss stagnates
        min_lr=1e-7, # Lower minimum LR
        verbose=True # Print message when LR is reduced
    )

    # --- Mixed Precision Scaler ---
    # Enable only if CUDA is used and available
    use_amp = str(device).startswith("cuda") and torch.cuda.is_available()
    # Use bfloat16 if available on CUDA, otherwise float16
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print(f"Using Automatic Mixed Precision (AMP) with dtype: {amp_dtype}.")
    else:
        print("AMP not used (requires CUDA).")
        if str(device).startswith("cuda"):
             print("Warning: CUDA selected but torch.cuda.is_available() is False.")


    # --- Main Training Loop ---
    print("Starting training...")
    training_history = []
    validation_history = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss, train_iou = train_epoch(
            predictor, train_dataloader, epoch, ACCUMULATION_STEPS, optimizer, scaler, device, EPOCHS
        )
        valid_loss, valid_iou = validate_epoch(
            predictor, test_dataloader, epoch, device, EPOCHS
        )

        # Scheduler step based on validation loss
        scheduler.step(valid_loss)

        # Log results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.6f}, Train IoU: {train_iou:.6f}")
        print(f"  Valid Loss: {valid_loss:.6f}, Valid IoU: {valid_iou:.6f}")
        print(f"  Learning Rate: {current_lr:.8f}")


        # Store history
        training_history.append({"epoch": epoch + 1, "loss": float(train_loss), "iou": float(train_iou)})
        validation_history.append({"epoch": epoch + 1, "loss": float(valid_loss), "iou": float(valid_iou)})


        # Save model checkpoint
        checkpoint_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME_PREFIX}_epoch_{epoch+1}.pt")
        best_checkpoint_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME_PREFIX}_best.pt")

        # Save best model based on validation loss
        if valid_loss < best_val_loss:
            print(f"  Validation loss improved ({best_val_loss:.6f} -> {valid_loss:.6f}). Saving best model...")
            best_val_loss = valid_loss
            # Save the current model state as the best
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': predictor.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': valid_loss,
                'val_iou': valid_iou,
                'config': { # Store hyperparameters used for this run
                    'DATA_DIR': DATA_DIR,
                    'MODEL_CFG': MODEL_CFG,
                    'CHECKPOINT': CHECKPOINT, # Original checkpoint
                    'EPOCHS': EPOCHS,
                    'BATCH_SIZE': BATCH_SIZE,
                    'LEARNING_RATE': LEARNING_RATE, # Initial LR
                    'WEIGHT_DECAY': WEIGHT_DECAY,
                    'ACCUMULATION_STEPS': ACCUMULATION_STEPS,
                    'MAX_POINTS': MAX_POINTS,
                    'TEST_SIZE': TEST_SIZE,
                    'SEED': SEED,
                }
            }, best_checkpoint_path)
            print(f"  Saved best model to {best_checkpoint_path}")

        # Optionally save checkpoint for every epoch (can take significant disk space)
        # torch.save({ ... }, checkpoint_path)
        # print(f"  Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")

        # Save last epoch checkpoint separately
        if epoch == EPOCHS - 1:
             last_checkpoint_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME_PREFIX}_last.pt")
             print("  Saving model for last epoch.")
             torch.save({
                'epoch': epoch + 1,
                'model_state_dict': predictor.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': valid_loss,
                'train_iou': train_iou,
                'val_iou': valid_iou,
                'learning_rate': current_lr,
                'config': { # Store hyperparameters used for this run
                    'DATA_DIR': DATA_DIR,
                    'MODEL_CFG': MODEL_CFG,
                    'CHECKPOINT': CHECKPOINT, # Original checkpoint
                    'EPOCHS': EPOCHS,
                    'BATCH_SIZE': BATCH_SIZE,
                    'LEARNING_RATE': LEARNING_RATE, # Initial LR
                    'WEIGHT_DECAY': WEIGHT_DECAY,
                    'ACCUMULATION_STEPS': ACCUMULATION_STEPS,
                    'MAX_POINTS': MAX_POINTS,
                    'TEST_SIZE': TEST_SIZE,
                    'SEED': SEED,
                }
            }, last_checkpoint_path)
             print(f"  Saved last model to {last_checkpoint_path}")


    print("Training complete!")

    # --- Save History ---
    history_file_train = os.path.join(HISTORY_DIR, f"{MODEL_NAME_PREFIX}_training_history.json")
    history_file_val = os.path.join(HISTORY_DIR, f"{MODEL_NAME_PREFIX}_validation_history.json")

    with open(history_file_train, 'w') as f:
        json.dump(training_history, f, indent=4)
    with open(history_file_val, 'w') as f:
        json.dump(validation_history, f, indent=4)
    print(f"Saved training history to {history_file_train}")
    print(f"Saved validation history to {history_file_val}")


    # --- Plot History ---
    try:
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot([h['epoch'] for h in training_history], [h['loss'] for h in training_history], marker='o', linestyle='-', label='Training Loss')
        plt.plot([h['epoch'] for h in validation_history], [h['loss'] for h in validation_history], marker='o', linestyle='-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0) # Loss should not be negative

        # Plot IoU
        plt.subplot(1, 2, 2)
        plt.plot([h['epoch'] for h in training_history], [h['iou'] for h in training_history], marker='o', linestyle='-', label='Training IoU')
        plt.plot([h['epoch'] for h in validation_history], [h['iou'] for h in validation_history], marker='o', linestyle='-', label='Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.title('Training & Validation Mean IoU')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1) # IoU is between 0 and 1

        plt.tight_layout()
        plot_file = os.path.join(HISTORY_DIR, f"{MODEL_NAME_PREFIX}_training_plot.png")
        plt.savefig(plot_file)
        print(f"Saved training plot to {plot_file}")
        # plt.show() # Uncomment to display plot if running interactively
        plt.close() # Close the plot to free memory
    except Exception as e:
         print(f"Could not generate or save plots: {e}")