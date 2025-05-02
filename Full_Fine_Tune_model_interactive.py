# --- START OF FILE Fine-tune_model_interactive_no_preprocess.py ---

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms # Likely not needed now
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random
import json
from skimage.morphology import skeletonize # Ensure skimage is installed
from scipy import ndimage # For finding center of mass

# Import SAM2 Modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ----- Helper Functions -----
# (set_seed, ensure_dir, create_data_list remain the same)
def set_seed(seed_value=42):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value); torch.backends.cudnn.benchmark = True

def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory); print(f"Created: {directory}")
    return directory

def create_data_list(df, image_dir, mask_dir):
    data = []
    for index, row in df.iterrows():
        data.append({"image": os.path.join(image_dir, row['image_id']),
                     "annotation": os.path.join(mask_dir, row['mask_id'])})
    return data

# ----- Loss Functions -----
# (dice_loss, focal_loss remain the same)
def dice_loss(pred, target, smooth=1.0):
    target = target.float(); pred_flat = pred.contiguous().view(-1); target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum(); union = pred_flat.sum() + target_flat.sum()
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)

def focal_loss(pred, target, gamma=2.0, alpha=0.25, reduction='mean'):
    target = target.float(); pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
    bce_loss = -target * torch.log(pred) * (1 - pred) ** gamma - (1 - target) * torch.log(1 - pred) * pred ** gamma
    focal_loss = alpha * bce_loss
    if reduction == 'mean': return focal_loss.mean()
    elif reduction == 'sum': return focal_loss.sum()
    else: return focal_loss


def get_optimizer_and_scheduler(model, max_epochs, encoder_lr_factor=0.1):
    base_lr = 1e-5; encoder_lr = base_lr * encoder_lr_factor; decoder_lr = base_lr
    encoder_params, decoder_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'image_encoder' in name: encoder_params.append(param)
            else: decoder_params.append(param)
    param_groups = [ {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': 1e-4},
                     {'params': decoder_params, 'lr': decoder_lr, 'weight_decay': 1e-4}]
    print(f"Optimizer: Using LR {encoder_lr:.2e} (encoder), {decoder_lr:.2e} (decoder/prompts).")
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    return optimizer, scheduler


# ----- Interactive Sampling Helper Functions -----
# (calculate_error_regions, sample_points_from_mask remain the same)
def calculate_error_regions(pred_mask_binary, gt_mask):
    pred_mask_binary = pred_mask_binary.astype(np.uint8); gt_mask = gt_mask.astype(np.uint8)
    fp_mask = np.maximum(pred_mask_binary - gt_mask, 0)
    fn_mask = np.maximum(gt_mask - pred_mask_binary, 0)
    return fp_mask, fn_mask

def sample_points_from_mask(mask, num_points, label, strategy='centroid'):
    points, labels_out = [], [] # Use labels_out to avoid conflict with cv2 labels_im
    if np.sum(mask) == 0 or num_points == 0: return np.array([]).reshape(0, 2), np.array([])
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_indices = list(range(1, num_labels)); random.shuffle(component_indices)
    sampled_count = 0
    for i in component_indices:
        if sampled_count >= num_points: break
        if strategy == 'centroid':
            center_x, center_y = centroids[i]; cy, cx = int(round(center_y)), int(round(center_x))
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and labels_im[cy, cx] == i:
                 points.append([cx, cy]); labels_out.append(label); sampled_count += 1
            else: # Fallback
                 component_pixels = np.argwhere(labels_im == i)
                 if len(component_pixels) > 0:
                    py, px = component_pixels[random.randint(0, len(component_pixels) - 1)]
                    points.append([px, py]); labels_out.append(label); sampled_count += 1
        elif strategy == 'random': # Fallback
             component_pixels = np.argwhere(labels_im == i)
             if len(component_pixels) > 0:
                py, px = component_pixels[random.randint(0, len(component_pixels) - 1)]
                points.append([px, py]); labels_out.append(label); sampled_count += 1
    remaining_points = num_points - sampled_count
    if remaining_points > 0:
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
             num_to_sample = min(remaining_points, len(y_indices))
             indices = np.random.choice(len(y_indices), num_to_sample, replace=False)
             for idx in indices: points.append([x_indices[idx], y_indices[idx]]); labels_out.append(label)
    return np.array(points, dtype=np.float32), np.array(labels_out, dtype=np.float32)


# ----- Dataset Class (No Internal Preprocessing) -----
class CoronaryArteryDatasetInteractive(Dataset):
    def __init__(self, data, initial_max_points=3, augment=False, target_size=1024):
        self.data = data
        self.initial_max_points = initial_max_points
        self.augment = augment
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        mask_path = item['annotation']

        # --- Load Image (assumed pre-processed externally) ---
        Img = cv2.imread(image_path)
        if Img is None: raise FileNotFoundError(f"Img not found: {image_path}")
        # Keep original channels (e.g., RGB if saved as such)
        if Img.ndim == 2: # If loaded as grayscale, convert to 3 channels if model expects it
             Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        elif Img.shape[2] == 1:
             Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else: # Assume BGR, convert to RGB
             Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

        # --- Load Mask ---
        ann_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ann_map is None: raise FileNotFoundError(f"Mask not found: {mask_path}")

        # --- Resize ---
        Img = cv2.resize(Img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        ann_map = cv2.resize(ann_map, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # --- Basic Type Conversion and Normalization [0, 1] ---
        # Convert image to float32 and scale to [0, 1] if it's 8-bit
        if Img.dtype == np.uint8:
            Img = Img.astype(np.float32) / 255.0
        # Convert mask to float32 and scale to [0, 1]
        if ann_map.dtype == np.uint8:
            ann_map = ann_map.astype(np.float32) / 255.0

        # --- REMOVED preprocess_xray_image CALL ---
        # Img = preprocess_xray_image(Img) # <--- This line is removed

        # --- Augmentation (applied to [0,1] float image) ---
        if self.augment:
            # Make sure augmentations handle float [0,1] images correctly
            if random.random() > 0.5: # Rotation
                angle = random.uniform(-15, 15); M = cv2.getRotationMatrix2D((self.target_size//2, self.target_size//2), angle, 1.0)
                # Pad with 0 for float image
                Img = cv2.warpAffine(Img, M, (self.target_size, self.target_size), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                ann_map = cv2.warpAffine(ann_map, M, (self.target_size, self.target_size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if random.random() > 0.5: # Brightness/Contrast
                # Adjust brightness (add) and contrast (multiply) for [0,1] range
                alpha = random.uniform(0.8, 1.2) # Contrast factor
                beta = random.uniform(-0.1, 0.1)  # Brightness shift
                Img = np.clip(alpha * Img + beta, 0.0, 1.0) # Apply and clip to [0,1]
            if random.random() > 0.5: # Flip
                Img = cv2.flip(Img, 1); ann_map = cv2.flip(ann_map, 1)

        # --- Generate Initial Points (Same logic as before) ---
        binary_mask = (ann_map > 0.5).astype(np.uint8)
        skeleton = skeletonize(binary_mask)
        points_list, labels_list = [], []
        y_indices, x_indices = np.where(skeleton)
        num_skeleton_points = len(y_indices)
        if num_skeleton_points > 0:
            count = min(num_skeleton_points, self.initial_max_points)
            indices = np.random.choice(num_skeleton_points, count, replace=False)
            for i in indices: points_list.append([x_indices[i], y_indices[i]]); labels_list.append(1)
        points_needed = self.initial_max_points - len(points_list)
        if points_needed > 0:
            y_mask_indices, x_mask_indices = np.where(binary_mask)
            num_mask_points = len(y_mask_indices)
            if num_mask_points > 0:
                sample_count = min(points_needed, num_mask_points)
                indices = np.random.choice(num_mask_points, sample_count, replace=False)
                for i in indices: points_list.append([x_mask_indices[i], y_mask_indices[i]]); labels_list.append(1)
        if not points_list: points_list.append([self.target_size // 2, self.target_size // 2]); labels_list.append(1)
        initial_points_np = np.array(points_list, dtype=np.float32)
        initial_labels_np = np.array(labels_list, dtype=np.float32)
        initial_points_with_labels = np.concatenate([initial_points_np, initial_labels_np[:, None]], axis=1)
        initial_points_batch = np.expand_dims(initial_points_with_labels, axis=0)

        # --- Convert to Tensors ---
        # Ensure image is (C, H, W)
        image_tensor = torch.tensor(Img.transpose((2, 0, 1)).copy(), dtype=torch.float32)
        mask_tensor = torch.tensor(ann_map, dtype=torch.float32).unsqueeze(0)
        initial_points_tensor = torch.tensor(initial_points_batch, dtype=torch.float32)

        return image_tensor, mask_tensor, initial_points_tensor, self.target_size

    def visualize(self, idx):
        # Visualization needs to handle potentially non-standardized images
        image, mask, points, _ = self.__getitem__(idx)
        image_np = image.numpy().transpose((1, 2, 0))
        # Image should be [0, 1] now, so clipping is enough for display
        image_disp = np.clip(image_np, 0, 1)
        mask_np = mask.squeeze(0).numpy()
        points_np = points.squeeze(0).numpy()
        initial_points_np_vis = points_np # Only initial points are generated here

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(image_disp)
        if initial_points_np_vis.shape[0] > 0:
             plt.scatter(initial_points_np_vis[:, 0], initial_points_np_vis[:, 1], c='red', marker='*', s=100)
        plt.title(f"Image {idx} (Initial Points)"); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(mask_np, cmap='gray'); plt.title(f"Mask {idx}"); plt.axis('off')
        plt.tight_layout(); plt.show()


# ----- Training & Validation Functions (INTERACTIVE SIMULATION) -----
# (train_epoch_interactive, validate_epoch_interactive remain the same as previous version)
# Ensure they use the updated CoronaryArteryDatasetInteractive
def train_epoch_interactive(
    predictor, train_dataloader, epoch, accumulation_steps, optimizer, scaler, device,
    NO_OF_EPOCHS, focal_weight=0.5, dice_weight=0.5,
    num_interaction_rounds=3, points_per_round=1
    ):
    predictor.model.train(); total_final_loss = 0.0; total_final_iou = 0.0; processed_samples_count = 0
    optimizer.zero_grad(set_to_none=True)
    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NO_OF_EPOCHS} (Interact)", unit="batch") as tepoch:
        for i, batch_data in enumerate(tepoch):
            images, masks_gt, initial_points_batch, target_size = batch_data
            images, masks_gt = images.to(device), masks_gt.to(device)
            batch_final_losses, batch_final_ious = [], []
            for j in range(images.shape[0]):
                current_image, current_mask_gt, current_initial_points = images[j], masks_gt[j], initial_points_batch[j]
                # Convert CHW tensor to HWC numpy array for set_image
                image_numpy = current_image.cpu().numpy().transpose((1, 2, 0))
                predictor.set_image(image_numpy)
                accumulated_coords = current_initial_points[0, :, :2].clone().detach().to(device)
                accumulated_labels = current_initial_points[0, :, 2].clone().detach().to(device)
                last_pred_mask_prob = None
                for round_num in range(num_interaction_rounds + 1):
                    if accumulated_coords.shape[0] == 0: break
                    current_coords_batch = accumulated_coords.unsqueeze(0)
                    current_labels_batch = accumulated_labels.unsqueeze(0)
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                            points=(current_coords_batch, current_labels_batch),
                            boxes=None,
                            masks=None
                        )
                        image_embed = predictor._features["image_embed"][-1].unsqueeze(0)
                        image_pe = predictor.model.sam_prompt_encoder.get_dense_pe()
                        
                        # Modified approach for high res features - create dummy tensors if needed
                        high_res_features = None
                        try:
                            if "high_res_feats" in predictor._features:
                                high_res_feats = predictor._features["high_res_feats"]
                                if isinstance(high_res_feats, (list, tuple)) and len(high_res_feats) >= 2:
                                    # Safely extract features
                                    if high_res_feats[0] is not None and len(high_res_feats[0]) > 0:
                                        feat_s0 = high_res_feats[0][-1]
                                        if not isinstance(feat_s0, torch.Tensor):
                                            feat_s0 = None
                                    else:
                                        feat_s0 = None
                                        
                                    if high_res_feats[1] is not None and len(high_res_feats[1]) > 0:
                                        feat_s1 = high_res_feats[1][-1]
                                        if not isinstance(feat_s1, torch.Tensor):
                                            feat_s1 = None
                                    else:
                                        feat_s1 = None
                                    
                                    # If both features are valid, create the list
                                    if feat_s0 is not None and feat_s1 is not None:
                                        high_res_features = [
                                            feat_s0.unsqueeze(0).to(device),
                                            feat_s1.unsqueeze(0).to(device)
                                        ]
                        except Exception as e:
                            print(f"Warning: Error processing high res features: {e}")
                            high_res_features = None

                        # Ensure all inputs are on the same device
                        image_embed = image_embed.to(device)
                        image_pe = image_pe.to(device)
                        sparse_embeddings = sparse_embeddings.to(device)
                        dense_embeddings = dense_embeddings.to(device)
                        
                        # If high_res_features is missing, create zero tensors with appropriate shapes
                        # This is a workaround to satisfy the unpacking in mask_decoder.py
                        if high_res_features is None:
                            # Create dummy tensors with appropriate shapes
                            # These are zero tensors that shouldn't affect the output but will satisfy the unpacking
                            dummy_shape = (1, 64, image_embed.shape[2]//2, image_embed.shape[3]//2)  # Adjust dimensions as needed
                            dummy_tensor = torch.zeros(dummy_shape, device=device)
                            high_res_features = [dummy_tensor, dummy_tensor.clone()]
                        
                        # Now we can always use high_res_features without a conditional check
                        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                            image_embeddings=image_embed,
                            image_pe=image_pe,
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=True,
                            high_res_features=high_res_features,
                            repeat_image=True
                        )
                    prd_masks_processed = predictor._transforms.postprocess_masks(low_res_masks.cpu(), predictor._orig_hw[-1]).to(device)
                    best_mask_idx = torch.argmax(prd_scores, dim=1); prd_mask_best = prd_masks_processed[torch.arange(prd_masks_processed.shape[0]), best_mask_idx]
                    last_pred_mask_prob = torch.sigmoid(prd_mask_best)
                    if round_num < num_interaction_rounds:
                        pred_mask_binary_np = (last_pred_mask_prob[0].detach().cpu().numpy() > 0.5)
                        gt_mask_np = current_mask_gt[0].detach().cpu().numpy()
                        fp_mask, fn_mask = calculate_error_regions(pred_mask_binary_np, gt_mask_np)
                        fp_points, fp_labels = sample_points_from_mask(fp_mask, points_per_round, 0)
                        fn_points, fn_labels = sample_points_from_mask(fn_mask, points_per_round, 1)
                        new_points = np.concatenate((fp_points, fn_points), axis=0) if len(fp_points)>0 and len(fn_points)>0 else fp_points if len(fp_points)>0 else fn_points if len(fn_points)>0 else np.array([])
                        new_labels = np.concatenate((fp_labels, fn_labels), axis=0) if len(fp_labels)>0 and len(fn_labels)>0 else fp_labels if len(fp_labels)>0 else fn_labels if len(fn_labels)>0 else np.array([])
                        if new_points.shape[0] > 0:
                            new_points_tensor = torch.tensor(new_points, dtype=torch.float32).to(device)
                            new_labels_tensor = torch.tensor(new_labels, dtype=torch.float32).to(device)
                            accumulated_coords = torch.cat((accumulated_coords, new_points_tensor), dim=0)
                            accumulated_labels = torch.cat((accumulated_labels, new_labels_tensor), dim=0)
                if last_pred_mask_prob is not None:
                    current_mask_gt_squeezed = current_mask_gt.squeeze(0)
                    final_pred_prob = last_pred_mask_prob[0]
                    focal_comp = focal_loss(final_pred_prob, current_mask_gt_squeezed); dice_comp = dice_loss(final_pred_prob, current_mask_gt_squeezed)
                    final_loss = focal_weight * focal_comp + dice_weight * dice_comp
                    batch_final_losses.append(final_loss)
                    final_pred_binary = (final_pred_prob > 0.5).float()
                    inter = (current_mask_gt_squeezed * final_pred_binary).sum(); union = current_mask_gt_squeezed.sum() + final_pred_binary.sum() - inter
                    final_iou = (inter + 1e-6) / (union + 1e-6)
                    batch_final_ious.append(final_iou.item())
            if batch_final_losses:
                avg_batch_loss = sum(batch_final_losses) / len(batch_final_losses); scaled_loss = avg_batch_loss / accumulation_steps
                scaler.scale(scaled_loss).backward()
                total_final_loss += avg_batch_loss.item() * len(batch_final_losses); total_final_iou += sum(batch_final_ious); processed_samples_count += len(batch_final_losses)
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                if any(p.grad is not None for p in predictor.model.parameters()): scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0); scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            current_lr = optimizer.param_groups[-1]['lr']; avg_epoch_loss = total_final_loss / processed_samples_count if processed_samples_count > 0 else 0; avg_epoch_iou = total_final_iou / processed_samples_count if processed_samples_count > 0 else 0
            tepoch.set_postfix({"loss": avg_epoch_loss, "iou": avg_epoch_iou, "lr": f"{current_lr:.2e}"})
    final_avg_loss = total_final_loss / processed_samples_count if processed_samples_count > 0 else 0
    final_avg_iou = total_final_iou / processed_samples_count if processed_samples_count > 0 else 0
    return final_avg_loss, final_avg_iou

def validate_epoch_interactive(
    predictor, test_dataloader, epoch, device, NO_OF_EPOCHS,
    focal_weight=0.5, dice_weight=0.5,
    num_interaction_rounds=3, points_per_round=1
    ):
    predictor.model.eval(); total_final_loss = 0.0; total_final_iou = 0.0; processed_samples_count = 0
    with torch.no_grad():
        with tqdm(test_dataloader, desc=f"Validation Epoch {epoch+1}/{NO_OF_EPOCHS} (Interact)", unit="batch") as tepoch:
            for i, batch_data in enumerate(tepoch):
                images, masks_gt, initial_points_batch, target_size = batch_data
                images, masks_gt = images.to(device), masks_gt.to(device)
                batch_final_losses, batch_final_ious = [], []
                for j in range(images.shape[0]):
                    current_image, current_mask_gt, current_initial_points = images[j], masks_gt[j], initial_points_batch[j]
                    # Convert CHW tensor to HWC numpy array for set_image
                    image_numpy = current_image.cpu().numpy().transpose((1, 2, 0))
                    predictor.set_image(image_numpy)
                    accumulated_coords = current_initial_points[0, :, :2].clone().detach().to(device); accumulated_labels = current_initial_points[0, :, 2].clone().detach().to(device)
                    last_pred_mask_prob = None
                    for round_num in range(num_interaction_rounds + 1):
                        if accumulated_coords.shape[0] == 0: break
                        current_coords_batch = accumulated_coords.unsqueeze(0); current_labels_batch = accumulated_labels.unsqueeze(0)
                        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                                points=(current_coords_batch, current_labels_batch),
                                boxes=None,
                                masks=None
                            )
                            image_embed = predictor._features["image_embed"][-1].unsqueeze(0)
                            image_pe = predictor.model.sam_prompt_encoder.get_dense_pe()
                            
                            # Modified approach for high res features - create dummy tensors if needed
                            high_res_features = None
                            try:
                                if "high_res_feats" in predictor._features:
                                    high_res_feats = predictor._features["high_res_feats"]
                                    if isinstance(high_res_feats, (list, tuple)) and len(high_res_feats) >= 2:
                                        # Safely extract features
                                        if high_res_feats[0] is not None and len(high_res_feats[0]) > 0:
                                            feat_s0 = high_res_feats[0][-1]
                                            if not isinstance(feat_s0, torch.Tensor):
                                                feat_s0 = None
                                        else:
                                            feat_s0 = None
                                            
                                        if high_res_feats[1] is not None and len(high_res_feats[1]) > 0:
                                            feat_s1 = high_res_feats[1][-1]
                                            if not isinstance(feat_s1, torch.Tensor):
                                                feat_s1 = None
                                        else:
                                            feat_s1 = None
                                        
                                        # If both features are valid, create the list
                                        if feat_s0 is not None and feat_s1 is not None:
                                            high_res_features = [
                                                feat_s0.unsqueeze(0).to(device),
                                                feat_s1.unsqueeze(0).to(device)
                                            ]
                            except Exception as e:
                                print(f"Warning: Error processing high res features: {e}")
                                high_res_features = None

                            # Ensure all inputs are on the same device
                            image_embed = image_embed.to(device)
                            image_pe = image_pe.to(device)
                            sparse_embeddings = sparse_embeddings.to(device)
                            dense_embeddings = dense_embeddings.to(device)
                            
                            # If high_res_features is missing, create zero tensors with appropriate shapes
                            # This is a workaround to satisfy the unpacking in mask_decoder.py
                            if high_res_features is None:
                                # Create dummy tensors with appropriate shapes
                                # These are zero tensors that shouldn't affect the output but will satisfy the unpacking
                                dummy_shape = (1, 64, image_embed.shape[2]//2, image_embed.shape[3]//2)  # Adjust dimensions as needed
                                dummy_tensor = torch.zeros(dummy_shape, device=device)
                                high_res_features = [dummy_tensor, dummy_tensor.clone()]
                            
                            # Now we can always use high_res_features without a conditional check
                            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                                image_embeddings=image_embed,
                                image_pe=image_pe,
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=True,
                                high_res_features=high_res_features,
                                repeat_image=True
                            )
                        prd_masks_processed = predictor._transforms.postprocess_masks(low_res_masks.cpu(), predictor._orig_hw[-1]).to(device)
                        best_mask_idx = torch.argmax(prd_scores, dim=1); prd_mask_best = prd_masks_processed[torch.arange(prd_masks_processed.shape[0]), best_mask_idx]
                        last_pred_mask_prob = torch.sigmoid(prd_mask_best)
                        if round_num < num_interaction_rounds:
                            pred_mask_binary_np = (last_pred_mask_prob[0].cpu().numpy() > 0.5); gt_mask_np = current_mask_gt[0].cpu().numpy()
                            fp_mask, fn_mask = calculate_error_regions(pred_mask_binary_np, gt_mask_np)
                            fp_points, fp_labels = sample_points_from_mask(fp_mask, points_per_round, 0); fn_points, fn_labels = sample_points_from_mask(fn_mask, points_per_round, 1)
                            new_points = np.concatenate((fp_points, fn_points), axis=0) if len(fp_points)>0 and len(fn_points)>0 else fp_points if len(fp_points)>0 else fn_points if len(fn_points)>0 else np.array([])
                            new_labels = np.concatenate((fp_labels, fn_labels), axis=0) if len(fp_labels)>0 and len(fn_labels)>0 else fp_labels if len(fp_labels)>0 else fn_labels if len(fn_labels)>0 else np.array([])
                            if new_points.shape[0] > 0:
                                new_points_tensor = torch.tensor(new_points, dtype=torch.float32).to(device); new_labels_tensor = torch.tensor(new_labels, dtype=torch.float32).to(device)
                                accumulated_coords = torch.cat((accumulated_coords, new_points_tensor), dim=0); accumulated_labels = torch.cat((accumulated_labels, new_labels_tensor), dim=0)
                    if last_pred_mask_prob is not None:
                        current_mask_gt_squeezed = current_mask_gt.squeeze(0); final_pred_prob = last_pred_mask_prob[0]
                        focal_comp = focal_loss(final_pred_prob, current_mask_gt_squeezed); dice_comp = dice_loss(final_pred_prob, current_mask_gt_squeezed)
                        final_loss = focal_weight * focal_comp + dice_weight * dice_comp
                        final_pred_binary = (final_pred_prob > 0.5).float()
                        inter = (current_mask_gt_squeezed * final_pred_binary).sum(); union = current_mask_gt_squeezed.sum() + final_pred_binary.sum() - inter
                        final_iou = (inter + 1e-6) / (union + 1e-6)
                        batch_final_losses.append(final_loss.item()); batch_final_ious.append(final_iou.item())
                if batch_final_losses: total_final_loss += sum(batch_final_losses); total_final_iou += sum(batch_final_ious); processed_samples_count += len(batch_final_losses)
                avg_epoch_loss = total_final_loss / processed_samples_count if processed_samples_count > 0 else 0; avg_epoch_iou = total_final_iou / processed_samples_count if processed_samples_count > 0 else 0
                tepoch.set_postfix({"loss": avg_epoch_loss, "iou": avg_epoch_iou})
    final_avg_loss = total_final_loss / processed_samples_count if processed_samples_count > 0 else 0
    final_avg_iou = total_final_iou / processed_samples_count if processed_samples_count > 0 else 0
    return final_avg_loss, final_avg_iou


# ----- Visualization Function -----
# (visualize_predictions_interactive remains the same)
def visualize_predictions_interactive(predictor, test_dataloader, device, epoch, save_dir,
                                       num_interaction_rounds=3, points_per_round=1, num_samples=4):
    ensure_dir(save_dir); predictor.model.eval(); dataset = test_dataloader.dataset
    samples_to_show = min(num_samples, len(dataset)); ncols = 3; nrows = samples_to_show
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5));
    if nrows == 1: axes = axes.reshape(1, ncols)
    indices = np.random.choice(len(dataset), samples_to_show, replace=False)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_tensor, mask_tensor, initial_points_tensor, _ = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device); mask_tensor = mask_tensor.unsqueeze(0).to(device); initial_points_tensor = initial_points_tensor.to(device)
            # Convert CHW tensor to HWC numpy array for set_image
            image_numpy = image_tensor[0].cpu().numpy().transpose((1, 2, 0))
            predictor.set_image(image_numpy)
            accumulated_coords = initial_points_tensor[0, :, :2].clone().detach(); accumulated_labels = initial_points_tensor[0, :, 2].clone().detach()
            last_pred_mask_prob = None
            for round_num in range(num_interaction_rounds + 1): # Simulate interaction
                if accumulated_coords.shape[0] == 0: break
                current_coords_batch = accumulated_coords.unsqueeze(0); current_labels_batch = accumulated_labels.unsqueeze(0)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=(current_coords_batch, current_labels_batch),
                        boxes=None,
                        masks=None
                    )
                    image_embed = predictor._features["image_embed"][-1].unsqueeze(0)
                    image_pe = predictor.model.sam_prompt_encoder.get_dense_pe()
                    
                    # Modified approach for high res features - create dummy tensors if needed
                    high_res_features = None
                    try:
                        if "high_res_feats" in predictor._features:
                            high_res_feats = predictor._features["high_res_feats"]
                            if isinstance(high_res_feats, (list, tuple)) and len(high_res_feats) >= 2:
                                # Safely extract features
                                if high_res_feats[0] is not None and len(high_res_feats[0]) > 0:
                                    feat_s0 = high_res_feats[0][-1]
                                    if not isinstance(feat_s0, torch.Tensor):
                                        feat_s0 = None
                                else:
                                    feat_s0 = None
                                    
                                if high_res_feats[1] is not None and len(high_res_feats[1]) > 0:
                                    feat_s1 = high_res_feats[1][-1]
                                    if not isinstance(feat_s1, torch.Tensor):
                                        feat_s1 = None
                                else:
                                    feat_s1 = None
                                
                                # If both features are valid, create the list
                                if feat_s0 is not None and feat_s1 is not None:
                                    high_res_features = [
                                        feat_s0.unsqueeze(0).to(device),
                                        feat_s1.unsqueeze(0).to(device)
                                    ]
                    except Exception as e:
                        print(f"Warning: Error processing high res features: {e}")
                        high_res_features = None
                        
                        if not high_res_features:  # If list is empty after processing
                            high_res_features = None
                    # Ensure all inputs are on the same device
                    if high_res_features:
                        high_res_features = [f.to(device) for f in high_res_features]
                    image_embed = image_embed.to(device)
                    image_pe = image_pe.to(device)
                    sparse_embeddings = sparse_embeddings.to(device)
                    dense_embeddings = dense_embeddings.to(device)
                    
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=image_embed,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        high_res_features=high_res_features if high_res_features else None,
                        repeat_image=True  # Add this parameter
                    )
                prd_masks_processed = predictor._transforms.postprocess_masks(low_res_masks.cpu(), predictor._orig_hw[-1]).to(device)
                best_mask_idx = torch.argmax(prd_scores, dim=1); prd_mask_best = prd_masks_processed[torch.arange(prd_masks_processed.shape[0]), best_mask_idx]
                last_pred_mask_prob = torch.sigmoid(prd_mask_best)
                if round_num < num_interaction_rounds:
                    pred_mask_binary_np = (last_pred_mask_prob[0].cpu().numpy() > 0.5); gt_mask_np = mask_tensor[0,0].cpu().numpy()
                    fp_mask, fn_mask = calculate_error_regions(pred_mask_binary_np, gt_mask_np)
                    fp_points, fp_labels = sample_points_from_mask(fp_mask, points_per_round, 0); fn_points, fn_labels = sample_points_from_mask(fn_mask, points_per_round, 1)
                    new_points = np.concatenate((fp_points, fn_points), axis=0) if len(fp_points)>0 and len(fn_points)>0 else fp_points if len(fp_points)>0 else fn_points if len(fn_points)>0 else np.array([])
                    new_labels = np.concatenate((fp_labels, fn_labels), axis=0) if len(fp_labels)>0 and len(fn_labels)>0 else fp_labels if len(fp_labels)>0 else fn_labels if len(fn_labels)>0 else np.array([])
                    if new_points.shape[0] > 0:
                        new_points_tensor = torch.tensor(new_points, dtype=torch.float32).to(device); new_labels_tensor = torch.tensor(new_labels, dtype=torch.float32).to(device)
                        accumulated_coords = torch.cat((accumulated_coords, new_points_tensor), dim=0); accumulated_labels = torch.cat((accumulated_labels, new_labels_tensor), dim=0)
            pred_np = last_pred_mask_prob[0].cpu().numpy() if last_pred_mask_prob is not None else np.zeros_like(mask_tensor[0, 0].cpu().numpy())
            image_disp = np.clip(image_numpy, 0, 1) # Display image as loaded ([0,1] float)
            mask_np = mask_tensor[0, 0].cpu().numpy()
            ax_img, ax_gt, ax_pred = axes[i, 0], axes[i, 1], axes[i, 2]
            ax_img.imshow(image_disp); initial_points_np_vis = initial_points_tensor[0].cpu().numpy()
            if initial_points_np_vis.shape[0] > 0: ax_img.scatter(initial_points_np_vis[:, 0], initial_points_np_vis[:, 1], c='red', marker='*', s=50, edgecolor='white', linewidth=0.5)
            ax_img.set_title(f"Input Image {idx} (Initial Pts)"); ax_img.axis("off")
            ax_gt.imshow(mask_np, cmap='gray'); ax_gt.set_title("Ground Truth"); ax_gt.axis("off")
            ax_pred.imshow(pred_np > 0.5, cmap='gray'); ax_pred.set_title("Final Prediction"); ax_pred.axis("off")
    plt.tight_layout(); save_path = os.path.join(save_dir, f"predictions_interactive_epoch_{epoch+1}.png")
    plt.savefig(save_path); print(f"Saved viz to {save_path}"); plt.close(fig)

# ----- Early Stopping Implementation -----
# (EarlyStopping class remains the same)
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0.001, mode='max', path='best_model.pt'):
        self.patience, self.verbose, self.delta, self.mode, self.path = patience, verbose, delta, mode, path
        self.counter, self.best_score, self.early_stop, self.improved = 0, None, False, False
        assert mode in ['min', 'max']
        # Use np.inf instead of np.Inf
        self.val_score_ref = -np.inf if mode == 'max' else np.inf

    def __call__(self, current_score, model_state=None, optimizer_state=None, epoch=-1, scheduler_state=None, model_cfg=None):
        score = current_score
        if self.best_score is None: self.best_score = score; self.save_checkpoint(score, model_state, optimizer_state, epoch, scheduler_state, model_cfg); self.improved = True; return False
        improved_flag = (self.mode == 'max' and score > self.best_score + self.delta) or (self.mode == 'min' and score < self.best_score - self.delta)
        if improved_flag:
            if self.verbose: print(f' EarlyStop: Score improved ({self.best_score:.6f} -> {score:.6f}). Save model.')
            self.best_score = score; self.save_checkpoint(score, model_state, optimizer_state, epoch, scheduler_state, model_cfg); self.counter = 0; self.improved = True
        else:
            self.counter += 1; self.improved = False
            if self.verbose: print(f' EarlyStop: No improvement {self.counter}/{self.patience}.')
            if self.counter >= self.patience: self.early_stop = True; print(f' EarlyStop: Triggered.')
        return self.early_stop

    def save_checkpoint(self, score, model_state, optimizer_state, epoch, scheduler_state, model_cfg):
        if self.verbose: print(f' Saving best model to {self.path} (Score: {score:.6f})')
        save_dict = {'epoch': epoch + 1, 'model_state_dict': model_state, f'best_val_{self.mode}': self.best_score, 'model_cfg': model_cfg}
        if optimizer_state: save_dict['optimizer_state_dict'] = optimizer_state
        if scheduler_state: save_dict['scheduler_state_dict'] = scheduler_state
        torch.save(save_dict, self.path)

# ----- Main Execution Function -----
# (main function remains the same, just ensure it calls the interactive train/validate functions
# and uses the interactive dataset)
def main():
    # --- Configuration ---
    SEED = 42; set_seed(SEED)
    DATA_DIR = "Aug_dataset_cor_arteries"
    IMAGE_DIR = os.path.join(DATA_DIR, "Augmented_image")
    MASK_DIR = os.path.join(DATA_DIR, "Augmented_mask")
    SAM2_CHECKPOINT = "sam2_hiera_tiny.pt"
    MODEL_CFG = "sam2_hiera_t.yaml"

    # --- Hyperparameters ---
    BATCH_SIZE = 1 # *** Keep low for interactive ***
    TARGET_SIZE = 1024
    INITIAL_MAX_POINTS = 3
    NUM_INTERACTION_ROUNDS = 2
    POINTS_PER_INTERACTION_ROUND = 3
    ENCODER_LR_FACTOR = 0.1
    # WEIGHT_DECAY applied in get_optimizer_and_scheduler
    ACCUMULATION_STEPS = 32 # *** Adjust based on BS=1 ***
    NO_OF_EPOCHS = 50
    CHECKPOINT_INTERVAL = 5
    VISUALIZATION_INTERVAL = 5
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_DELTA = 0.001
    FOCAL_LOSS_WEIGHT = 0.5
    DICE_LOSS_WEIGHT = 0.5
    RUN_NAME = f"interactive_noPreproc_R{NUM_INTERACTION_ROUNDS}_P{POINTS_PER_INTERACTION_ROUND}_{MODEL_CFG.split('.')[0]}"
    FINE_TUNED_MODEL_NAME = f"fine_tuned_{RUN_NAME}"
    FINE_TUNED_DIR = ensure_dir(FINE_TUNED_MODEL_NAME)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run Dir: {FINE_TUNED_DIR}\nDevice: {DEVICE}")
    print(f"Rounds: {NUM_INTERACTION_ROUNDS}, Pts/Round: {POINTS_PER_INTERACTION_ROUND}")

    # --- Data Loading ---
    print("Loading data...")
    train_df_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_df_path): print(f"Error: train.csv not found..."); return
    train_df = pd.read_csv(train_df_path)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=SEED)
    train_data = create_data_list(train_df, IMAGE_DIR, MASK_DIR)
    test_data = create_data_list(test_df, IMAGE_DIR, MASK_DIR)
    # Use the Interactive Dataset
    train_dataset = CoronaryArteryDatasetInteractive(train_data, initial_max_points=INITIAL_MAX_POINTS, augment=True, target_size=TARGET_SIZE)
    test_dataset = CoronaryArteryDatasetInteractive(test_data, initial_max_points=INITIAL_MAX_POINTS, augment=False, target_size=TARGET_SIZE)
    num_workers = 0 # Keep low for interactive debug
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    print("Data loaded.")

    # --- Model Initialization ---
    print(f"Loading model: {MODEL_CFG} from {SAM2_CHECKPOINT}")
    try:
        sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
        for param in sam2_model.parameters(): param.requires_grad = True
        sam2_model.to(DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
        print("Model loaded.")
    except Exception as e: print(f"Error loading model: {e}"); return

    # --- Optimizer, Scheduler, Scaler ---
    params_to_optimize = list(predictor.model.parameters())
    print(f"Optimizing {sum(p.numel() for p in params_to_optimize)} parameters.")
    optimizer, scheduler = get_optimizer_and_scheduler(predictor.model, NO_OF_EPOCHS, ENCODER_LR_FACTOR)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # --- Training Loop ---
    training_history, validation_history = [], []
    best_val_iou = -1.0
    BEST_MODEL_PATH = os.path.join(FINE_TUNED_DIR, "best_model_interactive_noPreproc.pt")
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, delta=EARLY_STOPPING_DELTA, mode='max', path=BEST_MODEL_PATH)

    print("\n--- Starting Interactive Fine-Tuning (No Internal Preprocessing) ---")
    start_epoch = 0
    for epoch in range(start_epoch, NO_OF_EPOCHS):
        current_epoch_display = epoch + 1
        # Call INTERACTIVE train/validate functions
        train_loss, train_mean_iou = train_epoch_interactive(predictor, train_dataloader, epoch, ACCUMULATION_STEPS, optimizer, scaler, DEVICE, NO_OF_EPOCHS, FOCAL_LOSS_WEIGHT, DICE_LOSS_WEIGHT, NUM_INTERACTION_ROUNDS, POINTS_PER_INTERACTION_ROUND)
        valid_loss, valid_mean_iou = validate_epoch_interactive(predictor, test_dataloader, epoch, DEVICE, NO_OF_EPOCHS, FOCAL_LOSS_WEIGHT, DICE_LOSS_WEIGHT, NUM_INTERACTION_ROUNDS, POINTS_PER_INTERACTION_ROUND)
        scheduler.step()
        current_lr = optimizer.param_groups[-1]['lr']
        training_history.append({"epoch": current_epoch_display, "loss": train_loss, "mean_iou": train_mean_iou, "lr": current_lr})
        validation_history.append({"epoch": current_epoch_display, "loss": valid_loss, "mean_iou": valid_mean_iou})
        print(f"\nEpoch {current_epoch_display}/{NO_OF_EPOCHS}: Train IoU:{train_mean_iou:.4f} Loss:{train_loss:.4f} | Valid IoU:{valid_mean_iou:.4f} Loss:{valid_loss:.4f} | LR:{current_lr:.3e}")
        if current_epoch_display % VISUALIZATION_INTERVAL == 0 or epoch == start_epoch:
             visualize_predictions_interactive(predictor, test_dataloader, DEVICE, epoch, FINE_TUNED_DIR, NUM_INTERACTION_ROUNDS, POINTS_PER_INTERACTION_ROUND)
        stop_training = early_stopping(valid_mean_iou, predictor.model.state_dict(), optimizer.state_dict(), epoch, scheduler.state_dict(), MODEL_CFG)
        if early_stopping.improved: best_val_iou = early_stopping.best_score
        if stop_training: print(f"Early stopping."); break
        if current_epoch_display % CHECKPOINT_INTERVAL == 0:
            CHECKPOINT_PATH = os.path.join(FINE_TUNED_DIR, f"checkpoint_epoch_{current_epoch_display}.pt")
            torch.save({ 'epoch': current_epoch_display, 'model_state_dict': predictor.model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'val_iou': valid_mean_iou, 'model_cfg': MODEL_CFG }, CHECKPOINT_PATH)
            print(f" Saved checkpoint: {CHECKPOINT_PATH}")

    print("\n--- Training Complete ---")
    if not stop_training: print("Reached max epochs.")

    # --- Save History and Plot ---
    # (Save/plot logic remains the same)
    training_history_serializable = [{'epoch': int(e['epoch']), 'loss': float(e['loss']), 'iou': float(e['mean_iou']), 'lr': float(e.get('lr', 0))} for e in training_history]
    validation_history_serializable = [{'epoch': int(e['epoch']), 'loss': float(e['loss']), 'iou': float(e['mean_iou'])} for e in validation_history]
    history_filename_base = os.path.join(FINE_TUNED_DIR, "training_history_interactive_noPreproc")
    try:
        with open(f'{history_filename_base}_train.json', 'w') as f: json.dump(training_history_serializable, f, indent=4)
        with open(f'{history_filename_base}_val.json', 'w') as f: json.dump(validation_history_serializable, f, indent=4)
        print(f"Saved history to {history_filename_base}_*.json")
    except Exception as e: print(f"Error saving history: {e}")
    try: # Plotting
        epochs = [x['epoch'] for x in validation_history_serializable]
        plt.figure(figsize=(18, 6)); plt.subplot(1, 3, 1); plt.plot([x['epoch'] for x in training_history_serializable], [x['loss'] for x in training_history_serializable], label='Train Loss', marker='.');
        if epochs: plt.plot(epochs, [x['loss'] for x in validation_history_serializable], label='Valid Loss', marker='.')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend(); plt.grid(True, alpha=0.6)
        plt.subplot(1, 3, 2); plt.plot([x['epoch'] for x in training_history_serializable], [x['iou'] for x in training_history_serializable], label='Train IoU', marker='.');
        if epochs: plt.plot(epochs, [x['iou'] for x in validation_history_serializable], label='Valid IoU', marker='.')
        plt.xlabel('Epoch'); plt.ylabel('Mean IoU'); plt.title('Mean IoU'); plt.legend(); plt.grid(True, alpha=0.6); plt.ylim(bottom=0)
        plt.subplot(1, 3, 3); lrs = [x.get('lr', 0) for x in training_history_serializable];
        if any(lr > 0 for lr in lrs): plt.plot([x['epoch'] for x in training_history_serializable], lrs, label='LR (Decoder)', marker='.'); plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('LR Schedule'); plt.legend(); plt.grid(True, alpha=0.6); plt.yscale('log')
        plt.suptitle(f'Training Metrics: {RUN_NAME}', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = os.path.join(FINE_TUNED_DIR, "training_plots_interactive_noPreproc.png")
        plt.savefig(plot_filename); print(f"Saved plots to {plot_filename}"); plt.close()
    except Exception as e: print(f"Error plotting: {e}")

    print(f"Script finished. Best model saved to: {BEST_MODEL_PATH}")


# --- Script Entry Point ---
if __name__ == "__main__":
    main()

# --- END OF FILE Fine-tune_model_interactive_no_preprocess.py ---