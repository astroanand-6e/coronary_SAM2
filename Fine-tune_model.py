import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random
import json

# Import SAM2 Modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ----- Helper Functions -----

def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = True  # Enable for faster training with fixed input sizes

def ensure_dir(directory):
    """Creates directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def create_data_list(df, image_dir, mask_dir):
    data = []
    for index, row in df.iterrows():
        image_name = row['image_id']
        mask_name = row['mask_id']
        data.append({
            "image": os.path.join(image_dir, image_name),
            "annotation": os.path.join(mask_dir, mask_name)
        })
    return data

# ----- Enhanced Loss Functions -----

def dice_loss(pred, target):
    """Calculate Dice Loss for segmentation tasks."""
    smooth = 1.0
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def combo_loss(pred, target, alpha=0.5):
    """Combined BCE and Dice loss for better segmentation."""
    # BCE Loss component
    bce = (-target * torch.log(pred + 1e-6) - (1 - target) * torch.log((1 - pred) + 1e-6)).mean()
    # Dice Loss component
    dice = dice_loss(pred, target)
    # Combined loss with weighting factor
    return alpha * bce + (1 - alpha) * dice

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """Focal Loss for handling class imbalance in segmentation."""
    bce = -target * torch.log(pred + 1e-6) * (1 - pred) ** gamma - (1 - target) * torch.log((1 - pred) + 1e-6) * pred ** gamma
    return (alpha * bce).mean()

# ----- X-ray Preprocessing -----

def preprocess_xray_image(image):
    """Apply preprocessing specific to X-ray images with vessels."""
    # Convert to float and normalize
    image = image.astype(np.float32)
    
    # Apply CLAHE for better vessel contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(image.shape[2]):
        image[:,:,i] = clahe.apply((image[:,:,i] * 255).astype(np.uint8)) / 255.0
    
    # Normalize to zero mean and unit variance
    image = (image - image.mean()) / (image.std() + 1e-6)
    
    return image

# ----- Model Tuning Functions -----

def setup_parameter_efficient_tuning(predictor):
    """Set up efficient parameter tuning strategy."""
    # Freeze all parameters initially
    for param in predictor.model.parameters():
        param.requires_grad = False
    
    # Enable training for mask decoder
    for param in predictor.model.sam_mask_decoder.parameters():
        param.requires_grad = True
    
    # Enable training for prompt encoder
    for param in predictor.model.sam_prompt_encoder.parameters():
        param.requires_grad = True
    
    # Optionally unfreeze last few layers of image encoder
    for name, param in predictor.model.image_encoder.named_parameters():
        if any(x in name for x in ['block4', 'neck', 'final']):
            param.requires_grad = True
    
    return predictor

def get_optimizer_and_scheduler(params, max_epochs):
    """Create optimizer and scheduler with improved learning rate strategy."""
    # Better initial learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        params,
        lr=4e-4,  # Lower initial learning rate for stable fine-tuning
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Multiply period by 2 after each restart
        eta_min=1e-7  # Minimum learning rate
    )
    
    return optimizer, scheduler

# ----- Improved Dataset Class -----

class CoronaryArteryDataset(Dataset):
    def __init__(self, data, transform=None, max_points=3, augment=False):
        self.data = data
        self.transform = transform
        self.max_points = max_points
        self.augment = augment
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        mask_path = item['annotation']
        
        # Load image and mask
        Img = cv2.imread(image_path)
        if Img is None:
            raise FileNotFoundError(f"Error: Could not read image file at path: {image_path}")
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        
        # Apply vessel-specific preprocessing
        Img = preprocess_xray_image(Img)
        
        ann_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ann_map = 255 - ann_map
        ann_map = ann_map.astype(np.float32) / 255.0
        
        # Resize images and masks if needed
        if Img.shape[0] != 1024 or Img.shape[1] != 1024:
            r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
            Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
            ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                              interpolation=cv2.INTER_NEAREST)
        else:
            ann_map = cv2.resize(ann_map, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentation if enabled
        if self.augment:
            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((Img.shape[1]//2, Img.shape[0]//2), angle, 1.0)
                Img = cv2.warpAffine(Img, M, (Img.shape[1], Img.shape[0]))
                ann_map = cv2.warpAffine(ann_map, M, (ann_map.shape[1], ann_map.shape[0]), 
                                        flags=cv2.INTER_NEAREST)
            
            # Random brightness/contrast adjustment
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # Contrast
                beta = random.uniform(-0.1, 0.1)  # Brightness
                Img = np.clip(alpha * Img + beta, 0, 1)
        
        # Generate vessel-specific points (improved sampling)
        binary_mask = (ann_map > 0.5).astype(np.uint8)
        
        # Skeletonize to find centerlines
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary_mask)
        
        points_list = []
        
        # Sample points along vessel centerlines if available
        if np.any(skeleton):
            y_indices, x_indices = np.where(skeleton)
            if len(y_indices) >= self.max_points:
                # Sample evenly spaced points along skeleton
                indices = np.linspace(0, len(y_indices)-1, self.max_points, dtype=int)
                for i in indices:
                    points_list.append([x_indices[i], y_indices[i]])
            else:
                # Use all available skeleton points
                for i in range(len(y_indices)):
                    points_list.append([x_indices[i], y_indices[i]])
        
        # If not enough points from skeleton, use original point sampling
        if len(points_list) < self.max_points:
            center_y, center_x = binary_mask.shape[0] // 2, binary_mask.shape[1] // 2
            
            def find_nearest_white_pixel(x, y, direction, push_distance=10):
                distance = 0
                while 0 <= x < binary_mask.shape[1] and 0 <= y < binary_mask.shape[0]:
                    if binary_mask[y, x] > 0:
                        break
                    x += direction[0]
                    y += direction[1]
                    distance += 1
                else:
                    return None, None
                
                for _ in range(push_distance):
                    x += direction[0]
                    y += direction[1]
                    if not (0 <= x < binary_mask.shape[1] and 0 <= y < binary_mask.shape[0] and binary_mask[y, x] > 0):
                        x -= direction[0]
                        y -= direction[1]
                        break
                
                return x, y
            
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            nearest_points = []
            
            for direction in directions:
                x, y = center_x, center_y
                nearest_x, nearest_y = find_nearest_white_pixel(x, y, direction)
                if nearest_x is not None and nearest_y is not None:
                    nearest_points.append((nearest_x, nearest_y))
            
            nearest_points.sort(key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
            nearest_points = nearest_points[:self.max_points - len(points_list)]
            
            for point in nearest_points:
                points_list.append([point[0], point[1]])
        
        points_np = np.array(points_list) if points_list else np.zeros((0, 2))
        
        # Padding to ensure consistent number of points
        if len(points_np) < self.max_points:
            padding_needed = self.max_points - len(points_np)
            padding_array = np.zeros((padding_needed, 2))
            points_np = np.concatenate([points_np, padding_array], axis=0)
        
        labels = np.ones(len(points_np))
        points_np = np.concatenate([points_np, labels[:, None]], axis=1)
        points = np.expand_dims(points_np, axis=0)
        
        # Convert to tensors
        image = torch.tensor(Img.transpose((2, 0, 1)).copy())
        mask = torch.tensor(ann_map, dtype=torch.float32).unsqueeze(0)
        
        return image, mask, points, 1  # Simplified num_masks

    def visualize(self, idx):
        # Implementation remains the same
        pass

# ----- Improved Training Function -----

def train_epoch(predictor, train_dataloader, epoch, accumulation_steps, optimizer, scaler, device, NO_OF_EPOCHS):
    predictor.model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = len(train_dataloader)
    mean_iou = 0.0
    
    optimizer.zero_grad()  # Zero gradients at beginning of epoch
    
    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NO_OF_EPOCHS}", unit="batch") as tepoch:
        for i, (images, masks, points, num_masks) in enumerate(tepoch):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            points = points.to(device)
            
            batch_losses = []
            batch_iou_values = []
            
            for j in range(len(images)):
                with torch.amp.autocast(device_type=device.type):
                    # Extract unnorm_coords and labels for this batch item
                    batch_unnorm_coords = points[j, 0, :, :2]
                    batch_labels = points[j, 0, :, 2]
                    batch_unnorm_coords = batch_unnorm_coords.unsqueeze(0)
                    batch_labels = batch_labels.unsqueeze(0)
                    
                    # Convert image to numpy and set the correct format
                    image_numpy = images[j].cpu().numpy().transpose((1, 2, 0))
                    predictor.set_image(image_numpy)
                    
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=(batch_unnorm_coords, batch_labels), boxes=None, masks=None
                    )
                    
                    batched_mode = batch_unnorm_coords.shape[0] > 1
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                    
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=batched_mode,
                        high_res_features=high_res_features,
                    )
                    
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                    prd_mask = torch.sigmoid(prd_masks[:, 0])
                    
                    gt_mask = masks[j].unsqueeze(0)
                    
                    # Calculate IoU
                    inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(-2, -1))
                    union = (gt_mask.sum(dim=(-2, -1)) + (prd_mask > 0.5).sum(dim=(-2, -1)) - inter)
                    iou = inter / (union + 1e-6)
                    
                    # Use improved combo loss
                    loss = combo_loss(prd_mask, gt_mask, alpha=0.5)
                    
                    # Add focal component for thin vessel structures (optional)
                    focal = focal_loss(prd_mask, gt_mask, gamma=2.0) * 0.5
                    loss = loss + focal
                    
                    # Score loss to help decoder predict confidence
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    loss = loss + score_loss * 0.05
                    
                    batch_losses.append(loss)
                    batch_iou_values.append(iou)
            
            # Accumulate batch gradients
            total_loss_batch = sum(batch_losses) / len(batch_losses)
            scaler.scale(total_loss_batch).backward()
            
            # Update weights every accumulation_steps or at the end of epoch
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Compute metrics
            total_iou_batch = torch.stack(batch_iou_values)
            mean_iou += torch.mean(total_iou_batch).item()
            total_loss += sum([l.item() for l in batch_losses])
            
            tepoch.set_postfix({
                "loss": total_loss / (i + 1),
                "iou": mean_iou / (i + 1),
                "lr": optimizer.param_groups[0]["lr"],
            })
    
    avg_loss = total_loss / num_batches
    avg_iou = mean_iou / num_batches
    
    return avg_loss, avg_iou

# ----- Validation Function -----

def validate_epoch(predictor, test_dataloader, epoch, device, NO_OF_EPOCHS):
    predictor.model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = len(test_dataloader)
    mean_iou = 0.0
    
    with torch.no_grad():
        with tqdm(test_dataloader, desc=f"Validation Epoch {epoch+1}/{NO_OF_EPOCHS}", unit="batch") as tepoch:
            for i, (images, masks, points, num_masks) in enumerate(tepoch):
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                points = points.to(device)
                
                batch_losses = []
                batch_iou_values = []
                
                for j in range(len(images)):
                    # Extract unnorm_coords and labels
                    batch_unnorm_coords = points[j, 0, :, :2]
                    batch_labels = points[j, 0, :, 2]
                    batch_unnorm_coords = batch_unnorm_coords.unsqueeze(0)
                    batch_labels = batch_labels.unsqueeze(0)
                    
                    if batch_unnorm_coords is None or batch_labels is None or batch_unnorm_coords.shape[0] == 0:
                        continue
                    
                    # Convert image to numpy
                    image_numpy = images[j].cpu().numpy().transpose((1, 2, 0))
                    predictor.set_image(image_numpy)
                    
                    with torch.amp.autocast(device_type=device.type):
                        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                            points=(batch_unnorm_coords, batch_labels), boxes=None, masks=None
                        )
                        
                        batched_mode = batch_unnorm_coords.shape[0] > 1
                        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                        
                        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=True,
                            repeat_image=batched_mode,
                            high_res_features=high_res_features,
                        )
                        
                        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                        prd_mask = torch.sigmoid(prd_masks[:, 0])
                        
                        gt_mask = masks[j].unsqueeze(0)
                        
                        # Calculate IoU
                        inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(-2, -1))
                        union = (gt_mask.sum(dim=(-2, -1)) + (prd_mask > 0.5).sum(dim=(-2, -1)) - inter)
                        iou = inter / (union + 1e-6)
                        
                        # Use same loss as training for consistency
                        loss = combo_loss(prd_mask, gt_mask, alpha=0.5)
                        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                        loss = loss + score_loss * 0.05
                        
                        batch_losses.append(loss)
                        batch_iou_values.append(iou)
                
                # Compute metrics
                total_iou_batch = torch.stack(batch_iou_values) if batch_iou_values else torch.zeros(1)
                total_iou += torch.mean(total_iou_batch).item()
                total_loss += sum([l.item() for l in batch_losses]) if batch_losses else 0
                mean_iou += torch.mean(total_iou_batch).item() if batch_iou_values else 0
                
                tepoch.set_postfix({"loss": total_loss / (i + 1), "iou": mean_iou / (i+1)})
    
    avg_loss = total_loss / num_batches
    avg_iou = mean_iou / num_batches
    
    return avg_loss, avg_iou

# ----- Visualization Function -----

def visualize_predictions(predictor, test_dataloader, device, epoch, save_dir, num_samples=4):
    """Visualize and save model predictions periodically."""
    predictor.model.eval()
    samples = min(num_samples, len(test_dataloader.dataset))
    
    fig, axes = plt.subplots(samples, 3, figsize=(15, 5*samples))
    
    indices = np.random.choice(len(test_dataloader.dataset), samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask, points, _ = test_dataloader.dataset[idx]
            
            # Process single image
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            # Convert points numpy array to tensor before moving to device
            points = torch.tensor(points).to(device) # points shape is likely (1, max_points, 3)
            
            # Extract coordinates and labels - Corrected indexing
            # Remove the extra '0' index
            unnorm_coords = points[0, :, :2].unsqueeze(0) # Shape becomes (1, max_points, 2)
            labels = points[0, :, 2].unsqueeze(0)        # Shape becomes (1, max_points)
            
            # Convert to numpy for prediction
            image_numpy = image[0].cpu().numpy().transpose((1, 2, 0))
            predictor.set_image(image_numpy)
            
            # Get prediction
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )
            
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            
            # Determine batched_mode (should be False here as we process one image)
            batched_mode = unnorm_coords.shape[0] > 1 

            low_res_masks, _, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode, # Add repeat_image argument
                high_res_features=high_res_features,
            )
            
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            
            # Convert tensors to numpy for visualization
            image_np = image[0].cpu().numpy().transpose((1, 2, 0))
            mask_np = mask[0, 0].cpu().numpy()
            pred_np = prd_mask[0].cpu().numpy()
            
            # Normalize image for visualization
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
            
            # Plot
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(pred_np > 0.5, cmap='gray')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"predictions_epoch_{epoch+1}.png"))
    plt.close(fig)

# ----- Early Stopping Implementation -----

class EarlyStopping:
    """Early stopping to prevent overfitting and save best model.
    
    Args:
        patience (int): Number of epochs to wait after last improvement.
        verbose (bool): If True, prints message for each validation improvement.
        delta (float): Minimum change to qualify as improvement.
        mode (str): 'min' for loss, 'max' for metrics like accuracy/IoU.
    """
    def __init__(self, patience=10, verbose=True, delta=0.001, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        assert mode in ['min', 'max'], "Mode must be either 'min' or 'max'"
        self.mode = mode
        self.improved = False
        
    def __call__(self, current_score):
        # For max mode, higher is better (IoU, accuracy)
        # For min mode, lower is better (loss)
        if self.mode == 'min':
            score = -current_score
        else:
            score = current_score
            
        # First epoch
        if self.best_score is None:
            self.best_score = score
            self.improved = True
            return False
            
        # Check if improved
        if score < self.best_score + self.delta:
            # No improvement
            self.counter += 1
            self.improved = False
            if self.verbose:
                print(f" EarlyStopping: No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f" EarlyStopping: Triggered! No improvement for {self.patience} epochs.")
            return self.early_stop
        else:
            # Improvement
            self.best_score = score
            self.counter = 0
            self.improved = True
            if self.verbose:
                print(f" EarlyStopping: Metric improved.")
            return False

# ----- Main Execution Function -----

def main():
    # --- Configuration and Hyperparameters ---
    SEED = 42
    set_seed(SEED)
    
    # Data Paths
    DATA_DIR = "Aug_dataset_cor_arteries"
    IMAGE_DIR = os.path.join(DATA_DIR, "Augmented_image")
    MASK_DIR = os.path.join(DATA_DIR, "Augmented_mask")
    
    # Model Configuration
    SAM2_CHECKPOINT = "sam2_hiera_large.pt"
    MODEL_CFG = "sam2_hiera_l.yaml"
    
    # Training Hyperparameters
    BATCH_SIZE = 4
    MAX_POINTS = 5  # Increased from 3 to 5 for better vessel coverage
    LEARNING_RATE = 4e-4  # Lowered from 1e-3 to 5e-5
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 8  # Increased from 4 to 8
    NO_OF_EPOCHS = 100
    CHECKPOINT_INTERVAL = 5
    VISUALIZATION_INTERVAL = 10  # Visualize predictions every N epochs
    EARLY_STOPPING_PATIENCE = 15  # Number of epochs with no improvement to wait
    EARLY_STOPPING_DELTA = 0.001  # Minimum improvement needed (0.1% IoU)
    FINE_TUNED_MODEL_NAME = f"fine_tuned_{MODEL_CFG.split('.')[0]}"
    
    # Create a dedicated folder for this fine-tuning run
    FINE_TUNED_DIR = ensure_dir(FINE_TUNED_MODEL_NAME)
    print(f"Saving checkpoints and history to: {FINE_TUNED_DIR}")
    
    # Device Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # --- Data Loading and Preparation ---
    print(f"Data directory: {DATA_DIR}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Mask directory: {MASK_DIR}")
    
    train_df_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_df_path):
        print(f"Error: train.csv not found at {train_df_path}")
        return
    
    train_df = pd.read_csv(train_df_path)
    print("First 5 rows of train_df:")
    print(train_df.head())
    
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=SEED)
    
    train_data = create_data_list(train_df, IMAGE_DIR, MASK_DIR)
    test_data = create_data_list(test_df, IMAGE_DIR, MASK_DIR)
    
    # Create Datasets and DataLoaders with augmentation for training
    train_dataset = CoronaryArteryDataset(train_data, max_points=MAX_POINTS, augment=True)
    test_dataset = CoronaryArteryDataset(test_data, max_points=MAX_POINTS, augment=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # --- Model Initialization ---
    print(f"Loading model: {MODEL_CFG} with checkpoint: {SAM2_CHECKPOINT}")
    sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Set up parameter-efficient fine-tuning
    predictor = setup_parameter_efficient_tuning(predictor)
    predictor.model.to(DEVICE)
    
    # --- Optimizer and Scheduler ---
    params_to_optimize = [p for p in predictor.model.parameters() if p.requires_grad]
    print(f"Number of parameters to optimize: {sum(p.numel() for p in params_to_optimize)}")
    
    optimizer, scheduler = get_optimizer_and_scheduler(params_to_optimize, NO_OF_EPOCHS)
    
    # Initialize scaler based on device
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    # --- Training Loop ---
    training_history = []
    validation_history = []
    best_val_iou = -1.0
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        verbose=True,
        delta=EARLY_STOPPING_DELTA,
        mode='max'  # We want to maximize IoU
    )
    
    print("\n--- Starting Training ---")
    for epoch in range(NO_OF_EPOCHS):
        # Train one epoch
        train_loss, train_mean_iou = train_epoch(
            predictor, train_dataloader, epoch, ACCUMULATION_STEPS, optimizer, scaler, DEVICE, NO_OF_EPOCHS
        )
        
        # Validate one epoch
        valid_loss, valid_mean_iou = validate_epoch(
            predictor, test_dataloader, epoch, DEVICE, NO_OF_EPOCHS
        )
        
        # Step the scheduler
        scheduler.step()
        
        # Store history
        training_history.append({
            "epoch": epoch + 1,
            "loss": train_loss,
            "mean_iou": train_mean_iou
        })
        
        validation_history.append({
            "epoch": epoch + 1,
            "loss": valid_loss,
            "mean_iou": valid_mean_iou
        })
        
        print(f"\nEpoch {epoch+1}/{NO_OF_EPOCHS}:")
        print(f" Train Loss: {train_loss:.6f}, Train IoU: {train_mean_iou:.6f}")
        print(f" Valid Loss: {valid_loss:.6f}, Valid IoU: {valid_mean_iou:.6f}")
        print(f" Current LR: {optimizer.param_groups[0]['lr']:.6e}")
        
        # Visualize predictions periodically
        if (epoch + 1) % VISUALIZATION_INTERVAL == 0 or epoch == 0:
            visualize_predictions(predictor, test_dataloader, DEVICE, epoch, FINE_TUNED_DIR)
        
        # Save best model
        is_best = valid_mean_iou > best_val_iou
        if is_best:
            best_val_iou = valid_mean_iou
            print(f" New best validation IoU: {best_val_iou:.6f}. Saving best model...")
            
            BEST_MODEL_PATH = os.path.join(FINE_TUNED_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': predictor.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou,
                'model_cfg': MODEL_CFG,
            }, BEST_MODEL_PATH)
            
            print(f" Saved best model to {BEST_MODEL_PATH}")
        
        # Check early stopping
        if early_stopping(valid_mean_iou):
            print(f"Early stopping triggered after {epoch+1} epochs!")
            # Save final model state before stopping
            FINAL_MODEL_PATH = os.path.join(FINE_TUNED_DIR, "final_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': predictor.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_iou': valid_mean_iou,
                'model_cfg': MODEL_CFG,
            }, FINAL_MODEL_PATH)
            print(f" Saved final model to {FINAL_MODEL_PATH}")
            break

        # Save checkpoint at intervals
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            CHECKPOINT_PATH = os.path.join(FINE_TUNED_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': predictor.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_iou': valid_mean_iou,
                'model_cfg': MODEL_CFG,
            }, CHECKPOINT_PATH)
            
            print(f" Saved checkpoint at interval to {CHECKPOINT_PATH}")
    
    print("\n--- Training Complete ---")
    
    # --- Save Training History ---
    training_history_serializable = [
        {'epoch': int(e['epoch']), 'loss': float(e['loss']), 'iou': float(e['mean_iou'])}
        for e in training_history
    ]
    
    validation_history_serializable = [
        {'epoch': int(e['epoch']), 'loss': float(e['loss']), 'iou': float(e['mean_iou'])}
        for e in validation_history
    ]
    
    # Save to JSON
    history_filename_base = os.path.join(FINE_TUNED_DIR, "training_history")
    
    try:
        with open(f'{history_filename_base}_train.json', 'w') as f:
            json.dump(training_history_serializable, f, indent=4)
        with open(f'{history_filename_base}_val.json', 'w') as f:
            json.dump(validation_history_serializable, f, indent=4)
        print(f"Saved training and validation history to {history_filename_base}_*.json")
    except Exception as e:
        print(f"Error saving history JSON files: {e}")
    
    # --- Plotting Results ---
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot([x['epoch'] for x in training_history_serializable], [x['loss'] for x in training_history_serializable], label='Training Loss', marker='o')
        plt.plot([x['epoch'] for x in validation_history_serializable], [x['loss'] for x in validation_history_serializable], label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot([x['epoch'] for x in training_history_serializable], [x['iou'] for x in training_history_serializable], label='Training IoU', marker='o')
        plt.plot([x['epoch'] for x in validation_history_serializable], [x['iou'] for x in validation_history_serializable], label='Validation IoU', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.title('Training and Validation Mean IoU')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(f'Training Metrics for {FINE_TUNED_MODEL_NAME}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_filename = os.path.join(FINE_TUNED_DIR, "training_plots.png")
        plt.savefig(plot_filename)
        print(f"Saved training plots to {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Error generating or saving plots: {e}")
    
    print("Script finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
