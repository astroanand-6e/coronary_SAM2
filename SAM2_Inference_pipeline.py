# SAM2_Inference_Pipeline.py
import os
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os.path # Added for path manipulation

# Import SAM2 modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Helper Functions ---
def read_image(image_path, mask_path=None, target_size=1024):
    """
    Reads and preprocesses image and mask (optional).
    
    Args:
        image_path (str): Path to the image file
        mask_path (str, optional): Path to the ground truth mask
        target_size (int): Target size for resizing
        
    Returns:
        tuple: Original image, resized image, ground truth mask (if provided)
    """
    # Read image
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise FileNotFoundError(f"Error: Could not read image file: {image_path}")
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Resize image
    h, w = img_orig.shape[:2]
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img_orig, (new_w, new_h))
    else:
        img_resized = img_orig.copy()
        
    # Read and preprocess mask if provided
    gt_mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask file: {mask_path}")
        else:
            # Check if mask needs inversion (assuming white foreground on black background)
            white_percentage = np.sum(mask > 127) / (mask.shape[0] * mask.shape[1])
            if white_percentage > 0.5:  # If mask is mostly white, invert it
                mask = 255 - mask
                
            # Resize mask to match image dimensions
            if max(h, w) > target_size:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
            # Convert to binary
            gt_mask = (mask > 127).astype(np.uint8)
    
    return img_orig, img_resized, gt_mask

def get_points(mask, num_points, foreground=True):
    """
    Samples points from mask foreground or background.
    
    Args:
        mask (numpy.ndarray): Binary mask
        num_points (int): Number of points to sample
        foreground (bool): Sample from foreground (True) or background (False)
        
    Returns:
        numpy.ndarray: Sampled points in format expected by SAM2
    """
    if foreground:
        coords = np.argwhere(mask > 0)
    else:
        coords = np.argwhere(mask == 0)

    if len(coords) == 0:
        print(f"Warning: No {'foreground' if foreground else 'background'} pixels found in the mask.")
        return np.zeros((1, 0, 2), dtype=np.float32)

    if len(coords) < num_points:
        print(f"Warning: Only found {len(coords)} {'foreground' if foreground else 'background'} pixels, requested {num_points}.")
        point_indices = np.arange(len(coords))
    else:
        point_indices = np.random.choice(len(coords), num_points, replace=False)

    points_list = []
    for idx in point_indices:
        yx = coords[idx]
        points_list.append([yx[1], yx[0]])  # Convert to [x,y] format for SAM2

    points_np = np.array(points_list, dtype=np.float32)
    if points_np.ndim == 1:
        points_np = points_np.reshape(1, -1)
    points_np = np.expand_dims(points_np, axis=0)  # Add batch dimension [B, N, 2]

    return points_np

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate evaluation metrics between predicted and ground truth masks.
    
    Args:
        pred_mask (numpy.ndarray): Predicted binary mask
        gt_mask (numpy.ndarray): Ground truth binary mask
        
    Returns:
        dict: Dictionary containing IoU, Dice, and other metrics
    """
    # Ensure binary masks
    pred_binary = pred_mask > 0.5
    gt_binary = gt_mask > 0

    # Calculate TP, FP, FN
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Handle edge cases
    eps = 1e-8
    
    # Calculate metrics
    iou = intersection / (union + eps)
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + eps)
    
    return {
        "iou": iou,
        "dice": dice
    }

def load_model(model_cfg, checkpoint_path, device):
    """
    Load SAM2 model and checkpoint.
    
    Args:
        model_cfg (str): Path to model config file
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Device to load model on
        
    Returns:
        SAM2ImagePredictor: Initialized predictor with loaded checkpoint
    """
    print(f"Loading model: {os.path.basename(checkpoint_path)}")
    
    # Initialize the model using config
    sam2_model = build_sam2(model_cfg, checkpoint_path=None, device=device)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            sam2_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            try:
                sam2_model.load_state_dict(checkpoint)
            except:
                print("Warning: Standard loading failed, trying with strict=False")
                sam2_model.load_state_dict(checkpoint, strict=False)
                
        print(f"Successfully loaded model weights from {os.path.basename(checkpoint_path)}")
        
    except FileNotFoundError:
        print(f"Error: Fine-tuned checkpoint file not found at {checkpoint_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise
        
    # Set model to evaluation mode
    sam2_model.eval()
    
    # Create and return predictor
    return SAM2ImagePredictor(sam2_model)

def visualize_results(image, gt_mask, pred_mask, points=None, metrics=None, output_path=None):
    """
    Visualize results and optionally save the visualization.

    Args:
        image (numpy.ndarray): Original image
        gt_mask (numpy.ndarray): Ground truth mask
        pred_mask (numpy.ndarray): Predicted mask
        points (numpy.ndarray, optional): Input points used for prediction
        metrics (dict, optional): Evaluation metrics
        output_path (str, optional): Path to save visualization

    Returns:
        None
    """
    plt.figure(figsize=(18, 6))

    # Image with points
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    if points is not None and points.shape[1] > 0:
        points_to_show = points[0]
        # Ensure points are within image bounds before plotting
        h, w = image.shape[:2]
        valid_points = points_to_show[(points_to_show[:, 0] >= 0) & (points_to_show[:, 0] < w) &
                                      (points_to_show[:, 1] >= 0) & (points_to_show[:, 1] < h)]
        if len(valid_points) > 0:
             plt.scatter(valid_points[:, 0], valid_points[:, 1], c='r', marker='*', s=100)
        else:
             print("Warning: No valid points to display within image bounds.")

    plt.title('Input Image with Points')
    plt.axis('off')

    # Ground truth mask
    plt.subplot(1, 3, 2)
    if gt_mask is not None:
        plt.imshow(gt_mask, cmap='gray')
        plt.title('Ground Truth Mask')
    else:
        plt.imshow(np.zeros_like(image[:, :, 0]), cmap='gray')
        plt.title('Ground Truth Mask (Not Available)')
    plt.axis('off')

    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='gray')
    if metrics:
        plt.title(f'Predicted Mask (IoU: {metrics["iou"]:.4f}, Dice: {metrics["dice"]:.4f})')
    else:
        plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()

    if output_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # First show plot in interactive window
        print("Displaying visualization. Close the plot window to save and continue...")

        # Create a block variable to track when the figure is closed
        block_result = {'blocking': True}

        # Function to handle figure close event
        def on_close(event):
            block_result['blocking'] = False
            plt.close()

        # Connect the close event with our handler
        plt.gcf().canvas.mpl_connect('close_event', on_close)

        # Show the plot and wait until it's closed
        plt.show(block=True)

        # Save the visualization after window is closed
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        # If no output path, just show the interactive plot
        plt.show()


def main():
    """
    Main execution function with hardcoded parameters for easier configuration.
    You can modify these variables directly instead of using command-line arguments.
    """
    # --- CONFIGURATION START ---
    # Set these variables according to your needs

    # Model parameters
    MODEL_CFG = "sam2_hiera_t.yaml"  # Config file for SAM2 architecture (tiny model)

    # Choose one of the available checkpoints:
    # - Pre-trained models: "sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"
    # - Fine-tuned models: "fine_tuned_sam2_hiera_t/best_model.pt", etc.
    CHECKPOINT_PATH = "/home/administrator/Dev/Anand/SAM2_CASBloDaM/fine_tuned_sam2_hiera_b+/best_model.pt"

    # Input options (Single Image Mode Only)
    IMAGE_PATH = "64 Images from doctor /Augmented_image/17.png"  # Path to input image
    MASK_PATH = "64 Images from doctor /Augmented_mask/17.png"    # Path to ground truth mask (optional, set to None if not available)

    # Output settings
    OUTPUT_DIR = "results"  # Base directory for saving results

    # Inference parameters
    NUM_POINTS = 5     # Number of prompt points to sample (0 for automatic mask generation)
    TARGET_SIZE = 1024  # Target size for image resizing
    SEED = 42          # Random seed for reproducibility (set to None to use random seed)
    DEVICE = None      # Device to use (None for auto-selection, or specify: "cuda", "mps", or "cpu")

    # --- CONFIGURATION END ---

    # Set random seed for reproducibility
    if SEED is not None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

    # Device configuration
    if DEVICE is not None:
        device = torch.device(DEVICE)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create base output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    predictor = load_model(MODEL_CFG, CHECKPOINT_PATH, device)

    # Extract model name base for output filename
    model_name_base = os.path.splitext(os.path.basename(MODEL_CFG))[0]

    # Process single image (Removed directory processing logic)
    # Generate dynamic output path
    image_name_base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_filename = f"inference_result_{model_name_base}_{image_name_base}.png"
    dynamic_output_path = os.path.join(OUTPUT_DIR, output_filename)

    process_single_image(
        predictor, IMAGE_PATH, MASK_PATH, dynamic_output_path,
        NUM_POINTS, TARGET_SIZE, device
    )

def process_single_image(predictor, image_path, mask_path, output_path, num_points, target_size, device):
    """Process a single image with the specified model."""
    try:
        # Load image and mask
        img_orig, img_resized, gt_mask = read_image(image_path, mask_path, target_size)

        # Sample points from ground truth mask if available
        input_points = None
        if gt_mask is not None and num_points > 0:
            input_points = get_points(gt_mask, num_points, foreground=True)
            if input_points.shape[1] == 0:
                print(f"Warning: Could not sample points from {os.path.basename(image_path)}")
        elif num_points == 0:
             print("num_points set to 0, proceeding with automatic mask generation.")
        elif gt_mask is None:
             print("Warning: Ground truth mask not provided. Cannot sample points. Proceeding with automatic mask generation.")


        # Run inference
        with torch.no_grad():
            # Set image for prediction
            predictor.set_image(img_resized)

            # Predict with points if available and valid
            if input_points is not None and input_points.shape[1] > 0:
                # Prepare point labels (1 for foreground points)
                input_labels = np.ones((input_points.shape[0], input_points.shape[1]), dtype=np.float32)

                # Get prediction
                masks_pred, scores_pred, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )

                # Handle different dimension formats in the output
                # Fix for IndexError: too many indices for array
                if masks_pred.ndim == 4:
                    # Original format [B, N, H, W]
                    best_mask_idx = np.argmax(scores_pred[0])
                    final_mask = (masks_pred[0, best_mask_idx] > 0.5).astype(np.uint8)
                elif masks_pred.ndim == 3:
                    # Alternative format [N, H, W]
                    best_mask_idx = np.argmax(scores_pred)
                    final_mask = (masks_pred[best_mask_idx] > 0.5).astype(np.uint8)
                else:
                    # Single mask format [H, W]
                    final_mask = (masks_pred > 0.5).astype(np.uint8)

                # Get score info based on dimension
                if scores_pred.ndim > 1:
                    score_info = f"Best Score: {scores_pred[0, best_mask_idx]:.4f}" if scores_pred.ndim == 2 else f"Best Score: {scores_pred[best_mask_idx]:.4f}"
                else:
                    score_info = f"Best Score: {scores_pred[0]:.4f}" if len(scores_pred) > 0 else "No score available"
                print(f"Prediction using {num_points} points. {score_info}")

            else:
                # Automatic mask generation if no points or points couldn't be sampled
                print("Using automatic mask generation...")
                try:
                    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

                    # Create automatic mask generator
                    mask_generator = SAM2AutomaticMaskGenerator(predictor.model)

                    # Generate masks
                    masks = mask_generator.generate(img_resized)

                    if not masks:
                        print("No masks generated automatically")
                        final_mask = np.zeros_like(img_resized[:, :, 0]).astype(np.uint8)
                        score_info = "No masks generated"
                    else:
                        # Use mask with highest score (predicted IoU)
                        best_mask = max(masks, key=lambda x: x['predicted_iou'])
                        final_mask = best_mask['segmentation'].astype(np.uint8)
                        score_info = f"Best Auto Mask IoU: {best_mask['predicted_iou']:.4f}"
                        print(score_info)

                except ImportError:
                     print("Error: Could not import SAM2AutomaticMaskGenerator. Automatic mask generation requires it.")
                     print("Please ensure the SAM2 library is correctly installed.")
                     final_mask = np.zeros_like(img_resized[:, :, 0]).astype(np.uint8) # Default to empty mask
                     score_info = "Automatic generation failed (ImportError)"
                except Exception as auto_gen_e:
                     print(f"Error during automatic mask generation: {auto_gen_e}")
                     final_mask = np.zeros_like(img_resized[:, :, 0]).astype(np.uint8) # Default to empty mask
                     score_info = "Automatic generation failed (Runtime Error)"


        # Calculate metrics if ground truth is available
        metrics = None
        if gt_mask is not None:
            metrics = calculate_metrics(final_mask, gt_mask)
            print(f"Metrics for {os.path.basename(image_path)} - IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")

        # Visualize results
        visualize_results(
            img_resized, gt_mask, final_mask,
            points=input_points, metrics=metrics,
            output_path=output_path
        )

        return metrics

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Main entry point ---
if __name__ == "__main__":
    main()

    # Note: The argparse section has been removed. To use command line arguments instead of hardcoded variables,
    # uncomment the commented section below and comment out the main() call above.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SAM2 Inference Pipeline (Single Image)")

    # Model parameters
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml",
                        help="Path to model config file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint file")
    parser.add_argument("--target_size", type=int, default=1024,
                        help="Target size for image resizing")

    # Input parameters (Single Image Only)
    parser.add_argument("--image_path", type=str, required=True, help="Path to single input image") # Made required
    parser.add_argument("--mask_path", type=str, help="Path to ground truth mask (optional)")

    # Inference parameters
    parser.add_argument("--num_points", type=int, default=5,
                        help="Number of prompt points to sample (0 for automatic mask generation)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save visualization output")
    parser.add_argument("--device", type=str, help="Device to use (e.g., 'cuda:0', 'mps', 'cpu')")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Parse arguments and run
    args = parser.parse_args()

    # Convert args to old main function (Simplified)
    def convert_args_to_main(args):
        # Device configuration
        if args.device:
            device = torch.device(args.device)
        else:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        print(f"Using device: {device}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True) # Use output_dir

        # Load model
        predictor = load_model(args.model_cfg, args.checkpoint_path, device)

        # Extract model name base for output filename
        model_name_base = os.path.splitext(os.path.basename(args.model_cfg))[0]

        # Generate dynamic output path
        image_name_base = os.path.splitext(os.path.basename(args.image_path))[0]
        output_filename = f"inference_result_{model_name_base}_{image_name_base}.png"
        dynamic_output_path = os.path.join(args.output_dir, output_filename) # Use output_dir

        # Process single image
        process_single_image(
            predictor, args.image_path, args.mask_path, dynamic_output_path,
            args.num_points, args.target_size, device
        )

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    convert_args_to_main(args)
    """