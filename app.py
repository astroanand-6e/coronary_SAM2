import os
import gradio as gr
import torch
import numpy as np
import cv2
import time
import functools
from PIL import Image

# --- Configuration ---

# IMPORTANT: Updated paths based on your input
MODEL_PATHS = {
    "SAM2_tiny": "/home/administrator/Dev/Anand/SAM2_CASBloDaM/fine_tuned_sam2_hiera_t/best_model.pt",
    "SAM2_small": "/home/administrator/Dev/Anand/SAM2_CASBloDaM/fine_tuned_sam2_hiera_s/checkpoint_epoch_70.pt",
    "SAM2_base_plus": "/home/administrator/Dev/Anand/SAM2_CASBloDaM/fine_tuned_sam2_hiera_b+/best_model.pt", # Assuming this path is correct now
    "SAM2_large": "/home/administrator/Dev/Anand/SAM2_CASBloDaM/fine_tuned_sam2_hiera_l/best_model.pt",   # Assuming this path is correct now
}

# Check if model files exist
models_available = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models_available[name] = path
    else:
        print(f"Warning: Model checkpoint not found for {name} at {path}. It will not be available in the dropdown.")

if not models_available:
    print("Error: No valid model paths found. Please check MODEL_PATHS.")
    # exit() # Or handle gracefully

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Try importing SAM2 modules
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: SAM2 modules not found. Make sure 'sam2' directory is in your Python path or installed.")
    exit()

# --- Preprocessing Functions ---

def normalize_xray_image(image, kernel_size=(51,51), sigma=0):
    """Normalize X-ray image by applying Gaussian blur and intensity normalization."""
    if image is None: return None
    is_color = len(image.shape) == 3
    if is_color:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.copy()
    gray_image = gray_image.astype(float)
    blurred = gray_image.copy()
    for _ in range(5): # Reduced iterations
        blurred = cv2.GaussianBlur(blurred, kernel_size, sigma)
    mean_intensity = np.mean(blurred)
    factor_image = mean_intensity / (blurred + 1e-10)
    if is_color:
        normalized_image = image.copy().astype(float)
        for i in range(3):
            normalized_image[:,:,i] = normalized_image[:,:,i] * factor_image
    else:
        normalized_image = gray_image * factor_image
    return np.clip(normalized_image, 0, 255).astype(np.uint8)

def apply_clahe(image_uint8):
    """Apply CLAHE for better vessel contrast. Expects uint8 input."""
    if image_uint8 is None: return None
    is_color = len(image_uint8.shape) == 3

    # --- ADJUST CLAHE STRENGTH HERE ---
    clahe_clip_limit = 2.0 
    clahe_tile_grid_size = (8, 8)
    print(f" Applying CLAHE with clipLimit={clahe_clip_limit}, tileGridSize={clahe_tile_grid_size}")
    # ---------------------------------

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)

    if is_color:
        # Apply CLAHE to each channel separately (same as in training)
        enhanced = np.zeros_like(image_uint8, dtype=np.uint8)
        for i in range(image_uint8.shape[2]):
            enhanced[:,:,i] = clahe.apply(image_uint8[:,:,i])
        return enhanced
    else:
        return clahe.apply(image_uint8)

def apply_zscore_normalization(image):
    """Apply Z-score normalization (zero mean and unit variance) as done in training."""
    if image is None: return None
    
    # Convert to float32 for calculations
    image = image.astype(np.float32)
    
    # Apply z-score normalization to each channel if color image
    if len(image.shape) == 3:
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            channel = image[:,:,i]
            mean = np.mean(channel)
            std = np.std(channel) + 1e-6  # Add small value to prevent division by zero
            normalized[:,:,i] = (channel - mean) / std
    else:
        mean = np.mean(image)
        std = np.std(image) + 1e-6
        normalized = (image - mean) / std
    
    return normalized

def resize_with_aspect_ratio(image, target_size=1024):
    """Resize image to target size while preserving aspect ratio."""
    if image is None: return None
    
    height, width = image.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    r = min(target_size / width, target_size / height)
    new_width, new_height = int(width * r), int(height * r)
    
    # Resize the image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def preprocess_image_for_sam2(image_rgb_numpy):
    """Combined preprocessing: normalization + CLAHE + Z-score normalization + resizing."""
    if image_rgb_numpy is None:
        print("Preprocessing: Input image is None.")
        return None

    start_time = time.time()
    print("Preprocessing Step 1: Normalizing X-ray image...")
    if image_rgb_numpy.dtype != np.uint8:
         image_rgb_numpy = np.clip(image_rgb_numpy, 0, 255).astype(np.uint8)
    if len(image_rgb_numpy.shape) == 2:
        image_rgb_numpy = cv2.cvtColor(image_rgb_numpy, cv2.COLOR_GRAY2RGB)

    normalized_uint8 = normalize_xray_image(image_rgb_numpy)
    if normalized_uint8 is None:
        print("Preprocessing failed at normalization step.")
        return None
    print(f"Normalization done in {time.time() - start_time:.2f}s")

    start_time_clahe = time.time()
    print("Preprocessing Step 2: Applying CLAHE...")
    clahe_image = apply_clahe(normalized_uint8) # CLAHE applied here
    if clahe_image is None:
        print("Preprocessing failed at CLAHE step.")
        return None
    print(f"CLAHE done in {time.time() - start_time_clahe:.2f}s")
    
    # Step 3: Apply Z-score normalization (as done in training)
    start_time_zscore = time.time()
    print("Preprocessing Step 3: Applying Z-score normalization...")
    zscore_normalized = apply_zscore_normalization(clahe_image)
    if zscore_normalized is None:
        print("Preprocessing failed at Z-score normalization step.")
        return None
    print(f"Z-score normalization done in {time.time() - start_time_zscore:.2f}s")
    
    # Step 4: Resize maintaining aspect ratio if needed
    start_time_resize = time.time()
    print("Preprocessing Step 4: Resizing image...")
    if zscore_normalized.shape[0] != 1024 or zscore_normalized.shape[1] != 1024:
        resized_image = resize_with_aspect_ratio(zscore_normalized, target_size=1024)
        print(f"Image resized from {zscore_normalized.shape[:2]} to {resized_image.shape[:2]}")
    else:
        resized_image = zscore_normalized
        print("Image already at target size 1024x1024")
    print(f"Resizing done in {time.time() - start_time_resize:.2f}s")
    
    print(f"Total preprocessing time: {time.time() - start_time:.2f}s")
    
    return resized_image # Return the fully preprocessed image

# --- Model Loading ---

@functools.lru_cache(maxsize=len(models_available)) # Cache based on available models
def load_model(model_name):
    """Loads the specified SAM2 model and creates a predictor."""
    print(f"\nAttempting to load model: {model_name}")
    if model_name not in models_available: # Check against available models
        print(f"Error: Model name '{model_name}' not found or checkpoint missing.")
        return None

    checkpoint_path = models_available[model_name] # Get path from available dict

    try:
        print(f" Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if 'model_cfg' not in checkpoint:
            print(f"Error: 'model_cfg' key not found in checkpoint {checkpoint_path}.")
            return None
        model_cfg_name = checkpoint['model_cfg']
        print(f" Using model config from checkpoint: {model_cfg_name}")
        sam2_model = build_sam2(model_cfg_name, checkpoint_path=None, device=DEVICE)
        if 'model_state_dict' not in checkpoint:
             print(f"Error: 'model_state_dict' not found in checkpoint {checkpoint_path}.")
             return None
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        sam2_model.load_state_dict(new_state_dict)
        print(" Successfully loaded fine-tuned model state_dict.")
        sam2_model.to(DEVICE)
        sam2_model.eval()
        predictor = SAM2ImagePredictor(sam2_model)
        print(f"Model '{model_name}' loaded successfully on {DEVICE}.")
        return predictor
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Gradio UI and Logic ---

def resize_image_fixed(image, target_size=1024):
    """Resizes image to a fixed square size (1024x1024)."""
    if image is None: return None
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def draw_points_on_image(image, points_state):
    """Draws points (green positive, red negative) on a copy of the image."""
    if image is None or not points_state: return image
    draw_image = image.copy()
    radius = max(3, int(min(image.shape[:2]) * 0.005))
    thickness = -1
    for x, y, label in points_state:
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(draw_image, (int(x), int(y)), radius, color, thickness)
        cv2.circle(draw_image, (int(x), int(y)), radius, (0,0,0), 1)
    return draw_image

def add_point(original_image, points_state, point_type, evt: gr.SelectData):
    """Callback function when user clicks on the input image."""
    if original_image is None:
        gr.Warning("Please upload an image first.")
        return original_image, points_state
    x, y = evt.index[0], evt.index[1]
    label = 1 if point_type == "Positive" else 0
    points_state.append([x, y, label])
    print(f"Added point: ({x}, {y}), Type: {'Positive' if label==1 else 'Negative'}, Total Points: {len(points_state)}")
    image_with_points = draw_points_on_image(original_image, points_state)
    return image_with_points, points_state

def clear_points(original_image):
    """Clears all points and resets the display image."""
    print("Clearing all points.")
    return original_image, []

def undo_last_point(original_image, points_state):
    """Removes the last added point and updates the display image."""
    if not points_state:
        print("No points to undo.")
        return original_image, points_state # Return original image if no points

    removed_point = points_state.pop()
    print(f"Removed point: {removed_point}, Remaining Points: {len(points_state)}")
    image_with_points = draw_points_on_image(original_image, points_state)
    return image_with_points, points_state


def run_segmentation(original_image, model_name, points_state):
    """Runs the SAM2 model segmentation and returns mask and preprocessed image."""
    start_total_time = time.time()
    # Initialize return values in case of early exit
    output_mask_image = None
    preprocessed_display_image = None

    if original_image is None:
        gr.Warning("Please upload an image first.")
        return output_mask_image, preprocessed_display_image, points_state

    print(f"\n--- Running Segmentation ---")
    print(f" Model Selected: {model_name}")
    print(f" Number of points: {len(points_state)}")

    # --- 1. Load Model ---
    predictor = load_model(model_name)
    if predictor is None:
        gr.Error(f"Failed to load model '{model_name}'. Check logs and paths.")
        return output_mask_image, preprocessed_display_image, points_state

    # --- 2. Prepare Image ---
    original_h, original_w = original_image.shape[:2]
    print(f" Original image size: {original_w}x{original_h}")
    image_resized_1024 = resize_image_fixed(original_image, 1024)
    print(f" Resized image to: 1024x1024")

    # Apply preprocessing
    image_preprocessed_uint8 = preprocess_image_for_sam2(image_resized_1024)
    if image_preprocessed_uint8 is None:
        gr.Error("Image preprocessing failed.")
        return output_mask_image, preprocessed_display_image, points_state

    # Prepare the preprocessed image for display (it's already uint8)
    # Ensure it's RGB for consistent display in Gradio
    if len(image_preprocessed_uint8.shape) == 2:
        preprocessed_display_image = cv2.cvtColor(image_preprocessed_uint8, cv2.COLOR_GRAY2RGB)
    else:
        preprocessed_display_image = image_preprocessed_uint8.copy()


    print(" Setting preprocessed image in predictor...")
    start_set_image = time.time()
    predictor.set_image(image_preprocessed_uint8) # Feed the preprocessed image to SAM
    print(f" predictor.set_image took {time.time() - start_set_image:.2f}s")

    # --- 3. Prepare Prompts ---
    scale_w = 1024.0 / original_w
    scale_h = 1024.0 / original_h
    if not points_state:
        center_x, center_y = 512, 512
        point_coords = np.array([[[center_x, center_y]]])
        point_labels = np.array([1])
        print(" No points provided. Using center point.")
    else:
        scaled_points = [[x * scale_w, y * scale_h] for x, y, label in points_state]
        labels = [label for x, y, label in points_state]
        point_coords = np.array([scaled_points])
        point_labels = np.array(labels)
        print(f" Using {len(points_state)} provided points.")

    point_coords_torch = torch.tensor(point_coords, dtype=torch.float32).to(DEVICE)
    point_labels_torch = torch.tensor(point_labels, dtype=torch.float32).unsqueeze(0).to(DEVICE) # Add batch dim

    # --- 4. Run Model Inference ---
    print(" Running model inference...")
    start_inference_time = time.time()
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(point_coords_torch, point_labels_torch), boxes=None, masks=None
        )
        if predictor._features is None:
             gr.Error("Image features not computed.")
             return output_mask_image, preprocessed_display_image, points_state
        image_embed = predictor._features["image_embed"][-1].unsqueeze(0)
        image_pe = predictor.model.sam_prompt_encoder.get_dense_pe()
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=image_embed, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
            multimask_output=True, repeat_image=False, high_res_features=high_res_features,
        )
        prd_masks_1024 = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
        best_mask_idx = torch.argmax(prd_scores[0]).item()
        best_mask_1024 = torch.sigmoid(prd_masks_1024[:, best_mask_idx])
        binary_mask_1024 = (best_mask_1024 > 0.5).cpu().numpy().squeeze()
    print(f" Model inference took {time.time() - start_inference_time:.2f}s")

    # --- 5. Resize Mask to Original ---
    print(" Resizing mask to original dimensions...")
    final_mask_resized = cv2.resize(
        binary_mask_1024.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )

    # --- 6. Format Mask for Display ---
    output_mask_image = (final_mask_resized * 255).astype(np.uint8)
    if len(output_mask_image.shape) == 2: # Ensure RGB for display consistency
        output_mask_image = cv2.cvtColor(output_mask_image, cv2.COLOR_GRAY2RGB)

    total_time = time.time() - start_total_time
    print(f"--- Segmentation Complete (Total time: {total_time:.2f}s) ---")

    # Return final mask, preprocessed image for display, and points state
    return output_mask_image, preprocessed_display_image, points_state


# --- Build Gradio Interface ---
css = """
    #output_mask_container .gradio-image, #preprocessed_image_container .gradio-image { height: 500px !important; object-fit: contain; }
    #input_image_container .gradio-image { height: 500px !important; object-fit: contain;}
    .output-image-col img { max-height: 500px; object-fit: contain; }
"""

with gr.Blocks(css=css, title="Coronary Artery Segmentation (Fine-tuned SAM2)") as demo:
    gr.Markdown("# Coronary Artery Segmentation using Fine-tuned SAM2")
    gr.Markdown(
        "Upload an X-ray image. Select a model. "
        "Click 'Run Segmentation' for automatic prediction or click on the image to add points first."
    )

    points_state = gr.State([])
    original_image_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Input Image & Controls")
            input_image = gr.Image(
                type="numpy", label="Upload Coronary X-ray Image",
                interactive=True, height=500, elem_id="input_image_container" # Increased height here too
            )
            # Add example images if available
            # gr.Examples(
            #     examples=["path/to/example1.jpg", "path/to/example2.png"],
            #     inputs=input_image,
            #     label="Example Images"
            # )


            model_selector = gr.Dropdown(
                choices=list(models_available.keys()), # Use only available models
                label="Select SAM2 Model Variant",
                value=list(models_available.keys())[-1] if models_available else None
            )
            prompt_type = gr.Radio(
                ["Positive", "Negative"], label="Point Prompt Type", value="Positive"
            )
            with gr.Row():
                clear_button = gr.Button("Clear Points")
                undo_button = gr.Button("Undo Last Point") # Added Undo Button
                run_button = gr.Button("Run Segmentation", variant="primary")

        with gr.Column(scale=1, elem_classes="output-image-col"):
            gr.Markdown("## 2. Outputs")
            # Renamed output_mask -> final_mask_display for clarity
            final_mask_display = gr.Image(
                type="numpy", label="Predicted Binary Mask (White = Artery)",
                interactive=False, height=500, elem_id="output_mask_container" # Increased height
            )

        with gr.Column(scale=1, elem_classes="output-image-col"):
             gr.Markdown("## 3. Preprocessed Input")
             # New component to show the preprocessed image
             preprocessed_image_display = gr.Image(
                 type="numpy", label="Preprocessed Image (for SAM Input)",
                 interactive=False, height=500, elem_id="preprocessed_image_container" # Increased height
             )


    # --- Define Interactions ---
    def store_uploaded_image(img):
        print("Image uploaded.")
        # Also clear previous results on new upload
        return img, img, [], None, None # input, original_state, points, clear mask, clear preproc
    input_image.upload(
        store_uploaded_image,
        inputs=[input_image],
        outputs=[input_image, original_image_state, points_state, final_mask_display, preprocessed_image_display]
    )
    def clear_all():
        # Clear display images and state
        return None, None, [], None, None
    input_image.clear(
        clear_all,
        [],
        [input_image, original_image_state, points_state, final_mask_display, preprocessed_image_display]
     )

    input_image.select(
        add_point,
        inputs=[input_image, points_state, prompt_type],
        outputs=[input_image, points_state]
    )

    def clear_points_and_outputs(original_img):
         # Clear points, mask, and preprocessed display, reset input display
         return original_img, [], None, None
    clear_button.click(
        clear_points_and_outputs,
        inputs=[original_image_state],
        outputs=[input_image, points_state, final_mask_display, preprocessed_image_display]
    )

    # Connect the Undo button
    undo_button.click(
        undo_last_point,
        inputs=[original_image_state, points_state],
        outputs=[input_image, points_state]
    )

    run_button.click(
        run_segmentation,
        inputs=[original_image_state, model_selector, points_state],
        # Update outputs to include the new preprocessed image display
        outputs=[final_mask_display, preprocessed_image_display, points_state]
    )


# --- Launch the App ---
if __name__ == "__main__":
    # Optional pre-loading
    # default_model = list(models_available.keys())[-1] if models_available else None
    # if default_model: load_model(default_model)

    print("Launching Gradio App...")
    # Share=True allows access over local network, remove if not needed
    demo.launch(debug=True, share=False)