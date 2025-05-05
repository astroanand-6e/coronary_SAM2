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
# ...existing code...
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
    # Lower clipLimit reduces the contrast enhancement effect.
    # Original was 2.0. Try values like 1.5, 1.0, or even disable by setting it very low.
    clahe_clip_limit = 2.0
    clahe_tile_grid_size = (8, 8)
    print(f" Applying CLAHE with clipLimit={clahe_clip_limit}, tileGridSize={clahe_tile_grid_size}")
    # ---------------------------------

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)

    if is_color:
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        clahe_image_uint8 = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    else:
         clahe_image_uint8 = clahe.apply(image_uint8)

    # Return uint8 [0, 255] suitable for predictor.set_image
    return clahe_image_uint8


def preprocess_image_for_sam2(image_rgb_numpy):
    """Combined preprocessing: normalization + CLAHE for SAM2 input."""
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
    preprocessed_uint8 = apply_clahe(normalized_uint8) # CLAHE applied here
    if preprocessed_uint8 is None:
        print("Preprocessing failed at CLAHE step.")
        return None
    print(f"CLAHE done in {time.time() - start_time_clahe:.2f}s")
    print(f"Total preprocessing time: {time.time() - start_time:.2f}s")

    return preprocessed_uint8 # Return the image after all steps

# --- Model Loading ---
# ...existing code...
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

# --- Utility Functions ---
# ...existing code...
def resize_image_fixed(image, target_size=1024):
    """Resizes image to a fixed square size (1024x1024)."""
    if image is None: return None
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def draw_points_on_image(image, points_state):
    """Draws points (green positive, red negative) on a copy of the image."""
    if image is None: return image # Return original if no image
    draw_image = image.copy()
    if not points_state: return draw_image # Return copy if no points

    # Make points slightly larger and add a black border
    base_radius = max(4, int(min(image.shape[:2]) * 0.006)) # Slightly larger base radius
    border_thickness = 1 # Thickness of the black border
    radius_with_border = base_radius + border_thickness
    thickness = -1 # Filled circle

    for x, y, label in points_state:
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        center = (int(x), int(y))
        # Draw black border circle first
        cv2.circle(draw_image, center, radius_with_border, (0, 0, 0), thickness)
        # Draw colored circle on top
        cv2.circle(draw_image, center, base_radius, color, thickness)

    return draw_image

# --- Gradio UI Interaction Functions ---

def get_point_counts_text(points_state):
    """Helper function to generate the point count markdown string."""
    pos_count = sum(1 for _, _, label in points_state if label == 1)
    neg_count = sum(1 for _, _, label in points_state if label == 0)
    return f"**Points Added:** <font color='green'>{pos_count} Positive</font>, <font color='red'>{neg_count} Negative</font>"

def add_point(preprocessed_image, points_state, point_type, evt: gr.SelectData):
    """Callback function when user clicks on the preprocessed image."""
    if preprocessed_image is None:
        gr.Warning("Please upload and preprocess an image first.")
        # Return original image, points state, and existing counts text
        return preprocessed_image, points_state, get_point_counts_text(points_state)
    x, y = evt.index[0], evt.index[1]
    label = 1 if point_type == "Positive" else 0
    # Store coordinates relative to the preprocessed image (1024x1024)
    points_state.append([x, y, label])
    print(f"Added point: ({x}, {y}), Type: {'Positive' if label==1 else 'Negative'}, Total Points: {len(points_state)}")
    image_with_points = draw_points_on_image(preprocessed_image, points_state)
    # Return updated image, points state, and updated counts text
    return image_with_points, points_state, get_point_counts_text(points_state)

def undo_last_point(preprocessed_image, points_state):
    """Removes the last added point and updates the preprocessed display image."""
    if preprocessed_image is None: # Handle case where image is cleared
         # Return None image, points state, and counts text
         return None, points_state, get_point_counts_text(points_state)
    if not points_state:
        print("No points to undo.")
        # Return the current preprocessed image without changes if no points
        return preprocessed_image, points_state, get_point_counts_text(points_state)

    removed_point = points_state.pop()
    print(f"Removed point: {removed_point}, Remaining Points: {len(points_state)}")
    image_with_points = draw_points_on_image(preprocessed_image, points_state)
    # Return updated image, points state, and updated counts text
    return image_with_points, points_state, get_point_counts_text(points_state)

def clear_points_and_display(preprocessed_image_state):
     """Clears points and resets the preprocessed display image."""
     print("Clearing points and resetting preprocessed display.")
     points_state = [] # Clear points
     # Return the stored preprocessed image without points, clear points state, clear mask, clear counts text
     return preprocessed_image_state, points_state, None, get_point_counts_text(points_state)

def run_segmentation(preprocessed_image_state, original_image_state, model_name, points_state):
    """Runs SAM2 segmentation using points on the preprocessed image."""
    start_total_time = time.time()
    # Initialize return values
    output_mask_display = None

    if preprocessed_image_state is None or original_image_state is None:
        gr.Warning("Please upload an image first.")
        return output_mask_display, points_state

    print(f"\n--- Running Segmentation ---")
    print(f" Model Selected: {model_name}")
    print(f" Number of points: {len(points_state)}")

    # --- 1. Load Model ---
    predictor = load_model(model_name)
    if predictor is None:
        gr.Error(f"Failed to load model '{model_name}'. Check logs and paths.")
        return output_mask_display, points_state

    # --- 2. Use Preprocessed Image ---
    # The image is already preprocessed and resized to 1024x1024
    image_for_predictor = preprocessed_image_state
    original_h, original_w = original_image_state.shape[:2] # Get original dims for final resize
    print(f" Using preprocessed image (1024x1024) for predictor.")
    print(f" Original image size for final mask resize: {original_w}x{original_h}")

    print(" Setting preprocessed image in predictor...")
    start_set_image = time.time()
    # Feed the preprocessed image (which is already 1024x1024 uint8) to SAM
    predictor.set_image(image_for_predictor)
    print(f" predictor.set_image took {time.time() - start_set_image:.2f}s")

    # --- 3. Prepare Prompts (No Scaling Needed) ---
    if not points_state:
        # Use center point if no points provided
        center_x, center_y = 512, 512
        point_coords = np.array([[[center_x, center_y]]])
        point_labels = np.array([1])
        print(" No points provided. Using center point (512, 512).")
    else:
        # Points are already relative to the 1024x1024 preprocessed image
        point_coords_list = [[x, y] for x, y, label in points_state]
        labels_list = [label for x, y, label in points_state]
        point_coords = np.array([point_coords_list])
        point_labels = np.array(labels_list)
        print(f" Using {len(points_state)} provided points (coords relative to 1024x1024).")

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
             gr.Error("Image features not computed. Predictor might not have been set correctly.")
             return output_mask_display, points_state
        # Ensure features are accessed correctly
        image_embed = predictor._features["image_embed"][-1].unsqueeze(0)
        image_pe = predictor.model.sam_prompt_encoder.get_dense_pe()
        # Handle potential missing high_res_features key gracefully
        high_res_features = None
        if "high_res_feats" in predictor._features and predictor._features["high_res_feats"]:
             try:
                 high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
             except IndexError:
                 print("Warning: Index error accessing high_res_feats. Proceeding without them.")
             except Exception as e:
                 print(f"Warning: Error processing high_res_features: {e}. Proceeding without them.")

        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=image_embed, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
            multimask_output=True, repeat_image=False, # repeat_image should be False for single image prediction
            high_res_features=high_res_features, # Pass None if not available
        )
        # Postprocess masks to 1024x1024
        prd_masks_1024 = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1]) # predictor._orig_hw should be (1024, 1024)
        # Select the best mask based on predicted score
        best_mask_idx = torch.argmax(prd_scores[0]).item()
        # Apply sigmoid and thresholding
        best_mask_1024_prob = torch.sigmoid(prd_masks_1024[:, best_mask_idx])
        binary_mask_1024 = (best_mask_1024_prob > 0.5).cpu().numpy().squeeze() # Squeeze to get (H, W)
    print(f" Model inference took {time.time() - start_inference_time:.2f}s")

    # --- 5. Resize Mask to Original Dimensions ---
    print(" Resizing mask to original dimensions...")
    final_mask_resized = cv2.resize(
        binary_mask_1024.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )

    # --- 6. Format Mask for Display ---
    # Mask for display (RGB)
    output_mask_display = (final_mask_resized * 255).astype(np.uint8)
    if len(output_mask_display.shape) == 2: # Ensure RGB for display consistency
        output_mask_display = cv2.cvtColor(output_mask_display, cv2.COLOR_GRAY2RGB)

    total_time = time.time() - start_total_time
    print(f"--- Segmentation Complete (Total time: {total_time:.2f}s) ---")

    # Return: mask for display, points state (unchanged)
    return output_mask_display, points_state # No change needed here as it doesn't modify points


def process_upload(uploaded_image):
    """Handles image upload: preprocesses, resizes, stores states."""
    if uploaded_image is None:
        # Clear everything including point counts
        return None, None, None, [], None, get_point_counts_text([])

    print("Image uploaded. Processing...")
    # 1. Store original image
    original_image = uploaded_image.copy()

    # 2. Resize to 1024x1024 for preprocessing
    image_resized_1024 = resize_image_fixed(original_image, 1024)
    if image_resized_1024 is None:
        gr.Error("Failed to resize image.")
        return None, None, None, [], None

    # 3. Preprocess the 1024x1024 image
    preprocessed_1024 = preprocess_image_for_sam2(image_resized_1024)
    if preprocessed_1024 is None:
        gr.Error("Image preprocessing failed.")
        return None, None, None, [], None

    # Ensure preprocessed image is RGB for display
    if len(preprocessed_1024.shape) == 2:
        preprocessed_1024_display = cv2.cvtColor(preprocessed_1024, cv2.COLOR_GRAY2RGB)
    else:
        preprocessed_1024_display = preprocessed_1024.copy()

    print("Image processed successfully.")
    points_state = [] # Clear points on new upload
    # Return:
    # 1. Preprocessed image for display (interactive)
    # 2. Preprocessed image for state
    # 3. Original image for state
    # 4. Cleared points state
    # 5. Cleared mask display
    # 6. Cleared point counts text
    return preprocessed_1024_display, preprocessed_1024, original_image, points_state, None, get_point_counts_text(points_state)


def clear_all_outputs():
    """Clears all input/output fields and states."""
    print("Clearing all inputs and outputs.")
    points_state = [] # Clear points
    # Clear everything including point counts
    return None, None, None, points_state, None, get_point_counts_text(points_state)


# --- Build Gradio Interface ---
css = """
    #mask_display_container .gradio-image { height: 450px !important; object-fit: contain; }
    #preprocessed_image_container .gradio-image { height: 450px !important; object-fit: contain; cursor: crosshair !important; }
    #upload_container .gradio-image { height: 150px !important; object-fit: contain; } /* Smaller upload preview */
    .output-col img { max-height: 450px; object-fit: contain; }
    .control-col { min-width: 500px; } /* Wider control column */
    .output-col { min-width: 500px; }
"""

with gr.Blocks(css=css, title="Coronary Artery Segmentation (Fine-tuned SAM2)") as demo:
    gr.Markdown("# Coronary Artery Segmentation using Fine-tuned SAM2")
    gr.Markdown(
        "**Let's find those arteries!**\n\n"
        "1. Upload your Coronary X-ray Image.\n"
        "2. The preprocessed image appears on the left. Time to guide the AI! Click directly on the image to add **Positive** (artery) or **Negative** (background) points.\n"
        "3. Choose your fine-tuned SAM2 model.\n"
        "4. Hit 'Run Segmentation' and watch the magic happen!\n"
        "5. Download your predicted mask (the white area) using the download button on the mask image."
    )

    # --- States ---
    points_state = gr.State([])
    # State to store the original uploaded image (needed for final mask resizing)
    original_image_state = gr.State(None)
    # State to store the preprocessed 1024x1024 image data (used for drawing points and predictor input)
    preprocessed_image_state = gr.State(None)


    with gr.Row():
        # --- Left Column (Controls & Preprocessed Image Interaction) ---
        with gr.Column(scale=1, elem_classes="control-col"):
            gr.Markdown("## 1. Upload & Controls")
            # Keep upload separate and smaller
            upload_image = gr.Image(
                type="numpy", label="Upload Coronary X-ray Image",
                height=150, elem_id="upload_container"
            )
            gr.Markdown("## 2. Add Points on Preprocessed Image")
            # Interactive Preprocessed Image Display
            preprocessed_image_display = gr.Image(
                type="numpy", label="Click on Image to Add Points",
                interactive=True, # Make this interactive
                height=450, elem_id="preprocessed_image_container"
            )
            # Add Point Counter Display
            point_counter_display = gr.Markdown(get_point_counts_text([]))

            model_selector = gr.Dropdown(
                choices=list(models_available.keys()),
                label="Select SAM2 Model Variant",
                value=list(models_available.keys())[-1] if models_available else None
            )
            prompt_type = gr.Radio(
                ["Positive", "Negative"], label="Point Prompt Type", value="Positive"
            )
            with gr.Row():
                clear_button = gr.Button("Clear Points")
                undo_button = gr.Button("Undo Last Point")
            run_button = gr.Button("Run Segmentation", variant="primary")
            clear_all_button = gr.Button("Clear All") # Added Clear All

        # --- Right Column (Output Mask) ---
        with gr.Column(scale=1, elem_classes="output-col"):
            gr.Markdown("## 3. Predicted Mask")
            final_mask_display = gr.Image(
                type="numpy", label="Predicted Binary Mask (White = Artery)",
                interactive=False, height=450, elem_id="mask_display_container",
                format="png" # Specify PNG format for download
            )


    # --- Define Interactions ---

    # 1. Upload triggers preprocessing and display
    upload_image.upload(
        fn=process_upload,
        inputs=[upload_image],
        outputs=[
            preprocessed_image_display, # Update interactive display
            preprocessed_image_state,   # Update state
            original_image_state,       # Update state
            points_state,               # Clear points
            final_mask_display,         # Clear mask display
            point_counter_display       # Clear point counts
        ]
    )

    # 2. Clicking on preprocessed image adds points
    preprocessed_image_display.select(
        fn=add_point,
        inputs=[preprocessed_image_state, points_state, prompt_type],
        outputs=[
            preprocessed_image_display, # Update display with points
            points_state,               # Update points state
            point_counter_display       # Update point counts
            ]
    )

    # 3. Clear points button resets points and preprocessed display
    clear_button.click(
        fn=clear_points_and_display,
        inputs=[preprocessed_image_state], # Needs the clean preprocessed image
        outputs=[
            preprocessed_image_display, # Reset display
            points_state,               # Clear points
            final_mask_display,         # Clear mask
            point_counter_display       # Reset point counts
            ]
    )

    # 4. Undo button removes last point and updates preprocessed display
    undo_button.click(
        fn=undo_last_point,
        inputs=[preprocessed_image_state, points_state], # Needs current preprocessed image
        outputs=[
            preprocessed_image_display, # Update display
            points_state,               # Update points state
            point_counter_display       # Update point counts
            ]
    )

    # 5. Run segmentation (Outputs don't change point counts)
    run_button.click(
        fn=run_segmentation,
        inputs=[
            preprocessed_image_state, # Use preprocessed image data
            original_image_state,     # Needed for final resize dim
            model_selector,
            points_state              # Points are relative to preprocessed
            ],
        outputs=[
            final_mask_display,       # Show the final mask
            points_state              # Pass points state (might be needed if run modifies it - currently doesn't)
            ]
    )

    # 7. Clear All button
    clear_all_button.click(
        fn=clear_all_outputs,
        inputs=[],
        outputs=[
            upload_image,
            preprocessed_image_display,
            preprocessed_image_state,
            points_state,
            final_mask_display,
            point_counter_display # Reset point counts
        ]
    )


# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(debug=True, share=False)