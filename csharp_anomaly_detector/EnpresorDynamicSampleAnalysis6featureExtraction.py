import tkinter as tk
from tkinter import filedialog, Label, Frame, Button as TkButton, Checkbutton, IntVar, messagebox
import tifffile
import numpy as np
from PIL import Image, ImageTk
from skimage.color import rgb2lab
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.widgets import RadioButtons, Button as MplButton
from matplotlib.colors import to_rgba
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
import umap.umap_ as umap
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import silhouette_score
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import messagebox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2

# -------------------------------------------------------------------
# GLOBAL VARIABLES
# -------------------------------------------------------------------
samples = []            # Dynamically added samples
border_sizes = {}       # Border sizes for each sample
file_paths = {}         # File paths for each sample
selected_channels = {}  # Selected channels for PCA
channels = ["R", "G", "B", "870", "1200", "1550", "L", "A", "B*"]
sample_extracted_data = {}
cached_distances = {}  # Store Mahalanobis distances for each sample
cached_reference_model = None  # Store the reference model
sensitivity_value_labels = {}
sensitivity_control_frames = {}


anomaly_sensitivities = {
    "Sample 1": 2.5,  # Default value
    "Sample 2": 2.5   # Default value
}

# Global variables for sharing computed parameters.
current_sample = None
adjust_params = {
    "rx": None, "ry": None, "rz": None,   # Ellipsoid semi-axis lengths
    "tx": None, "ty": None, "tz": None,     # Ellipsoid center (translation)
    "rot_x": None, "rot_y": None, "rot_z": None,  # Rotation angles (degrees)
    "sensitivity" : 50 # New: default = 50
}
cached_ellipsoid_params = {}   # Cached (initial) ellipse parameters per sample
filtered_points_by_sample = {} # Filtered PCA points for each sample

# We'll store both the raw preview images (as NumPy arrays) and the Tkinter Label objects.
preview_images = {}   # sample_name -> NumPy array (resized image)
preview_labels = {}   # sample_name -> Tkinter Label widget

root = tk.Tk() 
apply_normalization_var = tk.IntVar()  # 0 = off, 1 = on
sample_extracted_data = {}

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def norm_im(im):
    """Normalize image to uint8 if needed."""
    if im.dtype != np.uint8:
        im = (im / im.max() * 255).astype(np.uint8)
    return im

def toggle_channel(sample_name, ch):
    if sample_name not in selected_channels:
        selected_channels[sample_name] = []
    if ch in selected_channels[sample_name]:
        selected_channels[sample_name].remove(ch)
    else:
        selected_channels[sample_name].append(ch)

def increase_sample_sensitivity(sample_name):
    global anomaly_sensitivities
    if sample_name in anomaly_sensitivities and sample_name != "Sample 1":
        # Use a smaller increment for more granular control (0.05 instead of 0.1)
        anomaly_sensitivities[sample_name] += 0.05
        update_sensitivity_labels(sample_name)
        # Update the affected sample
        update_sample_masks_fast(sample_name)
        # Also update Sample 1 highlights based on new sensitivity
        update_sample1_highlights()

def decrease_sample_sensitivity(sample_name):
    global anomaly_sensitivities
    print(f"Starting decrease_sample_sensitivity for {sample_name}")
    
    if sample_name in anomaly_sensitivities and sample_name != "Sample 1":
        old_value = anomaly_sensitivities[sample_name]
        # Use a smaller decrement for more granular control (0.05 instead of 0.1)
        # Don't go below 0.5
        anomaly_sensitivities[sample_name] = max(0.5, old_value - 0.05)
        new_value = anomaly_sensitivities[sample_name]
        
        print(f"  Changed sensitivity from {old_value:.2f} to {new_value:.2f}")
        
        print("  Updating sensitivity label...")
        update_sensitivity_labels(sample_name)
        
        print("  Updating sample masks...")
        update_sample_masks_fast(sample_name)
        
        print("  Updating Sample 1 highlights...")
        update_sample1_highlights()
        
        print("Finished decrease_sample_sensitivity")
    else:
        print(f"  Sample not eligible for sensitivity adjustment: {sample_name}")

# Function to update all sensitivity labels
def update_sensitivity_labels(sample_name=None):
    """
    Update sensitivity labels with current values.
    If sample_name is provided, only update that sample's label.
    Otherwise, update all labels.
    Uses two decimal places for more precision.
    """
    global sensitivity_value_labels, anomaly_sensitivities
    
    # If a specific sample was provided, only update that one
    if sample_name is not None:
        if sample_name in sensitivity_value_labels and sample_name in anomaly_sensitivities:
            try:
                label_widget = sensitivity_value_labels[sample_name]
                # Show 2 decimal places instead of 1
                label_widget.config(text=f"{anomaly_sensitivities[sample_name]:.2f}")
            except Exception as e:
                print(f"Error updating label for {sample_name}: {str(e)}")
        return
    
    # Otherwise update all labels
    for sname, label_widget in sensitivity_value_labels.items():
        if sname in anomaly_sensitivities:
            try:
                # Show 2 decimal places instead of 1
                label_widget.config(text=f"{anomaly_sensitivities[sname]:.2f}")
            except Exception as e:
                print(f"Error updating label for {sname}: {str(e)}")

def apply_erosion(mask, border_size):
    valid = ~mask
    for _ in range(border_size):
        valid = binary_erosion(valid)
    return valid

from PIL import Image, ImageTk
import numpy as np
import tifffile

def display_channels_with_modifications(label_container, filepaths, sample_name):
    """
    Loads and processes multiple images for a sample.
    Supports both TIFF (multi-channel) and PNG (RGB).
    Applies border processing and background modifications to each image.
    Displays all images for a sample in the GUI.
    """
    try:
        images_rgb = []  # Store processed raw images
        images_mod = []  # Store modified images

        for filepath in filepaths:  # Loop through multiple images
            if filepath.lower().endswith((".tif", ".tiff")):
                # Load TIFF file
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()

                # Ensure TIFF file has expected shape (4D with RGB channels or 3D with RGB)
                if len(arr.shape) == 4 and arr.shape[-1] == 3:
                    rgb = norm_im(arr[0])  # First frame as RGB
                elif len(arr.shape) == 3 and arr.shape[-1] == 3:
                    rgb = norm_im(arr)  # Directly use as RGB
                else:
                    raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

            elif filepath.lower().endswith(".png"):
                # Load PNG file correctly
                img = Image.open(filepath).convert("RGB")  # Convert PNG to RGB
                rgb = np.array(img)  # Convert to NumPy array
                rgb = norm_im(rgb)  # Normalize

            else:
                raise ValueError(f"Unsupported file type: {filepath}")

            border_size = border_sizes.get(sample_name, 1)

            # Convert white background to blue (255, 255, 255 → 0, 0, 255)
            white_mask = np.all(rgb == [255, 255, 255], axis=-1)
            rgb_bg = rgb.copy()
            rgb_bg[white_mask] = [0, 0, 255]  # Convert white to blue

            # Apply border processing (erosion)
            valid = apply_erosion(white_mask, border_size)
            border = (~white_mask) & (~valid)  # Identify border pixels
            rgb_bg[border] = [255, 0, 0]  # Convert border pixels to red

            # Resize for GUI display
            img_rgb = Image.fromarray(rgb).resize((100, 100))
            img_mod = Image.fromarray(rgb_bg).resize((100, 100))

            images_rgb.append(img_rgb)
            images_mod.append(img_mod)

        # Store the first processed image as the preview for overlay
        preview_images[sample_name] = np.array(images_rgb[0])

        # Clear the sample's display container before adding new images
        for widget in label_container.winfo_children():
            widget.destroy()

        # Display all images for the sample in the GUI
        for img_rgb, img_mod in zip(images_rgb, images_mod):
            raw_photo = ImageTk.PhotoImage(img_rgb)
            raw_label = Label(label_container, image=raw_photo)
            raw_label.image = raw_photo
            raw_label.pack(side=tk.LEFT, padx=2)  # Pack images horizontally

            mod_photo = ImageTk.PhotoImage(img_mod)
            mod_label = Label(label_container, image=mod_photo)
            mod_label.image = mod_photo
            mod_label.pack(side=tk.LEFT, padx=2)

        # Save a reference to the first image label for overlay updates
        preview_labels[sample_name] = raw_label

    except Exception as e:
        print(f"Error in display_channels_with_modifications for {sample_name}: {e}")

def load_sample(sample_name, label_container):
    """
    Loads multiple TIFF, PNG, or BMP files for a sample.
    - TIFF files: Load all available channels.
    - PNG/BMP files: Load only RGB channels.
    Applies erosion-based border processing and normalization.
    Allows adding multiple images to the same sample.
    """
    # Select multiple file paths
    selected_file_paths = filedialog.askopenfilenames(
        title=f"Select Images for {sample_name}",
        filetypes=[
            ("Image Files", "*.tif;*.tiff;*.png;*.bmp"),
            ("TIFF Files", "*.tif;*.tiff"),
            ("PNG Files", "*.png"),
            ("BMP Files", "*.bmp")
        ]
    )

    if selected_file_paths:
        # Initialize file paths for this sample if not existing
        if sample_name not in file_paths:
            file_paths[sample_name] = []
        
        # Add new file paths to the existing list
        file_paths[sample_name].extend(selected_file_paths)

        # Ensure the sample has a border size entry
        if sample_name not in border_sizes:
            border_sizes[sample_name] = 1

        # Display all images, including previously loaded ones
        display_channels_with_modifications(label_container, file_paths[sample_name], sample_name)

        print(f"Loaded {len(selected_file_paths)} images for {sample_name}")
        print(f"Total images for {sample_name}: {len(file_paths[sample_name])}")

def add_sample():
    sample_num = len(samples) + 3
    sample_name = f"Sample {sample_num}"
    frame_sample = Frame(root, bg="lightyellow")
    frame_sample.pack(side=tk.TOP, fill=tk.X, pady=5, before=btn_add_sample)
    
    btn_load = TkButton(frame_sample, text=f"Load {sample_name}", 
                       command=lambda: load_sample(sample_name, img_frame))
    btn_load.pack(side=tk.LEFT, padx=5)
    
    erosion_frame = Frame(frame_sample, bg="lightcyan")
    erosion_frame.pack(side=tk.LEFT, padx=5)
    
    TkButton(erosion_frame, text="Erosion +", 
            command=lambda: increase_border(sample_name, img_frame)).pack(side=tk.TOP, pady=2)
    
    TkButton(erosion_frame, text="Erosion -", 
            command=lambda: decrease_border(sample_name, img_frame)).pack(side=tk.TOP, pady=2)
    
    img_frame = Frame(frame_sample, bg="pink")
    img_frame.pack(side=tk.LEFT, padx=5)
    
    # Add highlight button
    highlight_btn = TkButton(frame_sample, text="Highlight", 
                            command=lambda: highlight_sample(sample_name, img_frame))
    highlight_btn.pack(side=tk.LEFT, padx=5)
    
    samples.append({'name': sample_name, 'frame': frame_sample, 'image_container': img_frame})

def increase_border(sample_name, label_container):
    """Increase erosion effect and update the image preview for all images in the sample."""
    if sample_name in file_paths and file_paths[sample_name]:
        border_sizes[sample_name] += 1
        display_channels_with_modifications(label_container, file_paths[sample_name], sample_name)

def decrease_border(sample_name, label_container):
    """Decrease erosion effect and update the image preview for all images in the sample."""
    if sample_name in file_paths and file_paths[sample_name] and border_sizes[sample_name] > 1:
        border_sizes[sample_name] -= 1
        display_channels_with_modifications(label_container, file_paths[sample_name], sample_name)

def store_sample_data(sample_name, rgb_frame, valid_pixels, anomaly_mask=None):
    """
    Store RGB values from valid pixels for Sample 1, and anomalous pixels for other samples.
    - sample_name: The name of the sample
    - rgb_frame: The RGB image
    - valid_pixels: Boolean mask of valid pixels (non-background, non-border)
    - anomaly_mask: Boolean mask of anomalous pixels (only for non-reference samples)
    """
    global sample_extracted_data
    
    # For Sample 1 (reference), we store ALL valid pixels as normal examples
    if sample_name == "Sample 1":
        # Create fresh entry for Sample 1 if running a new analysis
        if "Sample 1" not in sample_extracted_data:
            sample_extracted_data["Sample 1"] = []
        
        # Extract all valid pixels from the reference sample
        if isinstance(valid_pixels, np.ndarray) and valid_pixels.dtype == bool:
            # It's already a boolean mask
            valid_rgb = rgb_frame[valid_pixels]
        else:
            # It's a tuple of indices
            valid_rgb = rgb_frame[valid_pixels[0], valid_pixels[1]]
        
        # Add all valid pixels to sample_extracted_data for Sample 1
        if len(valid_rgb) > 0:
            # Extend the existing list with new valid pixels
            sample_extracted_data["Sample 1"].extend(valid_rgb)
            print(f"Stored {len(valid_rgb)} valid RGB values for {sample_name}")
            print(f"Total Sample 1 data points: {len(sample_extracted_data['Sample 1'])}")
    
    # For other samples, we only store the anomalous pixels (the defects)
    else:
        if anomaly_mask is None:
            return  # No anomaly mask provided, nothing to store
            
        # Initialize entry if needed
        if sample_name not in sample_extracted_data:
            sample_extracted_data[sample_name] = []
            
        # Extract only the anomalous pixels
        if isinstance(valid_pixels, np.ndarray) and valid_pixels.dtype == bool:
            # Both are boolean masks, we need to combine them
            combined_mask = valid_pixels & anomaly_mask
            valid_rgb = rgb_frame[combined_mask]
        else:
            # valid_pixels is a tuple of indices
            # We need to find indices where both valid_pixels and anomaly_mask are True
            y_indices, x_indices = valid_pixels
            anomalous_indices = []
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                if anomaly_mask[y, x]:
                    anomalous_indices.append(i)
                    
            # If no anomalous pixels, return
            if not anomalous_indices:
                print(f"No anomalous pixels found for {sample_name}")
                return
                
            # Extract RGB values at anomalous indices
            valid_rgb = rgb_frame[y_indices[anomalous_indices], x_indices[anomalous_indices]]
        
        # Add the anomalous pixels to sample_extracted_data
        if len(valid_rgb) > 0:
            # Extend the existing list with new anomalous pixels
            sample_extracted_data[sample_name].extend(valid_rgb)
            print(f"Stored {len(valid_rgb)} anomalous RGB values for {sample_name}")
            print(f"Total {sample_name} data points: {len(sample_extracted_data[sample_name])}")

# -------------------------------------------------------------------
# SAMPLE ANALYSIS FUNCTIONS 
# -------------------------------------------------------------------
def extract_reference_features(sample_name):
    """Extract color features from Sample 1 images to use as reference"""
    if sample_name not in file_paths or not file_paths[sample_name]:
        return None
    
    all_rgb_values = []
    all_lab_values = []
    
    for filepath in file_paths[sample_name]:
        try:
            # Load image
            if filepath.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                    rgb_frame = arr[0]  # First frame is RGB
            elif filepath.lower().endswith(('.png', '.bmp')):
                img = Image.open(filepath).convert("RGB")
                rgb_frame = np.array(img)
            else:
                continue
                
            # Filter background/border
            white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
            border_size = border_sizes.get(sample_name, 1)
            valid_region = ~white_mask
            for _ in range(border_size):
                valid_region = binary_erosion(valid_region)
            border_mask = (~white_mask) & (~valid_region)
            valid_pixels = ~white_mask & ~border_mask
            
            # Skip if no valid pixels
            if np.sum(valid_pixels) == 0:
                continue
                
            # Extract RGB values
            valid_rgb = rgb_frame[valid_pixels]
            all_rgb_values.append(valid_rgb)
            
            # Convert to LAB and extract values
            lab = rgb2lab(rgb_frame)
            valid_lab = lab[valid_pixels]
            all_lab_values.append(valid_lab)
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    if not all_rgb_values or not all_lab_values:
        return None
        
    # Combine all values
    all_rgb = np.vstack(all_rgb_values)
    all_lab = np.vstack(all_lab_values)
    
    # Calculate statistics
    rgb_mean = np.mean(all_rgb, axis=0)
    rgb_std = np.std(all_rgb, axis=0)
    lab_mean = np.mean(all_lab, axis=0)
    lab_std = np.std(all_lab, axis=0)
    
    return {
        'rgb_mean': rgb_mean,
        'rgb_std': rgb_std,
        'lab_mean': lab_mean,
        'lab_std': lab_std,
        'rgb_variance': np.var(all_rgb, axis=0),
        'lab_variance': np.var(all_lab, axis=0)
    }
    
    if not all_lab_values:
        return None
        
    # Combine all values
    all_lab = np.vstack(all_lab_values)
    
    # Calculate statistics
    lab_mean = np.mean(all_lab, axis=0)
    lab_std = np.std(all_lab, axis=0)
    
    return {
        'lab_mean': lab_mean,
        'lab_std': lab_std,
        'lab_variance': np.var(all_lab, axis=0)
    }

def detect_anomalies(reference_model, test_img, sample_name="Sample 2"):
    """
    Detect anomalies in the test image compared to the reference model.
    Uses sample-specific sensitivity parameter to adjust strictness.
    Returns a mask of anomalous pixels.
    """
    global anomaly_sensitivities
    
    # Get sample-specific sensitivity or use default
    sensitivity = anomaly_sensitivities.get(sample_name, 2.5)
    
    # Convert to LAB color space
    lab = rgb2lab(test_img)
    
    # Get reference statistics
    ref_mean = reference_model['lab_mean']
    ref_std = reference_model['lab_std'] + 1e-6  # Avoid division by zero
    
    # Initialize mask
    anomaly_mask = np.zeros(test_img.shape[:2], dtype=bool)
    
    # Find background (white pixels)
    white_mask = np.all(test_img == [255, 255, 255], axis=-1)
    
    # For each non-background pixel
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            if not white_mask[i, j]:
                pixel_lab = lab[i, j]
                
                # Calculate Mahalanobis-like distance
                delta = pixel_lab - ref_mean
                distance = np.sqrt(np.sum((delta / ref_std) ** 2))
                
                # Mark as anomaly if distance exceeds threshold - use sample-specific sensitivity
                if distance > sensitivity:
                    anomaly_mask[i, j] = True
    
    # Optional: Refine with morphological operations
    anomaly_mask = binary_dilation(anomaly_mask, iterations=1)
    anomaly_mask = binary_erosion(anomaly_mask, iterations=1)
    
    return anomaly_mask

def create_highlighted_image(original_img, anomaly_mask):
    """Create a new image with anomalies highlighted in red"""
    # Create copy of original
    highlighted = original_img.copy()
    
    # Apply red overlay to anomalous pixels
    highlighted[anomaly_mask] = [255, 0, 0]  # Pure red
    
    return highlighted

def analyze_sample(sample_name, reference_model):
    """Analyze a sample and return highlighted images - with distance caching and per-sample sensitivity"""
    global cached_distances
    
    if sample_name not in file_paths or not file_paths[sample_name]:
        return []
    
    # Ensure the sample has a sensitivity value
    if sample_name not in anomaly_sensitivities:
        anomaly_sensitivities[sample_name] = 2.5  # Default value
    
    results = []
    
    for filepath in file_paths[sample_name]:
        try:
            # Load image
            if filepath.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                    rgb_frame = arr[0]
            elif filepath.lower().endswith(('.png', '.bmp')):
                img = Image.open(filepath).convert("RGB")
                rgb_frame = np.array(img)
            else:
                continue
            
            # Create masks for background and borders
            white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
            border_size = border_sizes.get(sample_name, 1)
            valid_region = ~white_mask
            for _ in range(border_size):
                valid_region = binary_erosion(valid_region)
            border_mask = (~white_mask) & (~valid_region)
            valid_pixels = ~white_mask & ~border_mask
            
            # Convert original to LAB for analysis
            lab = rgb2lab(rgb_frame)
            
            # Initialize anomaly mask (all False to start)
            anomaly_mask = np.zeros_like(white_mask)
            
            # Create a unique key for this image
            image_key = f"{sample_name}_{filepath}"
            
            # Check if we need to compute distances or if they're cached
            if image_key not in cached_distances:
                # Initialize distance array
                pixel_distances = np.zeros(rgb_frame.shape[:2])
                pixel_distances.fill(np.nan)  # Fill with NaN for invalid pixels
                
                # Get reference statistics from the model
                ref_mean = reference_model['lab_mean']
                ref_std = reference_model['lab_std'] + 1e-6  # Avoid division by zero
                
                # Loop through valid pixels only and calculate distances
                y_indices, x_indices = np.where(valid_pixels)
                for i in range(len(y_indices)):
                    y, x = y_indices[i], x_indices[i]
                    pixel_lab = lab[y, x]
                    
                    # Calculate Mahalanobis-like distance
                    delta = pixel_lab - ref_mean
                    distance = np.sqrt(np.sum((delta / ref_std) ** 2))
                    
                    # Store the distance
                    pixel_distances[y, x] = distance
                
                # Cache the distances
                cached_distances[image_key] = pixel_distances
            else:
                # Use cached distances
                pixel_distances = cached_distances[image_key]
            
            # Get sample-specific sensitivity
            sensitivity = anomaly_sensitivities[sample_name]
            
            # Apply current sensitivity threshold to create mask
            y_indices, x_indices = np.where(valid_pixels)
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                if pixel_distances[y, x] > sensitivity:
                    anomaly_mask[y, x] = True
            
            # Optional: Refine with morphological operations
            anomaly_mask = binary_dilation(anomaly_mask, iterations=1)
            anomaly_mask = binary_erosion(anomaly_mask, iterations=1)
            
            # Create highlighted image
            highlighted = rgb_frame.copy()
            highlighted[anomaly_mask] = [255, 0, 0]  # Mark anomalies as red
            
            # Calculate anomaly percentage based on valid pixels only
            total_valid = np.sum(valid_pixels)
            anomaly_count = np.sum(anomaly_mask)
            if total_valid > 0:
                anomaly_percentage = (anomaly_count / total_valid) * 100
            else:
                anomaly_percentage = 0
            
            # Store the RGB data for this sample
            # For Sample 1, store all valid pixels
            # For other samples, store only anomalous pixels
            store_sample_data(sample_name, rgb_frame, np.where(valid_pixels), anomaly_mask)
            
            # Add to results
            results.append({
                'original': rgb_frame,
                'anomaly_mask': anomaly_mask,
                'highlighted': highlighted,
                'anomaly_percentage': anomaly_percentage,
                'valid_mask': valid_pixels
            })
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {str(e)}")
    
    return results

def update_anomaly_masks_fast():
    """
    Update anomaly masks using cached distances - much faster than full reprocessing
    """
    global sample_extracted_data, cached_distances, cached_reference_model
    
    # Clear existing extracted data - we'll rebuild it with the new sensitivity
    sample_extracted_data = {}
    
    # Check if we have cached data
    if not cached_distances or cached_reference_model is None:
        print("No cached data available - running full analysis")
        update_anomaly_masks()
        return
    
    print(f"Fast-updating masks with sensitivity: {anomaly_sensitivity:.1f}")
    
    # Process Sample 1 (for reference data)
    process_sample_masks_fast("Sample 1")
    
    # Process Sample 2
    if "Sample 2" in file_paths and file_paths["Sample 2"]:
        process_sample_masks_fast("Sample 2")
    
    # Process other samples
    for sample in samples:
        sample_name = sample['name']
        if sample_name in file_paths and file_paths[sample_name]:
            process_sample_masks_fast(sample_name)
    
    print(f"Masks updated with sensitivity {anomaly_sensitivity:.1f}")
    
    # Update display of current sample
    if "Sample 1" in file_paths:
        display_with_masks_fast("Sample 1", img_frame_s1)
    if "Sample 2" in file_paths:
        display_with_masks_fast("Sample 2", img_frame_s2)
    for sample in samples:
        sample_name = sample['name']
        if sample_name in file_paths:
            display_with_masks_fast(sample_name, sample['image_container'])

def process_sample_masks_fast(sample_name):
    """Re-process masks using cached distances with sample-specific sensitivity"""
    if sample_name not in file_paths:
        return
    
    # Make sure the sample has a sensitivity value
    if sample_name not in anomaly_sensitivities:
        anomaly_sensitivities[sample_name] = 2.5  # Default value
    
    # Get sample-specific sensitivity
    sensitivity = anomaly_sensitivities[sample_name]
    
    # Clear existing extracted data for this sample
    if sample_name in sample_extracted_data:
        del sample_extracted_data[sample_name]
    
    for filepath in file_paths[sample_name]:
        try:
            # Create a unique key for this image
            image_key = f"{sample_name}_{filepath}"
            
            # Check if we have cached distances for this image
            if image_key not in cached_distances:
                print(f"Warning: No cached distances for {image_key}")
                continue
            
            # Load image
            if filepath.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                    rgb_frame = arr[0]
            elif filepath.lower().endswith(('.png', '.bmp')):
                img = Image.open(filepath).convert("RGB")
                rgb_frame = np.array(img)
            else:
                continue
            
            # Create masks for background and borders
            white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
            border_size = border_sizes.get(sample_name, 1)
            valid_region = ~white_mask
            for _ in range(border_size):
                valid_region = binary_erosion(valid_region)
            border_mask = (~white_mask) & (~valid_region)
            valid_pixels = ~white_mask & ~border_mask
            
            # For Sample 1, store all valid pixels regardless of anomaly status
            if sample_name == "Sample 1":
                store_sample_data(sample_name, rgb_frame, np.where(valid_pixels), None)
                continue  # No need to check for anomalies in the reference sample
            
            # Get cached distances
            pixel_distances = cached_distances[image_key]
            
            # Initialize anomaly mask
            anomaly_mask = np.zeros_like(white_mask)
            
            # Apply current threshold to distances
            y_indices, x_indices = np.where(valid_pixels)
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                if not np.isnan(pixel_distances[y, x]) and pixel_distances[y, x] > sensitivity:
                    anomaly_mask[y, x] = True
            
            # Refine with morphological operations
            anomaly_mask = binary_dilation(anomaly_mask, iterations=1)
            anomaly_mask = binary_erosion(anomaly_mask, iterations=1)
            
            # Store only the anomalous pixels for non-reference samples
            store_sample_data(sample_name, rgb_frame, np.where(valid_pixels), anomaly_mask)
            
        except Exception as e:
            print(f"Error processing masks for {filepath}: {str(e)}")


def show_analysis_results(sample_name, results):
    """Show the analysis results in a new window"""
    global anomaly_sensitivities
    
    if not results:
        messagebox.showinfo("Analysis", f"No results for {sample_name}")
        return
    
    # Get sample-specific sensitivity
    sensitivity = anomaly_sensitivities.get(sample_name, 2.5)
    
    # Create a new window
    result_window = Toplevel(root)
    result_window.title(f"Anomaly Analysis - {sample_name} (Sensitivity: {sensitivity:.1f})")
    result_window.geometry("800x600")
    
    # Create a frame for the results
    frame = Frame(result_window)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Add title
    title_label = Label(frame, text=f"Anomaly Analysis Results - {sample_name}", font=("Arial", 14, "bold"))
    title_label.pack(pady=(0, 10))
    
    # Create a canvas with scrollbar for multiple images
    canvas_frame = Frame(frame)
    canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(canvas_frame)
    scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Create a frame for the content
    content_frame = Frame(canvas)
    canvas.create_window((0, 0), window=content_frame, anchor="nw")
    
    # Add results
    row = 0
    for i, result in enumerate(results):
        # Create a frame for this result
        result_frame = Frame(content_frame, bd=2, relief=tk.GROOVE, padx=5, pady=5)
        result_frame.grid(row=row, column=0, padx=10, pady=10, sticky="ew")
        
        # Add title
        Label(result_frame, text=f"Image {i+1}", font=("Arial", 12)).grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        # Add images
        # Original image
        original_img = Image.fromarray(result['original'])
        original_img = original_img.resize((150, 150))
        original_photo = ImageTk.PhotoImage(original_img)
        original_label = Label(result_frame, image=original_photo)
        original_label.image = original_photo
        original_label.grid(row=1, column=0, padx=5)
        Label(result_frame, text="Original").grid(row=2, column=0)
        
        # Anomaly mask
        mask_img = np.zeros_like(result['original'])
        mask_img[result['anomaly_mask']] = [255, 255, 255]
        mask_pil = Image.fromarray(mask_img)
        mask_pil = mask_pil.resize((150, 150))
        mask_photo = ImageTk.PhotoImage(mask_pil)
        mask_label = Label(result_frame, image=mask_photo)
        mask_label.image = mask_photo
        mask_label.grid(row=1, column=1, padx=5)
        Label(result_frame, text="Anomaly Mask").grid(row=2, column=1)
        
        # Highlighted image
        highlighted_img = Image.fromarray(result['highlighted'])
        highlighted_img = highlighted_img.resize((150, 150))
        highlighted_photo = ImageTk.PhotoImage(highlighted_img)
        highlighted_label = Label(result_frame, image=highlighted_photo)
        highlighted_label.image = highlighted_photo
        highlighted_label.grid(row=1, column=2, padx=5)
        Label(result_frame, text="Highlighted").grid(row=2, column=2)
        
        # Add stats
        stats_frame = Frame(result_frame)
        stats_frame.grid(row=3, column=0, columnspan=3, pady=5)
        Label(stats_frame, text=f"Anomaly percentage: {result['anomaly_percentage']:.2f}%").pack()
        Label(stats_frame, text=f"Using sensitivity: {sensitivity:.1f}").pack()
        
        row += 1
    
    # Update scroll region
    content_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

def highlight_sample(sample_name, img_container):
    """Analyze a sample and display the highlighted images in the original container"""
    global anomaly_sensitivities
    
    if "Sample 1" not in file_paths or not file_paths["Sample 1"]:
        messagebox.showerror("Error", "Please load Sample 1 images first as reference")
        return
        
    if sample_name not in file_paths or not file_paths[sample_name]:
        messagebox.showerror("Error", f"No images loaded for {sample_name}")
        return
    
    # Extract reference features from Sample 1
    reference_model = extract_reference_features("Sample 1")
    if not reference_model:
        messagebox.showerror("Error", "Failed to extract features from reference")
        return
    
    # Analyze this sample
    results = analyze_sample(sample_name, reference_model)
    if not results:
        messagebox.showerror("Error", f"Failed to analyze {sample_name}")
        return
    
    # Clear container
    for widget in img_container.winfo_children():
        widget.destroy()
    
    # Display original and highlighted images
    for result in results:
        # Original image
        original_img = Image.fromarray(result['original'])
        original_img = original_img.resize((100, 100))
        original_photo = ImageTk.PhotoImage(original_img)
        original_label = Label(img_container, image=original_photo)
        original_label.image = original_photo
        original_label.pack(side=tk.LEFT, padx=2)
        
        # Highlighted image
        highlighted_img = Image.fromarray(result['highlighted'])
        highlighted_img = highlighted_img.resize((100, 100))
        highlighted_photo = ImageTk.PhotoImage(highlighted_img)
        highlighted_label = Label(img_container, image=highlighted_photo)
        highlighted_label.image = highlighted_photo
        highlighted_label.pack(side=tk.LEFT, padx=2)
    
    # Optional: Show detailed results in a new window
    if len(results) > 0:
        show_button = TkButton(img_container, text="Show Details", 
                             command=lambda: show_analysis_results(sample_name, results))
        show_button.pack(side=tk.LEFT, padx=5)

def perform_sample_analysis():
    """Analyze all samples and show results"""
    global sample_extracted_data, cached_distances, cached_reference_model, anomaly_sensitivities
    
    # Clear existing cached data
    sample_extracted_data = {}
    cached_distances = {}
    cached_reference_model = None
    
    # Remove Sample 1 from anomaly_sensitivities if it exists
    if "Sample 1" in anomaly_sensitivities:
        del anomaly_sensitivities["Sample 1"]
    
    if "Sample 1" not in file_paths or not file_paths["Sample 1"]:
        messagebox.showerror("Error", "Please load Sample 1 images first as reference")
        return
    
    # Extract reference features and cache them
    reference_model = extract_reference_features("Sample 1")
    if not reference_model:
        messagebox.showerror("Error", "Failed to extract features from reference")
        return
    
    # Cache the reference model
    cached_reference_model = reference_model
    
    # Show reference model info
    print("Reference model statistics:")
    print(f"  RGB mean: {reference_model['rgb_mean']}")
    print(f"  RGB std: {reference_model['rgb_std']}")
    print(f"  LAB mean: {reference_model['lab_mean']}")
    print(f"  LAB std: {reference_model['lab_std']}")
    
    # Ensure all samples (except Sample 1) have sensitivity values
    if "Sample 2" not in anomaly_sensitivities:
        anomaly_sensitivities["Sample 2"] = 2.5
    
    # Analyze each sample with a try-except block to continue even if one fails
    try:
        # Analyze Sample 1 (reference) - this will cache distances and store ALL valid pixels
        highlight_sample("Sample 1", img_frame_s1)
    except Exception as e:
        print(f"Error analyzing Sample 1: {str(e)}")
    
    try:
        # Analyze Sample 2 - this will cache distances and store only anomalous pixels
        if "Sample 2" in file_paths and file_paths["Sample 2"]:
            highlight_sample("Sample 2", img_frame_s2)
    except Exception as e:
        print(f"Error analyzing Sample 2: {str(e)}")
    
    # Analyze other samples - this will cache distances
    for sample in samples:
        try:
            sample_name = sample['name']
            if sample_name in file_paths and file_paths[sample_name]:
                # Ensure sample has a sensitivity value
                if sample_name not in anomaly_sensitivities:
                    anomaly_sensitivities[sample_name] = 2.5
                highlight_sample(sample_name, sample['image_container'])
        except Exception as e:
            print(f"Error analyzing {sample_name}: {str(e)}")
    
    # Update Sample 1 highlights based on other samples' sensitivities
    update_sample1_highlights()
    
    # Update all sensitivity labels
    update_sensitivity_labels()
    
    # Debug info - print counts of stored data
    for name, data in sample_extracted_data.items():
        print(f"Sample {name}: {len(data)} stored data points")
    
    messagebox.showinfo("Analysis Complete", "All samples have been analyzed")

# Create a dictionary to store sensitivity label widgets
sensitivity_value_labels = {}

def create_sample_sensitivity_controls():
    """Create sensitivity controls for each sample"""
    global sensitivity_value_labels
    
    # Clear existing sensitivity controls first
    for frame in sensitivity_control_frames.values():
        frame.destroy()
    sensitivity_control_frames.clear()
    sensitivity_value_labels.clear()
    
    # Create controls for built-in samples
    create_sensitivity_control("Sample 1", frame_s1)
    create_sensitivity_control("Sample 2", frame_s2)
    
    # Create controls for dynamically added samples
    for sample in samples:
        sample_name = sample['name']
        create_sensitivity_control(sample_name, sample['frame'])

# Dictionary to store sensitivity control frames
sensitivity_control_frames = {}

def create_sensitivity_control(sample_name, parent_frame):
    """Create sensitivity control for a specific sample (except Sample 1) with fine-grained adjustments"""
    global sensitivity_value_labels, sensitivity_control_frames, anomaly_sensitivities
    
    # Skip Sample 1 - no sensitivity controls needed
    if sample_name == "Sample 1":
        return
    
    # Create a frame for sensitivity controls
    sensitivity_frame = Frame(parent_frame, bg="lightgray", bd=1, relief=tk.RAISED)
    sensitivity_frame.pack(side=tk.RIGHT, padx=10, pady=5)
    sensitivity_control_frames[sample_name] = sensitivity_frame
    
    # Add a label for the sensitivity controls
    sensitivity_label = Label(sensitivity_frame, text="Sens:")
    sensitivity_label.pack(side=tk.LEFT, padx=2)
    
    # Add fine decrement button (--) for -0.25
    fine_minus_btn = TkButton(sensitivity_frame, text="--", width=2, 
                          command=lambda: big_decrease_sensitivity(sample_name))
    fine_minus_btn.pack(side=tk.LEFT, padx=1)
    
    # Add regular decrement button (-) for -0.05
    sensitivity_decrease_btn = TkButton(sensitivity_frame, text="-", width=2, 
                                    command=lambda: decrease_sample_sensitivity(sample_name))
    sensitivity_decrease_btn.pack(side=tk.LEFT, padx=1)
    
    # Add a label to display the current sensitivity value with more precision
    current_sensitivity = anomaly_sensitivities.get(sample_name, 2.5)
    sensitivity_value_label = Label(sensitivity_frame, text=f"{current_sensitivity:.2f}", width=4)
    sensitivity_value_label.pack(side=tk.LEFT, padx=2)
    sensitivity_value_labels[sample_name] = sensitivity_value_label
    
    # Add regular increment button (+) for +0.05
    sensitivity_increase_btn = TkButton(sensitivity_frame, text="+", width=2,
                                    command=lambda: increase_sample_sensitivity(sample_name))
    sensitivity_increase_btn.pack(side=tk.LEFT, padx=1)
    
    # Add fine increment button (++) for +0.25
    fine_plus_btn = TkButton(sensitivity_frame, text="++", width=2,
                         command=lambda: big_increase_sensitivity(sample_name))
    fine_plus_btn.pack(side=tk.LEFT, padx=1)

def big_increase_sensitivity(sample_name):
    global anomaly_sensitivities
    if sample_name in anomaly_sensitivities and sample_name != "Sample 1":
        # Larger increment (0.25)
        anomaly_sensitivities[sample_name] += 0.25
        update_sensitivity_labels(sample_name)
        # Update the affected sample
        update_sample_masks_fast(sample_name)
        # Also update Sample 1 highlights based on new sensitivity
        update_sample1_highlights()

def big_decrease_sensitivity(sample_name):
    global anomaly_sensitivities
    if sample_name in anomaly_sensitivities and sample_name != "Sample 1":
        # Larger decrement (0.25)
        # Don't go below 0.5
        anomaly_sensitivities[sample_name] = max(0.5, anomaly_sensitivities[sample_name] - 0.25)
        update_sensitivity_labels(sample_name)
        # Update the affected sample
        update_sample_masks_fast(sample_name)
        # Also update Sample 1 highlights based on new sensitivity
        update_sample1_highlights()

def create_ellipsoid(rx, ry, rz, rotation_matrix, tx, ty, tz, n=50):
    """
    Creates a 3D ellipsoid for plotting.
    """
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    x *= rx; y *= ry; z *= rz
    coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    coords = coords @ rotation_matrix.T
    coords += np.array([tx, ty, tz])
    X = coords[:,0].reshape(n, n)
    Y = coords[:,1].reshape(n, n)
    Z = coords[:,2].reshape(n, n)
    return X, Y, Z

def rotation_matrix_x(angle_degs):
    a = np.radians(angle_degs)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def rotation_matrix_y(angle_degs):
    a = np.radians(angle_degs)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotation_matrix_z(angle_degs):
    a = np.radians(angle_degs)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def matrix_to_eulerXYZ(M):
    sy = np.sqrt(M[0,0]**2 + M[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        rx = np.arctan2(M[2,1], M[2,2])
        ry = np.arctan2(-M[2,0], sy)
        rz = np.arctan2(M[1,0], M[0,0])
    else:
        rx = np.arctan2(-M[1,2], M[1,1])
        ry = np.arctan2(-M[2,0], sy)
        rz = 0
    return np.degrees(rx), np.degrees(ry), np.degrees(rz)

def mvee(P, tol=0.001, max_iter=100, reg=1e-6):
    """
    Computes the Minimum Volume Enclosing Ellipsoid (MVEE) for a set of points.
    
    Parameters:
      - P: an (n x d) matrix of n points in d dimensions.
      - tol: tolerance for convergence.
      - max_iter: maximum number of iterations allowed.
      - reg: regularization term added to the matrix to avoid singularity.
      
    Returns:
      - c: the center of the ellipsoid (d-dimensional vector).
      - A: the shape matrix such that 
           E = { x | (x-c)^T A (x-c) <= 1 }.
      - iterations: the number of iterations performed.
    """
    n, d = P.shape
    Q = np.column_stack((P, np.ones(n))).T  # shape: (d+1, n)
    u = np.ones(n) / n
    err = tol + 1.0
    
    # Initialize iteration counter
    iterations = 0
    
    while err > tol and iterations < max_iter:
        X = Q @ np.diag(u) @ Q.T
        # Add regularization to avoid singular matrix issues:
        X += np.eye(X.shape[0]) * reg
        M = np.einsum('ij,jk,ik->i', Q.T, np.linalg.inv(X), Q.T)
        j = np.argmax(M)
        step_size = (M[j] - d - 1) / ((d+1) * (M[j]-1))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u
        
        # Increment iteration counter
        iterations += 1
    
    c = P.T @ u
    # When computing A, add a regularization term:
    A = np.linalg.inv(P.T @ np.diag(u) @ P - np.outer(c, c) + np.eye(d)*reg) / d
    
    # Return the center, shape matrix, and number of iterations
    return c, A, iterations

def compute_ellipsoid(points, centroid, reg=1e-6, max_iter=100):
    """
    Given a set of points and a centroid, compute the MVEE-based ellipsoid parameters.
    Returns a tuple containing:
    1. (rx, ry, rz) - the radii of the ellipsoid
    2. eigvecs - the rotation matrix
    3. iterations - number of iterations used
    4. eigenvalues - the eigenvalues
    5. eigvecs - the eigenvectors
    
    A small regularization term 'reg' is added to the covariance matrix to avoid singularity.
    """
    C = np.cov(points, rowvar=False)
    # Add a small value to the diagonal to regularize the covariance matrix
    C += np.eye(C.shape[0]) * reg
    from scipy.stats import chi2
    C_inv = np.linalg.inv(C)
    mahal_sq = np.array([ (p - centroid).T @ C_inv @ (p - centroid) for p in points ])
    threshold_m = chi2.ppf(0.75, df=points.shape[1])
    core_points = points[mahal_sq <= threshold_m]
    if core_points.shape[0] < 2:
        core_points = points
    
    # Call mvee with max_iter parameter
    c_mvee, A, iterations = mvee(core_points, max_iter=max_iter)
    
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    rx = 1 / np.sqrt(eigvals[0])
    ry = 1 / np.sqrt(eigvals[1])
    rz = 1 / np.sqrt(eigvals[2])
    
    # Return iterations as the third parameter
    return (rx, ry, rz), eigvecs, iterations, eigvals, eigvecs
                          
def apply_overlay_to_image(raw_img, ellipsoid_params, overlay_color):
    # raw_img is assumed to be (H, W, 3) in pixel coordinates.
    # Optionally, define a scale factor (tweak as needed).
    scale_factor = 1.0  # Change this if needed, e.g., 0.1 if PCA space is 10× larger.
    
    img = raw_img.astype(np.float32)
    H, W, C = img.shape
    pixels = img.reshape(-1, C)
    
    # Apply scaling to ellipsoid parameters:
    centroid = np.array(ellipsoid_params["centroid"]) * scale_factor
    rx = ellipsoid_params["rx"] * scale_factor
    ry = ellipsoid_params["ry"] * scale_factor
    rz = ellipsoid_params["rz"] * scale_factor
    R = ellipsoid_params["R"]  # (Assume rotation does not need scaling)
    
    local = (pixels - centroid) @ R.T
    vals = (local[:, 0] / rx)**2 + (local[:, 1] / ry)**2 + (local[:, 2] / rz)**2
    mask = vals <= 1
    overlay = np.array(to_rgba(overlay_color))[:3] * 255
    new_pixels = pixels.copy()
    new_pixels[mask] = (pixels[mask] + overlay) / 2
    new_img = new_pixels.reshape(H, W, C).clip(0,255).astype(np.uint8)
    
    return Image.fromarray(new_img)

def update_overlay_for_sample(sample_name):
    """
    Updates the Tkinter preview Label for the given sample with an overlay.
    """
    if sample_name not in preview_images or sample_name not in preview_labels:
        return
    raw_img = preview_images[sample_name]
    # Choose overlay color based on scatter color.
    color_scheme = {"Sample 1": "green", "Sample 2": "red"}
    overlay_color = color_scheme.get(sample_name, "blue")
    scale_factor= 1
    if sample_name == current_sample and current_sample is not None:
        ellipsoid_params = {
            "centroid": np.array([adjust_params["tx"], adjust_params["ty"], adjust_params["tz"]]),
            "rx": adjust_params["rx"]*6.11,
            "ry": adjust_params["ry"]*3.7,
            "rz": adjust_params["rz"]*2.7,
            "R": (rotation_matrix_z(adjust_params["rot_z"]) @
                  rotation_matrix_y(adjust_params["rot_y"]) @
                  rotation_matrix_x(adjust_params["rot_x"]))
        }
    else:
        ellipsoid_params = cached_ellipsoid_params.get(sample_name, None)
        if ellipsoid_params is None:
            return
    overlay_img = apply_overlay_to_image(raw_img, ellipsoid_params, overlay_color)
    new_photo = ImageTk.PhotoImage(overlay_img)
    preview_labels[sample_name].configure(image=new_photo)
    preview_labels[sample_name].image = new_photo

def update_sample1_highlights():
    """
    Check Sample 1 pixels against all other samples' sensitivity thresholds.
    Highlight any pixels that would be considered anomalous by any sample.
    Updates ALL images in Sample 1 container.
    """
    global sample_extracted_data, cached_distances, cached_reference_model, anomaly_sensitivities
    
    # Check if we have Sample 1 data
    if "Sample 1" not in file_paths or not file_paths["Sample 1"]:
        return
    
    # Check if we have cached distances
    if not cached_distances:
        return
    
    # Clear existing Sample 1 container first
    for widget in img_frame_s1.winfo_children():
        if not isinstance(widget, TkButton):  # Keep buttons
            widget.destroy()
    
    # Clear existing Sample 1 extracted data
    if "Sample 1" in sample_extracted_data:
        sample_extracted_data["Sample 1"] = []
    
    # Process each Sample 1 image
    for filepath in file_paths["Sample 1"]:
        try:
            # Create a unique key for this image
            image_key = f"Sample 1_{filepath}"
            
            # Check if we have cached distances for this image
            if image_key not in cached_distances:
                print(f"Warning: No cached distances for {image_key}")
                continue
            
            # Load the image
            if filepath.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                    rgb_frame = arr[0]
            elif filepath.lower().endswith(('.png', '.bmp')):
                img = Image.open(filepath).convert("RGB")
                rgb_frame = np.array(img)
            else:
                continue
            
            # Get masks
            white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
            border_size = border_sizes.get("Sample 1", 1)
            valid_region = ~white_mask
            for _ in range(border_size):
                valid_region = binary_erosion(valid_region)
            border_mask = (~white_mask) & (~valid_region)
            valid_pixels = ~white_mask & ~border_mask
            
            # Get the cached distances
            pixel_distances = cached_distances[image_key]
            
            # Initialize arrays to track highlighted pixels
            highlighted_mask = np.zeros_like(white_mask)
            
            # Check each pixel against all other samples' sensitivities
            y_indices, x_indices = np.where(valid_pixels)
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                
                # Skip pixels with NaN distances
                if np.isnan(pixel_distances[y, x]):
                    continue
                    
                # Check if this pixel would be anomalous according to any other sample
                for other_sample, sensitivity in anomaly_sensitivities.items():
                    if other_sample != "Sample 1":
                        if pixel_distances[y, x] > sensitivity:
                            highlighted_mask[y, x] = True
                            # Store the RGB of this pixel for display
                            if "Sample 1" not in sample_extracted_data:
                                sample_extracted_data["Sample 1"] = []
                            sample_extracted_data["Sample 1"].append(rgb_frame[y, x])
                            break  # Once we know it's anomalous, no need to check other samples
            
            # Create an overlay image
            highlighted = rgb_frame.copy()
            highlighted[white_mask] = [0, 0, 255]  # Blue background
            highlighted[border_mask] = [255, 0, 0]  # Red border
            highlighted[highlighted_mask] = [255, 165, 0]  # Orange for highlighted pixels
            
            # Display both original and highlighted images side by side
            # Original image (with background in blue for consistency)
            original_colored = rgb_frame.copy()
            original_colored[white_mask] = [0, 0, 255]  # Blue background
            original_colored[border_mask] = [255, 0, 0]  # Red border
            
            original_img = Image.fromarray(original_colored).resize((100, 100))
            original_photo = ImageTk.PhotoImage(original_img)
            original_label = Label(img_frame_s1, image=original_photo)
            original_label.image = original_photo
            original_label.pack(side=tk.LEFT, padx=2)
            
            # Highlighted image
            highlighted_img = Image.fromarray(highlighted).resize((100, 100))
            highlighted_photo = ImageTk.PhotoImage(highlighted_img)
            highlighted_label = Label(img_frame_s1, image=highlighted_photo)
            highlighted_label.image = highlighted_photo
            highlighted_label.pack(side=tk.LEFT, padx=2)
            
            # Add message about which image this is
            print(f"Updated Sample 1 image: {filepath}")
            
        except Exception as e:
            print(f"Error updating Sample 1 highlights for {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()

# Also update display_with_masks_fast to ensure consistent background coloring
def display_with_masks_fast(sample_name, container):
    """Update display with current masks - optimized version with per-sample sensitivity"""
    global anomaly_sensitivities
    
    if sample_name not in file_paths:
        return
    
    # Get sample-specific sensitivity
    sensitivity = anomaly_sensitivities.get(sample_name, 2.5)
    
    # Clear container
    for widget in container.winfo_children():
        if not isinstance(widget, TkButton):  # Keep buttons
            widget.destroy()
    
    for filepath in file_paths[sample_name]:
        try:
            # Load image
            if filepath.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                    rgb_frame = arr[0]
            elif filepath.lower().endswith(('.png', '.bmp')):
                img = Image.open(filepath).convert("RGB")
                rgb_frame = np.array(img)
            else:
                continue
            
            # Create masks
            white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
            border_size = border_sizes.get(sample_name, 1)
            valid_region = ~white_mask
            for _ in range(border_size):
                valid_region = binary_erosion(valid_region)
            border_mask = (~white_mask) & (~valid_region)
            valid_pixels = ~white_mask & ~border_mask
            
            # Display background in blue and border in red
            rgb_display = rgb_frame.copy()
            rgb_display[white_mask] = [0, 0, 255]  # Blue background
            rgb_display[border_mask] = [255, 0, 0]  # Red border
            
            # Create a unique key for this image
            image_key = f"{sample_name}_{filepath}"
            
            # If we have cached distances, use them to highlight anomalies
            if image_key in cached_distances:
                pixel_distances = cached_distances[image_key]
                
                # Apply current threshold to distances
                y_indices, x_indices = np.where(valid_pixels)
                for i in range(len(y_indices)):
                    y, x = y_indices[i], x_indices[i]
                    # Check if it's an anomaly based on sample-specific sensitivity
                    if not np.isnan(pixel_distances[y, x]) and pixel_distances[y, x] > sensitivity:
                        rgb_display[y, x] = [255, 0, 0]  # Red for anomaly
            
            # Create original image with just background and border colored (for non-Sample 1)
            if sample_name != "Sample 1":
                original_colored = rgb_frame.copy()
                original_colored[white_mask] = [0, 0, 255]  # Blue background
                original_colored[border_mask] = [255, 0, 0]  # Red border
                
                # Display original
                orig_img = Image.fromarray(original_colored).resize((100, 100))
                orig_photo = ImageTk.PhotoImage(orig_img)
                orig_label = Label(container, image=orig_photo)
                orig_label.image = orig_photo
                orig_label.pack(side=tk.LEFT, padx=2)
            
            # Display highlighted
            img_display = Image.fromarray(rgb_display).resize((100, 100))
            photo = ImageTk.PhotoImage(img_display)
            label = Label(container, image=photo)
            label.image = photo
            label.pack(side=tk.LEFT, padx=2)
            
        except Exception as e:
            print(f"Error displaying masks for {filepath}: {str(e)}")

def update_sample_masks_fast(sample_name):
    """
    Update anomaly masks for a specific sample using cached distances
    """
    global sample_extracted_data, cached_distances, cached_reference_model
    
    # Check if we have cached data
    if not cached_distances or cached_reference_model is None:
        print(f"No cached data available for {sample_name} - running full analysis")
        # We would need to implement a per-sample full update function here
        # but for simplicity, just call the existing function
        update_anomaly_masks()
        return
    
    print(f"Fast-updating masks for {sample_name} with sensitivity: {anomaly_sensitivities[sample_name]:.1f}")
    
    # Process just this sample
    process_sample_masks_fast(sample_name)
    
    # Update display of this sample
    if sample_name == "Sample 1":
        display_with_masks_fast("Sample 1", img_frame_s1)
    elif sample_name == "Sample 2":
        display_with_masks_fast("Sample 2", img_frame_s2)
    else:
        # Find the sample's image container
        for sample in samples:
            if sample['name'] == sample_name:
                display_with_masks_fast(sample_name, sample['image_container'])
                break

def update_sample1_colors_from_current_lda(sample_plot_dict, selected_sample):
    global filtered_points_by_sample, cached_ellipsoid_params

    if "Sample 1" not in filtered_points_by_sample:
        return

    sample1_points = filtered_points_by_sample["Sample 1"]

    # Default colors
    default_color = np.array(to_rgba("green"))  # Default green for Sample 1
    highlight_color = np.array(to_rgba("darkgreen"))  # Dark Green for inside points
    highlight_edge = np.array(to_rgba("orange"))  # Orange outline for inside points

    if selected_sample not in cached_ellipsoid_params:
        return  # Exit if selected sample has no ellipsoid

    # Get ellipsoid parameters for the selected sample
    params = cached_ellipsoid_params[selected_sample]
    centroid = np.array(params["centroid"])
    rx, ry, rz = params["rx"], params["ry"], params["rz"]
    R = params["R"]  # Rotation matrix

    # Transform Sample 1 points into the ellipsoid's local coordinate system
    local_coords = (sample1_points - centroid) @ R.T

    # Check if each point falls inside the ellipsoid (Mahalanobis distance)
    vals = (local_coords[:, 0] / rx)**2 + (local_coords[:, 1] / ry)**2 + (local_coords[:, 2] / rz)**2
    mask_inside = vals <= 1  # Points inside the ellipsoid

    # Update scatter plot colors
    sc = sample_plot_dict["Sample 1"]["scatter_points"]
    n_points = sc.get_facecolors().shape[0]
    
    # Create new colors for all points
    new_fc = np.tile(default_color, (n_points, 1))  # Default to green
    new_ec = np.tile(default_color, (n_points, 1))  # Default edge color

    # Apply dark green fill and orange edge for inside points
    new_fc[mask_inside] = highlight_color
    new_ec[mask_inside] = highlight_edge

    # Update scatter plot colors
    sc.set_facecolors(new_fc)
    sc.set_edgecolors(new_ec)

    # Redraw the figure
    sc.figure.canvas.draw_idle()

def update_sample1_colors_from_current(sample_plot_dict):
    global filtered_points_by_sample, adjust_params
    default_color = np.array(to_rgba("green"))
    orange = np.array(to_rgba("orange"))
    if "Sample 1" not in filtered_points_by_sample:
        return
    sample1_points = filtered_points_by_sample["Sample 1"]
    E = np.array([adjust_params["tx"], adjust_params["ty"], adjust_params["tz"]])
    rx = adjust_params["rx"]
    ry = adjust_params["ry"]
    rz = adjust_params["rz"]
    R_current = (rotation_matrix_z(adjust_params["rot_z"]) @
                 rotation_matrix_y(adjust_params["rot_y"]) @
                 rotation_matrix_x(adjust_params["rot_x"]))
    local_coords = (sample1_points - E) @ R_current.T
    vals = (local_coords[:, 0] / rx)**2 + (local_coords[:, 1] / ry)**2 + (local_coords[:, 2] / rz)**2
    mask_inside = vals <= 1
    sc = sample_plot_dict["Sample 1"]["scatter_points"]
    n_points = sc.get_facecolors().shape[0]
    new_fc = np.tile(default_color, (n_points, 1))
    new_ec = np.tile(default_color, (n_points, 1))
    new_fc[mask_inside] = orange
    new_ec[mask_inside] = orange
    sc.set_facecolors(new_fc)
    sc.set_edgecolors(new_ec)
    
def update_ellipsoid(selected_sample, ax, sample_plot_dict):
    global cached_ellipsoid_params, current_sample
    
    # 1) Remove the old ellipsoid surface if present
    if "ellipsoid_surface" in sample_plot_dict[selected_sample]:
        sample_plot_dict[selected_sample]["ellipsoid_surface"].remove()
        del sample_plot_dict[selected_sample]["ellipsoid_surface"]
    
    # 2) Remove any old axis lines and text labels
    old_lines = sample_plot_dict[selected_sample].get("axes_lines", [])
    old_texts = sample_plot_dict[selected_sample].get("axis_labels", [])
    for ln in old_lines:
        ln.remove()  # remove line from the figure
    for txt in old_texts:
        txt.remove() # remove text from the figure
    sample_plot_dict[selected_sample]["axes_lines"] = []
    sample_plot_dict[selected_sample]["axis_labels"] = []
    
    # 3) Re-create the ellipsoid surface
    rx = adjust_params["rx"]
    ry = adjust_params["ry"]
    rz = adjust_params["rz"]
    tx = adjust_params["tx"]
    ty = adjust_params["ty"]
    tz = adjust_params["tz"]
    rot_x = adjust_params["rot_x"]
    rot_y = adjust_params["rot_y"]
    rot_z = adjust_params["rot_z"]
    
    R = (rotation_matrix_z(rot_z) @
         rotation_matrix_y(rot_y) @
         rotation_matrix_x(rot_x))
    
    X_e, Y_e, Z_e = create_ellipsoid(rx, ry, rz, R, tx, ty, tz)
    new_surf = ax.plot_surface(
        X_e, Y_e, Z_e,
        color=sample_plot_dict[selected_sample]["scatter_points"].get_facecolor()[0],
        alpha=0.4, edgecolor='none'
    )
    sample_plot_dict[selected_sample]["ellipsoid_surface"] = new_surf
    
    # 4) Re-add axis lines (using your original R/G/B color scheme)
    extension_factor = 1.25
    cx, cy, cz = tx, ty, tz
    
    line_x = ax.plot(
        [cx - extension_factor * rx, cx + extension_factor * rx],
        [cy, cy],
        [cz, cz],
        color='red', linestyle='--', linewidth=1.5
    )[0]
    line_y = ax.plot(
        [cx, cx],
        [cy - extension_factor * ry, cy + extension_factor * ry],
        [cz, cz],
        color='green', linestyle='--', linewidth=1.5
    )[0]
    line_z = ax.plot(
        [cx, cx],
        [cy, cy],
        [cz - extension_factor * rz, cz + extension_factor * rz],
        color='blue', linestyle='--', linewidth=1.5
    )[0]
    
    # 5) Re-add axis labels in the same R/G/B colors
    text_x = ax.text(cx + extension_factor * rx, cy, cz,  "X",
                     color='red', fontsize=12, fontweight='bold')
    text_y = ax.text(cx, cy + extension_factor * ry, cz,  "Y",
                     color='green', fontsize=12, fontweight='bold')
    text_z = ax.text(cx, cy, cz + extension_factor * rz,  "Z",
                     color='blue', fontsize=12, fontweight='bold')
    
    # Show or hide based on whether this sample is the "current_sample"
    lines_visible = (selected_sample == current_sample)
    for ln in (line_x, line_y, line_z):
        ln.set_visible(lines_visible)
    for txt in (text_x, text_y, text_z):
        txt.set_visible(lines_visible)
    
    # Save references so we can remove them on the next update
    sample_plot_dict[selected_sample]["axes_lines"] = [line_x, line_y, line_z]
    sample_plot_dict[selected_sample]["axis_labels"] = [text_x, text_y, text_z]
    
    # ...
    # 6) Update coloring of points (e.g., Sample 1 highlights, etc.)
    update_sample1_colors_from_current(sample_plot_dict)

    # *** NEW STEP: highlight any Sample 1 points inside other ellipsoids ***
    update_sample1_colors_against_all_ellipsoids(sample_plot_dict)

    ax.figure.canvas.draw_idle()


def add_distance_text_to_figure(fig, ax):
    """ Adds a text box to the top-left corner of Figure 1 showing distances from Sample 1. """
    global cached_ellipsoid_params

    s1_params = cached_ellipsoid_params.get("Sample 1", None)
    if s1_params is None:
        return
    
    s1_centroid = s1_params["centroid"]
    s1_radii = np.array([s1_params["rx"], s1_params["ry"], s1_params["rz"]])

    distances_text = "Distance from Sample 1:\n"
    overlap_text = "Mahalanobis Overlap Distances:\n"

    for sname, params in cached_ellipsoid_params.items():
        if sname == "Sample 1":
            continue
        
        # Compute Euclidean distance between centroids
        distance = np.linalg.norm(s1_centroid - params["centroid"])
        distances_text += f"{sname}: {distance:.2f}\n"
        
        # Compute Mahalanobis overlap ONLY using Sample 1 as the reference
        sample_radii = np.array([params["rx"], params["ry"], params["rz"]])

        # Use only Sample 1's covariance for Mahalanobis calculation
        s1_cov_matrix = np.cov(np.vstack([s1_radii, s1_radii]).T) + np.eye(3) * 1e-6
        inv_cov = np.linalg.pinv(s1_cov_matrix)

        # Compute Mahalanobis distance from Sample 1 to this sample
        mahal_dist = mahalanobis(s1_radii, sample_radii, inv_cov)
        overlap_text += f"{sname}: {mahal_dist:.2f}\n"

    # Combine text
    text_box_content = distances_text + "\n" + overlap_text

    # Remove existing text box if it exists
    if hasattr(fig, "dist_text"):
        fig.dist_text.remove()

    # Add the text box in the figure's top-left (outside plot)
    fig.dist_text = fig.text(
        0.02, 0.98, text_box_content, ha='left', va='top',
        fontsize=10, bbox=dict(facecolor='white', alpha=0.5)
    )

    fig.canvas.draw_idle()

def compute_mahalanobis_overlap(sample1_name, sample2_name):
    """
    Computes the Mahalanobis distance between the ellipsoids of two samples.
    """
    global cached_ellipsoid_params

    if sample1_name not in cached_ellipsoid_params or sample2_name not in cached_ellipsoid_params:
        return None  # One of the samples is missing

    params1 = cached_ellipsoid_params[sample1_name]
    params2 = cached_ellipsoid_params[sample2_name]

    centroid1 = params1["centroid"]
    centroid2 = params2["centroid"]

    # Compute the pooled covariance matrix from both ellipsoids
    cov1 = np.diag([params1["rx"]**2, params1["ry"]**2, params1["rz"]**2])
    cov2 = np.diag([params2["rx"]**2, params2["ry"]**2, params2["rz"]**2])
    pooled_cov = (cov1 + cov2) / 2  # Average covariance

    try:
        # Compute Mahalanobis distance
        inv_cov = np.linalg.pinv(pooled_cov)  # Pseudo-inverse for numerical stability
        mahalanobis_dist = np.sqrt((centroid1 - centroid2).T @ inv_cov @ (centroid1 - centroid2))
        return mahalanobis_dist
    except np.linalg.LinAlgError:
        return None  # Return None if covariance matrix is singular

def add_distance_text_to_figure(fig, ax):
    """ Adds a text box to Figure 1 (outside the plot) showing distances from Sample 1. """
    global cached_ellipsoid_params

    s1_centroid = cached_ellipsoid_params.get("Sample 1", {}).get("centroid", None)
    if s1_centroid is None:
        return  # No Sample 1 found, do nothing

    distances_text = "Distance from Sample 1:\n"
    mahalanobis_text = "Mahalanobis Overlap:\n"

    for sname, params in cached_ellipsoid_params.items():
        if sname != "Sample 1":
            # Compute Euclidean distance
            euclidean_distance = np.linalg.norm(s1_centroid - params["centroid"])
            distances_text += f"{sname}: {euclidean_distance:.2f}\n"

            # Compute Mahalanobis overlap
            mahal_dist = compute_mahalanobis_overlap("Sample 1", sname)
            mahalanobis_text += f"{sname}: {mahal_dist:.2f}\n" if mahal_dist is not None else f"{sname}: N/A\n"

    # Add text box to Figure 1 (not inside the plot)
    text_box_content = distances_text + "\n" + mahalanobis_text
    if hasattr(fig, "dist_text"):
        fig.dist_text.remove()  # Remove existing text if any
    fig.dist_text = fig.text(0.05, 0.95, text_box_content, fontsize=10,
                             verticalalignment='top', transform=fig.transFigure,
                             bbox=dict(facecolor='white', alpha=0.5))

    fig.canvas.draw_idle()

def update_sample1_colors_against_all_ellipsoids(sample_plot_dict):
    """
    Checks every point from 'Sample 1' to see if it falls inside any other sample's ellipsoid.
    If inside at least one, color that point yellow. Otherwise, revert to green.
    """
    global filtered_points_by_sample, cached_ellipsoid_params

    # If Sample 1 doesn't exist or hasn't been computed, do nothing
    if "Sample 1" not in filtered_points_by_sample:
        return

    # Retrieve Sample 1's 3D points and the scatter object
    s1_points = filtered_points_by_sample["Sample 1"]
    scatter_obj = sample_plot_dict["Sample 1"]["scatter_points"]

    # Prepare default (green) or any color scheme you prefer
    default_color_rgba = np.array(to_rgba("green"))
    highlight_color_rgba = np.array(to_rgba("yellow"))

    n_points = len(s1_points)
    new_facecolors = np.tile(default_color_rgba, (n_points, 1))

    # We'll build a union "inside_any_ellipsoid" mask
    inside_any_ellipsoid = np.zeros(n_points, dtype=bool)

    # Loop over all ellipsoids EXCEPT Sample 1's own ellipsoid
    for other_sample_name, ell_params in cached_ellipsoid_params.items():
        if other_sample_name == "Sample 1":
            continue  # Skip checking Sample 1's own ellipsoid if you only care about "other" ellipsoids

        # Get center/radii/rotation
        c = ell_params["centroid"]
        rx = ell_params["rx"]
        ry = ell_params["ry"]
        rz = ell_params["rz"]
        R = ell_params["R"]  # 3x3 rotation

        # Transform Sample1 points into that ellipsoid's local coords
        local_coords = (s1_points - c) @ R.T
        vals = (local_coords[:, 0] / rx)**2 + \
               (local_coords[:, 1] / ry)**2 + \
               (local_coords[:, 2] / rz)**2

        # Any point <= 1 is inside this ellipsoid
        inside_mask = (vals <= 1)
        inside_any_ellipsoid |= inside_mask  # accumulate a union

    # Now color any inside points yellow
    new_facecolors[inside_any_ellipsoid] = highlight_color_rgba

    # Update the scatter plot's facecolors
    scatter_obj.set_facecolors(new_facecolors)
    scatter_obj.set_edgecolors(new_facecolors)

    # Force a re-draw
    scatter_obj.figure.canvas.draw_idle()

def process_pca():
    global cached_ellipsoid_params, filtered_points_by_sample, current_sample, adjust_params
    from scipy.stats import chi2
    from matplotlib.widgets import RadioButtons, Button
    import numpy as np

    selected_chs = [ch for ch, var in channel_vars.items() if var.get() == 1]
    print("Selected channels:", selected_chs)
    if len(selected_chs) < 3:
        messagebox.showerror("Error", "At least 3 channels must be selected for PCA")
        return

    sample_data_dict = {}
    all_samples = [{"name": "Sample 1"}, {"name": "Sample 2"}] + samples
    for sample_entry in all_samples:
        sname = sample_entry["name"]
        if sname in file_paths and file_paths[sname]:
            print(f"\nProcessing {sname} - {len(file_paths[sname])} files")
            sample_pixels = []
            for filepath in file_paths[sname]:
                print(f"  Processing file: {filepath}")
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                    rgb_frame = arr[0]
                    spectral_frame = arr[1]
                white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
                border_size = border_sizes.get(sname, 1)
                valid_region = ~white_mask
                for _ in range(border_size):
                    valid_region = binary_erosion(valid_region)
                border_mask = (~white_mask) & (~valid_region)
                valid_pixels = ~white_mask & ~border_mask
                valid_coords = np.where(valid_pixels)
                print(f"    Valid pixels: {np.sum(valid_pixels)}")
                channels_data = []
                for ch in selected_chs:
                    if ch == "R":
                        data = rgb_frame[valid_coords][..., 0]
                    elif ch == "G":
                        data = rgb_frame[valid_coords][..., 1]
                    elif ch == "B":
                        data = rgb_frame[valid_coords][..., 2]
                    elif ch == "870":
                        data = spectral_frame[valid_coords][..., 0]
                    elif ch == "1200":
                        data = spectral_frame[valid_coords][..., 1]
                    elif ch == "1550":
                        data = spectral_frame[valid_coords][..., 2]
                    elif ch in ["L", "A", "B*"]:
                        lab = rgb2lab(rgb_frame)
                        idx = ["L", "A", "B*"].index(ch)
                        data = lab[valid_coords][..., idx]
                    if apply_normalization_var.get() == 1:
                        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
                    channels_data.append(data)
                if channels_data:
                    image_data = np.column_stack(channels_data)
                    sample_pixels.append(image_data)
            if sample_pixels:
                sample_data_dict[sname] = np.vstack(sample_pixels)
                print(f"  => {sname} total pixels: {len(sample_data_dict[sname])}")

    if not sample_data_dict:
        messagebox.showerror("Error", "No valid data for PCA")
        return

    all_data = []
    all_labels = []
    for sname, data_arr in sample_data_dict.items():
        all_data.append(data_arr)
        all_labels.extend([sname] * len(data_arr))
    combined_data = np.vstack(all_data)
    all_labels = np.array(all_labels)
    print(f"\nFinal data shape (all samples combined): {combined_data.shape}")
    
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(combined_data)

    centroid_dict = {}
    filtered_points_by_sample = {}
    for sname, data_arr in sample_data_dict.items():
        mask = (all_labels == sname)
        points_3d = pca_data[mask]
        if len(points_3d) < 2:
            continue
        km = KMeans(n_clusters=1, random_state=0)
        km.fit(points_3d)
        centroid_initial = km.cluster_centers_[0]
        dist = np.linalg.norm(points_3d - centroid_initial, axis=1)
        dist_mean = dist.mean()
        dist_std = dist.std()
        threshold = dist_mean + 2.0 * dist_std
        inlier_mask = (dist < threshold)
        if np.sum(inlier_mask) > 1:
            km_refined = KMeans(n_clusters=1, random_state=0)
            km_refined.fit(points_3d[inlier_mask])
            centroid_refined = km_refined.cluster_centers_[0]
        else:
            centroid_refined = centroid_initial
        centroid_dict[sname] = centroid_refined
        filtered_points_by_sample[sname] = points_3d

    global cached_ellipsoid_params
    cached_ellipsoid_params = {}
    for sname, points in filtered_points_by_sample.items():
        (rx_val, ry_val, rz_val), R = compute_ellipsoid(points, centroid_dict[sname])
        cached_ellipsoid_params[sname] = {"centroid": centroid_dict[sname],
                                          "rx": rx_val, "ry": ry_val, "rz": rz_val,
                                          "R": R}

    # Change subplot layout to a single panel.
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.15, right=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')
    ax.zaxis.label.set_color('blue')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='green')
    ax.tick_params(axis='z', colors='blue')

    color_scheme = {"Sample 1": "green", "Sample 2": "red"}
    additional_colors = ["blue", "purple", "orange", "yellow", "cyan", "magenta"]
    color_idx = 0
    sample_plot_dict = {}

    def create_ellipsoid_surface(rx, ry, rz, rotation_matrix, tx, ty, tz, n=50):
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0, np.pi, n)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        x *= rx; y *= ry; z *= rz
        coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        coords = coords @ rotation_matrix.T
        coords += np.array([tx, ty, tz])
        X = coords[:, 0].reshape(n, n)
        Y = coords[:, 1].reshape(n, n)
        Z = coords[:, 2].reshape(n, n)
        return X, Y, Z

    sqrt_chi2 = 1.0
    s1_centroid = centroid_dict.get("Sample 1", None)

    for sname, points_3d in filtered_points_by_sample.items():
        if sname in color_scheme:
            color = color_scheme[sname]
        else:
            color = additional_colors[color_idx % len(additional_colors)]
            color_idx += 1
        scatter_pts = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                                color=color, alpha=0.3, s=10,
                                label=f"{sname} (N={len(points_3d)})")
        ctd = centroid_dict[sname]
        dist_to_s1 = 0.0
        if s1_centroid is not None and sname != "Sample 1":
            dist_to_s1 = np.linalg.norm(ctd - s1_centroid)
        centroid_label = f"{sname} centroid (Dist→S1={dist_to_s1:.2f})"
        scatter_ctd = ax.scatter(ctd[0], ctd[1], ctd[2],
                                color=color, edgecolor='k', marker='o', s=200,
                                label=centroid_label)
        params = cached_ellipsoid_params[sname]
        rx_val = params["rx"]
        ry_val = params["ry"]
        rz_val = params["rz"]
        R = params["R"]
        X_e, Y_e, Z_e = create_ellipsoid_surface(rx_val, ry_val, rz_val, R, ctd[0], ctd[1], ctd[2])
        surf_ellipsoid = ax.plot_surface(X_e, Y_e, Z_e, color=color, alpha=0.4, edgecolor='none')
        extension_factor = 1.05
        cx, cy, cz = ctd
        line_x = ax.plot([cx - extension_factor * rx_val, cx + extension_factor * rx_val],
                        [cy, cy],
                        [cz, cz],
                        color='red', linestyle='--', linewidth=1.5)[0]
        line_y = ax.plot([cx, cx],
                        [cy - extension_factor * ry_val, cy + extension_factor * ry_val],
                        [cz, cz],
                        color='green', linestyle='--', linewidth=1.5)[0]
        line_z = ax.plot([cx, cx],
                        [cy, cy],
                        [cz - extension_factor * rz_val, cz + extension_factor * rz_val],
                        color='blue', linestyle='--', linewidth=1.5)[0]
        text_x = ax.text(cx + extension_factor * rx_val, cy, cz, "X",
                        color="red", fontsize=12, fontweight='bold')
        text_y = ax.text(cx, cy + extension_factor * ry_val, cz, "Y",
                        color="green", fontsize=12, fontweight='bold')
        text_z = ax.text(cx, cy, cz + extension_factor * rz_val, "Z",
                        color="blue", fontsize=12, fontweight='bold')
        for line in [line_x, line_y, line_z]:
            line.set_visible(False)
        for txt in [text_x, text_y, text_z]:
            txt.set_visible(False)
        sample_plot_dict[sname] = {
            "scatter_points": scatter_pts,
            "centroid_scatter": scatter_ctd,
            "ellipsoid_surface": surf_ellipsoid,
            "axes_lines": [line_x, line_y, line_z],
            "axis_labels": [text_x, text_y, text_z]
        }
    ax.legend()

    # 8) Create a radio-button selector in the right panel.
    sample_names_excluding_1 = [s for s in sample_plot_dict.keys() if s != "Sample 1"]
    radio_labels = ["All"] + sample_names_excluding_1
    rax = plt.axes([0.78, 0.3, 0.20, 0.40], facecolor=(0.9, 0.9, 0.9))
    radio = RadioButtons(rax, radio_labels, active=0)

    def on_clicked(label):
        global current_sample, adjust_params
        if "Sample 1" in sample_plot_dict:
            s1_arts = sample_plot_dict["Sample 1"]
            s1_arts["scatter_points"].set_visible(True)
            s1_arts["centroid_scatter"].set_visible(True)
            s1_arts["ellipsoid_surface"].set_visible(True)
            for line in s1_arts["axes_lines"]:
                line.set_visible(label == "Sample 1")
            for txt in s1_arts["axis_labels"]:
                txt.set_visible(label == "Sample 1")
        if label == "All":
            for sname, arts in sample_plot_dict.items():
                arts["scatter_points"].set_visible(True)
                arts["centroid_scatter"].set_visible(True)
                arts["ellipsoid_surface"].set_visible(True)
                for line in arts["axes_lines"]:
                    line.set_visible(False)
                for txt in arts["axis_labels"]:
                    txt.set_visible(False)
            current_sample = None
        else:
            for sname, arts in sample_plot_dict.items():
                if sname == "Sample 1" or sname == label:
                    arts["scatter_points"].set_visible(True)
                    arts["centroid_scatter"].set_visible(True)
                    arts["ellipsoid_surface"].set_visible(True)
                    if sname == label:
                        for line in arts["axes_lines"]:
                            line.set_visible(True)
                        for txt in arts["axis_labels"]:
                            txt.set_visible(True)
                    else:
                        for line in arts["axes_lines"]:
                            line.set_visible(False)
                        for txt in arts["axis_labels"]:
                            txt.set_visible(False)
                else:
                    arts["scatter_points"].set_visible(False)
                    arts["centroid_scatter"].set_visible(False)
                    arts["ellipsoid_surface"].set_visible(False)
                    for line in arts["axes_lines"]:
                        line.set_visible(False)
                    for txt in arts["axis_labels"]:
                        txt.set_visible(False)
            current_sample = label
            params_cache = cached_ellipsoid_params[label]
            adjust_params["rx"] = params_cache["rx"]
            adjust_params["ry"] = params_cache["ry"]
            adjust_params["rz"] = params_cache["rz"]
            adjust_params["tx"] = params_cache["centroid"][0]
            adjust_params["ty"] = params_cache["centroid"][1]
            adjust_params["tz"] = params_cache["centroid"][2]
            adjust_params["rot_x"], adjust_params["rot_y"], adjust_params["rot_z"] = matrix_to_eulerXYZ(params_cache["R"])
            for param, (minus_btn, txt, plus_btn) in adjust_buttons.items():
                txt.set_text(f"{adjust_params[param]:.2f}")
            update_sample1_colors_from_current(sample_plot_dict)
        fig.canvas.draw_idle()

    radio.on_clicked(on_clicked)

    # 9) Create an adjustment panel in the right panel below the radio buttons.
    params = [
        ("rx", 1), ("ry", 1), ("rz", 1),
        ("tx", 1), ("ty", 1), ("tz", 1),
        ("rot_x", 1), ("rot_y", 1), ("rot_z", 1)
    ]
    panel_left = 0.78
    panel_bottom = 0.05
    panel_width = 0.22
    panel_height = 0.20
    row_height = panel_height / len(params)
    adjust_buttons = {}

    def make_callback(param, d, text_obj):
        def callback(event):
            global adjust_params, current_sample
            if current_sample is None:
                return
            adjust_params[param] += d
            text_obj.set_text(f"{adjust_params[param]:.2f}")
            update_ellipsoid(current_sample, ax, sample_plot_dict)
            if current_sample != "Sample 1":
                update_sample1_colors_from_current(sample_plot_dict)
        return callback

    for i, (pname, delta) in enumerate(params):
        row_bottom = panel_bottom + panel_height - (i+1)*row_height
        label_ax = fig.add_axes([panel_left - 0.07, row_bottom, 0.06, row_height * 0.8])
        label_ax.axis("off")
        label_ax.text(0.5, 0.5, pname, ha="center", va="center", fontsize=10)
        if pname in ["rx", "tx", "rot_x"]:
            btn_color = "lightcoral"
            hover_color = "red"
        elif pname in ["ry", "ty", "rot_y"]:
            btn_color = "lightgreen"
            hover_color = "green"
        elif pname in ["rz", "tz", "rot_z"]:
            btn_color = "lightblue"
            hover_color = "blue"
        else:
            btn_color = "gray"
            hover_color = "dimgray"
        minus_ax = fig.add_axes([panel_left, row_bottom, 0.06, row_height * 0.8])
        text_ax = fig.add_axes([panel_left + 0.07, row_bottom, 0.07, row_height * 0.8])
        plus_ax  = fig.add_axes([panel_left + 0.14, row_bottom, 0.06, row_height * 0.8])
        text_ax.axis("off")
        init_val = adjust_params[pname] if adjust_params[pname] is not None else 0.0
        txt = text_ax.text(0.5, 0.5, f"{init_val:.2f}", ha="center", va="center", fontsize=10)
        minus_btn = MplButton(minus_ax, label="-", color=btn_color, hovercolor=hover_color)
        plus_btn = MplButton(plus_ax, label="+", color=btn_color, hovercolor=hover_color)
        minus_btn.on_clicked(make_callback(pname, -delta, txt))
        plus_btn.on_clicked(make_callback(pname, delta, txt))
        adjust_buttons[pname] = (minus_btn, txt, plus_btn)

    # 10) Create a Reset button below the adjustment panel.
    reset_ax = fig.add_axes([panel_left, panel_bottom - 0.07, panel_width, 0.06])
    reset_button = MplButton(reset_ax, label="Reset", color="lightgray", hovercolor="gray")
    def reset_callback(event):
        global adjust_params, current_sample
        if current_sample is None:
            return
        params_cache = cached_ellipsoid_params[current_sample]
        adjust_params["rx"] = params_cache["rx"]
        adjust_params["ry"] = params_cache["ry"]
        adjust_params["rz"] = params_cache["rz"]
        adjust_params["tx"] = params_cache["centroid"][0]
        adjust_params["ty"] = params_cache["centroid"][1]
        adjust_params["tz"] = params_cache["centroid"][2]
        adjust_params["rot_x"], adjust_params["rot_y"], adjust_params["rot_z"] = matrix_to_eulerXYZ(params_cache["R"])
        for param, (mb, txt, pb) in adjust_buttons.items():
            txt.set_text(f"{adjust_params[param]:.2f}")
        update_ellipsoid(current_sample, ax, sample_plot_dict)
        # Reset the Tkinter overlay: restore the raw image.
        if "Sample 1" in preview_labels:
            raw_photo = ImageTk.PhotoImage(Image.fromarray(preview_images["Sample 1"]))
            preview_labels["Sample 1"].configure(image=raw_photo)
            preview_labels["Sample 1"].image = raw_photo
        fig.canvas.draw_idle()
    reset_button.on_clicked(reset_callback)

    # Comment out the plot title.
    # plt.title(f"PCA of {', '.join(selected_chs)} channels\nUse the radio buttons to select a sample and adjust its ellipsoid parameters.\nPress 'Reset' to restore initial settings.")
    plt.show()


# -------------------------------------------------------------------
#Process LDA
# -------------------------------------------------------------------
def process_lda():
    """
    This function:
      - Uses the same input data (from the selected channels) as for PCA.
      - Treats each sample as its own class and uses Linear Discriminant Analysis (LDA)
        to obtain a low-dimensional representation.
      - If LDA produces fewer than 3 dimensions (since max = c-1), it pads the data with zeros so
        that each data point is represented in 3D.
      - Then, using the same logic as in your process_pca() routine, it computes the centroids,
        computes an MVEE-based ellipsoid for each sample, and sets up the same interactive
        adjustment controls and overlay updates.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import chi2
    from matplotlib.widgets import RadioButtons, Button
    import numpy as np

    # Ensure at least 3 channels are selected
    selected_chs = [ch for ch, var in channel_vars.items() if var.get() == 1]
    if len(selected_chs) < 3:
        messagebox.showerror("Error", "At least 3 channels must be selected for processing.")
        return

    # --- Step 1: Build the sample data dictionary (same as in process_pca) ---
    sample_data_dict = {}
    all_samples = [{"name": "Sample 1"}, {"name": "Sample 2"}] + samples
    for sample_entry in all_samples:
        sname = sample_entry["name"]
        if sname in file_paths and file_paths[sname]:
            print(f"\nProcessing {sname} - {len(file_paths[sname])} files")
            sample_pixels = []  # Store pixels from all images for the sample

            for filepath in file_paths[sname]:  # Loop over all images for the sample
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()

                rgb_frame = arr[0]  # First frame is RGB
                spectral_frame = arr[1]  # Second frame contains spectral channels

                # Create background mask (white pixels turn blue)
                white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)

                # Apply erosion to remove border regions
                border_size = border_sizes.get(sname, 1)
                valid_region = ~white_mask
                for _ in range(border_size):
                    valid_region = binary_erosion(valid_region)

                # Identify valid pixels (exclude both background and borders)
                border_mask = (~white_mask) & (~valid_region)
                valid_pixels = ~white_mask & ~border_mask

                # Get coordinates of valid pixels
                valid_coords = np.where(valid_pixels)

                # Extract selected channels for PCA/LDA
                channels_data = []
                for ch in selected_chs:
                    if ch == "R":
                        data = rgb_frame[valid_coords][..., 0]
                    elif ch == "G":
                        data = rgb_frame[valid_coords][..., 1]
                    elif ch == "B":
                        data = rgb_frame[valid_coords][..., 2]
                    elif ch == "870":
                        data = spectral_frame[valid_coords][..., 0]
                    elif ch == "1200":
                        data = spectral_frame[valid_coords][..., 1]
                    elif ch == "1550":
                        data = spectral_frame[valid_coords][..., 2]
                    elif ch in ["L", "A", "B*"]:
                        lab = rgb2lab(rgb_frame)
                        idx = ["L", "A", "B*"].index(ch)
                        data = lab[valid_coords][..., idx]

                    # Apply normalization if enabled
                    if apply_normalization_var.get() == 1:
                        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

                    channels_data.append(data)

                # Stack channel data for this image
                if channels_data:
                    image_data = np.column_stack(channels_data)
                    sample_pixels.append(image_data)

            # Combine all image data for this sample
            if sample_pixels:
                sample_data_dict[sname] = np.vstack(sample_pixels)
                print(f"  => {sname} total pixels: {len(sample_data_dict[sname])}")


    if not sample_data_dict:
        messagebox.showerror("Error", "No valid data for LDA")
        return

    # --- Step 2: Combine data and create labels ---
    all_data = []
    label_list = []
    for sname, data_arr in sample_data_dict.items():
        all_data.append(data_arr)
        label_list.extend([sname] * len(data_arr))
    combined_data = np.vstack(all_data)
    label_array = np.array(label_list)
    
    # Encode string labels as integers
    le = LabelEncoder()
    numeric_labels = le.fit_transform(label_array)
    print(f"[DEBUG] Process LDA - Data Shape: {combined_data.shape}, Unique Labels: {np.unique(label_array)}")

    # --- Step 3: Apply LDA ---
    # The maximum number of LDA components is (number_of_classes - 1)
    num_classes = len(le.classes_)
    n_components = min(3, num_classes - 1)
    lda = LDA(n_components=n_components)
    lda_data = lda.fit_transform(combined_data, numeric_labels)
    # If LDA returns fewer than 3 dimensions, pad with zeros.
    if lda_data.shape[1] < 3:
        padding = np.zeros((lda_data.shape[0], 3 - lda_data.shape[1]))
        lda_data = np.hstack([lda_data, padding])

    # --- Step 4: Compute centroids and filter points (same as PCA) ---
    centroid_dict = {}
    filtered_points_by_sample = {}
    for sname in sample_data_dict.keys():
        mask = (label_array == sname)
        points_3d = lda_data[mask]
        if points_3d.shape[0] < 2:
            continue
        km = KMeans(n_clusters=1, random_state=0)
        km.fit(points_3d)
        centroid_initial = km.cluster_centers_[0]
        dist = np.linalg.norm(points_3d - centroid_initial, axis=1)
        threshold = dist.mean() + 2.0 * dist.std()
        inlier_mask = (dist < threshold)
        if np.sum(inlier_mask) > 1:
            km_refined = KMeans(n_clusters=1, random_state=0)
            km_refined.fit(points_3d[inlier_mask])
            centroid_refined = km_refined.cluster_centers_[0]
        else:
            centroid_refined = centroid_initial
        centroid_dict[sname] = centroid_refined
        filtered_points_by_sample[sname] = points_3d

    global cached_ellipsoid_params
    cached_ellipsoid_params = {}
    for sname, points in filtered_points_by_sample.items():
        (rx_val, ry_val, rz_val), R = compute_ellipsoid(points, centroid_dict[sname])
        cached_ellipsoid_params[sname] = {"centroid": centroid_dict[sname],
                                          "rx": rx_val, "ry": ry_val, "rz": rz_val,
                                          "R": R}

    # --- Step 5: Plot in 3D (using the same interactive controls as in process_pca) ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.15, right=0.75)
    ax.set_xlabel("LDA Dim 1")
    ax.set_ylabel("LDA Dim 2")
    ax.set_zlabel("LDA Dim 3")
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')
    ax.zaxis.label.set_color('blue')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='green')
    ax.tick_params(axis='z', colors='blue')

    color_scheme = {"Sample 1": "green", "Sample 2": "red"}
    additional_colors = ["blue", "purple", "orange", "yellow", "cyan", "magenta"]
    color_idx = 0
    sample_plot_dict = {}

    def create_ellipsoid_surface(rx, ry, rz, rotation_matrix, tx, ty, tz, n=50):
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0, np.pi, n)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        x *= rx; y *= ry; z *= rz
        coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        coords = coords @ rotation_matrix.T
        coords += np.array([tx, ty, tz])
        X = coords[:, 0].reshape(n, n)
        Y = coords[:, 1].reshape(n, n)
        Z = coords[:, 2].reshape(n, n)
        return X, Y, Z

    for sname, points_3d in filtered_points_by_sample.items():
        if sname in color_scheme:
            color = color_scheme[sname]
        else:
            color = additional_colors[color_idx % len(additional_colors)]
            color_idx += 1
        scatter_pts = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                                 color=color, alpha=0.3, s=10,
                                 label=f"{sname} (N={points_3d.shape[0]})")
        ctd = centroid_dict[sname]
        scatter_ctd = ax.scatter(ctd[0], ctd[1], ctd[2],
                                 color=color, edgecolor='k', marker='o', s=200,
                                 label=f"{sname} centroid")
        params = cached_ellipsoid_params[sname]
        rx_val = params["rx"]
        ry_val = params["ry"]
        rz_val = params["rz"]
        R = params["R"]
        X_e, Y_e, Z_e = create_ellipsoid_surface(rx_val, ry_val, rz_val, R, ctd[0], ctd[1], ctd[2])
        surf_ellipsoid = ax.plot_surface(X_e, Y_e, Z_e, color=color, alpha=0.4, edgecolor='none')
        extension_factor = 1.5
        cx, cy, cz = ctd
        line_x = ax.plot([cx - extension_factor * rx_val, cx + extension_factor * rx_val],
                         [cy, cy],
                         [cz, cz],
                         color='red', linestyle='--', linewidth=1.5)[0]
        line_y = ax.plot([cx, cx],
                         [cy - extension_factor * ry_val, cy + extension_factor * ry_val],
                         [cz, cz],
                         color='green', linestyle='--', linewidth=1.5)[0]
        line_z = ax.plot([cx, cx],
                         [cy, cy],
                         [cz - extension_factor * rz_val, cz + extension_factor * rz_val],
                         color='blue', linestyle='--', linewidth=1.5)[0]
        # Create axis label texts:
        text_x = ax.text(cx + extension_factor * rx_val, cy, cz, "X",
                         color="red", fontsize=12, fontweight='bold')
        text_y = ax.text(cx, cy + extension_factor * ry_val, cz, "Y",
                         color="green", fontsize=12, fontweight='bold')
        text_z = ax.text(cx, cy, cz + extension_factor * rz_val, "Z",
                         color="blue", fontsize=12, fontweight='bold')
        for line in [line_x, line_y, line_z]:
            line.set_visible(False)
        for txt in [text_x, text_y, text_z]:
            txt.set_visible(False)
        sample_plot_dict[sname] = {
            "scatter_points": scatter_pts,
            "centroid_scatter": scatter_ctd,
            "ellipsoid_surface": surf_ellipsoid,
            "axes_lines": [line_x, line_y, line_z],
            "axis_labels": [text_x, text_y, text_z]
        }
    ax.legend()

    add_distance_text_to_figure(fig,ax)

    # --- Create radio buttons (right panel) to toggle axis lines ---
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.transforms as mtransforms

    # Create radio button labels (Add "All" at the top)
    sample_names_excluding_1 = [s for s in sample_plot_dict.keys() if s != "Sample 1"]
    radio_labels = ["All"] + sample_names_excluding_1

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # Define positions for each radio button label (adjust values for fine-tuning)
    base_position = 0.8
    offset_step = 0.20
    button_positions = {"All": base_position}  # Start with "All"
    for i, label in enumerate(radio_labels[1:]):  # Skip "All", process other samples
        button_positions[label] = base_position - ((i + 1) * offset_step)  # Offset each sample



    # Create the radio button UI
    rax = plt.axes([0.78, 0.3, 0.20, 0.40], facecolor=(0.9, 0.9, 0.9))
    radio = RadioButtons(rax, radio_labels, active=0)

    # Attach images next to radio button labels
    for label in radio_labels:
        if label == "All":
            sample_name = "Sample 1"  # Align "All" with Sample 1's image
        else:
            sample_name = label

        if sample_name in preview_images:
            img = preview_images[sample_name]
            img_resized = Image.fromarray(img).resize((50, 50))  # Adjust size if needed
            img_array = np.array(img_resized)

            # Convert image to OffsetImage
            imagebox = OffsetImage(img_array, zoom=0.6)

            # Create an AnnotationBbox to place next to the radio button text
            ab = AnnotationBbox(imagebox, (0.73, button_positions[label]),  # Modify 0.73 (horizontal) and button_positions[label] (vertical)
                                frameon=False, xycoords=rax.transAxes, box_alignment=(1, 0.5))

            rax.add_artist(ab)  # Add the image next to the radio button

    plt.draw()




    def on_clicked(label):
        global current_sample, adjust_params

        if "Sample 1" in sample_plot_dict:
            s1_arts = sample_plot_dict["Sample 1"]
            s1_arts["scatter_points"].set_visible(True)
            s1_arts["centroid_scatter"].set_visible(True)
            s1_arts["ellipsoid_surface"].set_visible(True)

            for line in s1_arts["axes_lines"]:
                line.set_visible(label == "Sample 1")
            for txt in s1_arts["axis_labels"]:
                txt.set_visible(label == "Sample 1")

        if label == "All":
            for sname, arts in sample_plot_dict.items():
                arts["scatter_points"].set_visible(True)
                arts["centroid_scatter"].set_visible(True)
                arts["ellipsoid_surface"].set_visible(True)
                for line in arts["axes_lines"]:
                    line.set_visible(False)
                for txt in arts["axis_labels"]:
                    txt.set_visible(False)
            current_sample = None
        else:
            for sname, arts in sample_plot_dict.items():
                if sname == "Sample 1" or sname == label:
                    arts["scatter_points"].set_visible(True)
                    arts["centroid_scatter"].set_visible(True)
                    arts["ellipsoid_surface"].set_visible(True)
                    if sname == label:
                        for line in arts["axes_lines"]:
                            line.set_visible(True)
                        for txt in arts["axis_labels"]:
                            txt.set_visible(True)
                    else:
                        for line in arts["axes_lines"]:
                            line.set_visible(False)
                        for txt in arts["axis_labels"]:
                            txt.set_visible(False)
                else:
                    arts["scatter_points"].set_visible(False)
                    arts["centroid_scatter"].set_visible(False)
                    arts["ellipsoid_surface"].set_visible(False)
                    for line in arts["axes_lines"]:
                        line.set_visible(False)
                    for txt in arts["axis_labels"]:
                        txt.set_visible(False)

            current_sample = label
            params_cache = cached_ellipsoid_params[label]
            adjust_params["rx"] = params_cache["rx"]
            adjust_params["ry"] = params_cache["ry"]
            adjust_params["rz"] = params_cache["rz"]
            adjust_params["tx"] = params_cache["centroid"][0]
            adjust_params["ty"] = params_cache["centroid"][1]
            adjust_params["tz"] = params_cache["centroid"][2]
            adjust_params["rot_x"], adjust_params["rot_y"], adjust_params["rot_z"] = matrix_to_eulerXYZ(params_cache["R"])

            for param, (minus_btn, txt, plus_btn) in adjust_buttons.items():
                txt.set_text(f"{adjust_params[param]:.2f}")

            # **Call the updated function to recolor Sample 1 points**
            update_sample1_colors_from_current_lda(sample_plot_dict, label)

        fig.canvas.draw_idle()


    radio.on_clicked(on_clicked)

    # --- Create adjustment panel below the radio buttons ---
    params = [
        ("rx", .3), ("ry", .3), ("rz", .3),
        ("tx", .3), ("ty", .3), ("tz", .3),
        ("rot_x", 3), ("rot_y", 3), ("rot_z", 3)
    ]
    panel_left = 0.78
    panel_bottom = 0.05
    panel_width = 0.22
    panel_height = 0.20
    row_height = panel_height / len(params)
    adjust_buttons = {}

    def make_callback(param, d, text_obj):
        def callback(event):
            global adjust_params, current_sample
            if current_sample is None:
                return
            adjust_params[param] += d
            text_obj.set_text(f"{adjust_params[param]:.2f}")
            update_ellipsoid(current_sample, ax, sample_plot_dict)
            if current_sample != "Sample 1":
                update_sample1_colors_from_current(sample_plot_dict)
        return callback

    for i, (pname, delta) in enumerate(params):
        row_bottom = panel_bottom + panel_height - (i+1)*row_height
        label_ax = fig.add_axes([panel_left - 0.07, row_bottom, 0.06, row_height * 0.8])
        label_ax.axis("off")
        label_ax.text(0.5, 0.5, pname, ha="center", va="center", fontsize=10)
        if pname in ["rx", "tx", "rot_x"]:
            btn_color = "lightcoral"
            hover_color = "red"
        elif pname in ["ry", "ty", "rot_y"]:
            btn_color = "lightgreen"
            hover_color = "green"
        elif pname in ["rz", "tz", "rot_z"]:
            btn_color = "lightblue"
            hover_color = "blue"
        else:
            btn_color = "gray"
            hover_color = "dimgray"
        minus_ax = fig.add_axes([panel_left, row_bottom, 0.06, row_height * 0.8])
        text_ax = fig.add_axes([panel_left + 0.07, row_bottom, 0.07, row_height * 0.8])
        plus_ax  = fig.add_axes([panel_left + 0.14, row_bottom, 0.06, row_height * 0.8])
        text_ax.axis("off")
        init_val = adjust_params[pname] if adjust_params[pname] is not None else 0.0
        txt = text_ax.text(0.5, 0.5, f"{init_val:.2f}", ha="center", va="center", fontsize=10)
        minus_btn = MplButton(minus_ax, label="-", color=btn_color, hovercolor=hover_color)
        plus_btn = MplButton(plus_ax, label="+", color=btn_color, hovercolor=hover_color)
        minus_btn.on_clicked(make_callback(pname, -delta, txt))
        plus_btn.on_clicked(make_callback(pname, delta, txt))
        adjust_buttons[pname] = (minus_btn, txt, plus_btn)

    # --- Create a Reset button below the adjustment panel.
    reset_ax = fig.add_axes([panel_left, panel_bottom - 0.07, panel_width, 0.06])
    reset_button = MplButton(reset_ax, label="Reset", color="lightgray", hovercolor="gray")
    def reset_callback(event):
        global adjust_params, current_sample
        if current_sample is None:
            return
        params_cache = cached_ellipsoid_params[current_sample]
        adjust_params["rx"] = params_cache["rx"]
        adjust_params["ry"] = params_cache["ry"]
        adjust_params["rz"] = params_cache["rz"]
        adjust_params["tx"] = params_cache["centroid"][0]
        adjust_params["ty"] = params_cache["centroid"][1]
        adjust_params["tz"] = params_cache["centroid"][2]
        adjust_params["rot_x"], adjust_params["rot_y"], adjust_params["rot_z"] = matrix_to_eulerXYZ(params_cache["R"])
        for param, (mb, txt, pb) in adjust_buttons.items():
            txt.set_text(f"{adjust_params[param]:.2f}")
        update_ellipsoid(current_sample, ax, sample_plot_dict)
        if "Sample 1" in preview_labels:
            raw_photo = ImageTk.PhotoImage(Image.fromarray(preview_images["Sample 1"]))
            preview_labels["Sample 1"].configure(image=raw_photo)
            preview_labels["Sample 1"].image = raw_photo
        fig.canvas.draw_idle()
    reset_button.on_clicked(reset_callback)

    # 11) Create a "Sensitivity" panel BELOW the existing parameter panel
    #     We'll place it just below 'reset_ax', reusing panel_left, etc.

    sensitivity_value = adjust_params["sensitivity"]  # default = 50.0

    # Position this row slightly below the reset button
    # so if reset was at panel_bottom - 0.07, let's do -0.15 here
    sens_panel_bottom = .01

    # We'll reuse 'row_height', but make these rectangles bigger (2x the size).
    large_height = row_height * 1.6  # about 1.6 times taller
    large_width_btn = 0.09           # a bit wider than the standard 0.06

    # Label: "Sensitivity"
    label_sens_ax = fig.add_axes([panel_left-.5 - 0.07, sens_panel_bottom, 0.06, large_height])
    label_sens_ax.axis("off")
    label_sens_ax.text(0.5, 0.5, "Sensitivity", ha="center", va="center",
                    fontsize=12, fontweight='bold')

    # Create the minus, text, plus controls
    minus_sens_ax = fig.add_axes([panel_left-.5, sens_panel_bottom, large_width_btn, large_height])
    text_sens_ax  = fig.add_axes([panel_left-.5 + 0.1, sens_panel_bottom, 0.06, large_height])
    plus_sens_ax  = fig.add_axes([panel_left-.5 + 0.17, sens_panel_bottom, large_width_btn, large_height])

    # Hide spines on the text axes
    text_sens_ax.axis("off")

    # Display "50" in bigger text
    txt_sens = text_sens_ax.text(0.5, 0.5, f"{sensitivity_value:.0f}",
                                ha="center", va="center", fontsize=14, fontweight='bold')

    # Create the bigger MplButton objects
    minus_sens_btn = MplButton(minus_sens_ax, label="-", color="lightgray", hovercolor="gray")
    plus_sens_btn  = MplButton(plus_sens_ax, label="+", color="lightgray", hovercolor="gray")


    def make_sensitivity_callback(delta):
        def callback(event):
            global adjust_params, current_sample
            if current_sample is None:
                return
            # Adjust rx, ry, rz all at once
            adjust_params["rx"] += delta
            adjust_params["ry"] += delta
            adjust_params["rz"] += delta

            # Redraw the ellipsoid to see the immediate size change
            update_ellipsoid(current_sample, ax, sample_plot_dict)
        return callback

    minus_sens_btn.on_clicked(make_sensitivity_callback(-0.1))
    plus_sens_btn.on_clicked(make_sensitivity_callback(+0.1))





    
    plt.show()

def get_filtered_data():
    """Load TIFF files or PNG files for Sample 1 and Sample 2 while filtering only selected channels."""
    try:
        if "Sample 1" not in file_paths or "Sample 2" not in file_paths:
            raise ValueError("Sample 1 or Sample 2 file paths are missing")

        # Ensure at least 3 channels are selected
        selected_chs = [ch for ch, var in channel_vars.items() if var.get() == 1]
        if len(selected_chs) < 3:
            raise ValueError("At least 3 channels must be selected for processing.")

        def load_and_filter_file(sample_name):
            """Loads image files (TIFF, PNG, BMP) and extracts only the selected channels."""
            if not file_paths[sample_name]:
                return None
                
            all_features = []
            
            for filepath in file_paths[sample_name]:
                try:
                    # Load image based on file type
                    if filepath.lower().endswith(('.tif', '.tiff')):
                        with tifffile.TiffFile(filepath) as tf:
                            arr = tf.asarray()
                            # For TIFF files, assume structure with RGB and maybe spectral channels
                            rgb_frame = arr[0]  # First frame is RGB
                            spectral_frame = arr[1] if len(arr) > 1 else None
                    elif filepath.lower().endswith(('.png', '.bmp')):
                        # For PNG/BMP, we only have the RGB data
                        img = Image.open(filepath).convert("RGB")
                        rgb_frame = np.array(img)
                        # Create a dummy spectral frame if needed (all zeros)
                        if any(ch in ["870", "1200", "1550"] for ch in selected_chs):
                            spectral_frame = np.zeros((*rgb_frame.shape[0:2], 3), dtype=np.uint8)
                        else:
                            spectral_frame = None
                    else:
                        continue
                    
                    # Filter background/border
                    white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
                    border_size = border_sizes.get(sample_name, 1)
                    valid_region = ~white_mask
                    for _ in range(border_size):
                        valid_region = binary_erosion(valid_region)
                    border_mask = (~white_mask) & (~valid_region)
                    valid_pixels = ~white_mask & ~border_mask
                    
                    # Skip if no valid pixels
                    if np.sum(valid_pixels) == 0:
                        continue
                    
                    # Get valid pixel coordinates
                    valid_coords = np.where(valid_pixels)
                    
                    # Extract selected channels
                    features = []
                    for ch in selected_chs:
                        if ch == "R":
                            data = rgb_frame[valid_coords][..., 0]
                        elif ch == "G":
                            data = rgb_frame[valid_coords][..., 1]
                        elif ch == "B":
                            data = rgb_frame[valid_coords][..., 2]
                        elif ch == "870" and spectral_frame is not None:
                            data = spectral_frame[valid_coords][..., 0]
                        elif ch == "1200" and spectral_frame is not None:
                            data = spectral_frame[valid_coords][..., 1]
                        elif ch == "1550" and spectral_frame is not None:
                            data = spectral_frame[valid_coords][..., 2]
                        elif ch in ["L", "A", "B*"]:
                            lab = rgb2lab(rgb_frame)
                            idx = ["L", "A", "B*"].index(ch)
                            data = lab[valid_coords][..., idx]
                        else:
                            # Skip channels that aren't available
                            continue
                        
                        # Apply optional normalization
                        if apply_normalization_var.get() == 1:
                            data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
                        
                        features.append(data)
                    
                    if len(features) == len(selected_chs):  # Only add if all channels were available
                        # Stack channel data for this image's valid pixels
                        image_features = np.column_stack(features)
                        all_features.append(image_features)
                except Exception as e:
                    print(f"Error processing {filepath}: {str(e)}")
            
            # Combine all features across all images
            if all_features:
                return np.vstack(all_features)
            return None

        # Load and filter both samples
        sample1_filtered = load_and_filter_file("Sample 1")
        sample2_filtered = load_and_filter_file("Sample 2")
        
        # Check if we got valid data for both samples
        if sample1_filtered is None or sample2_filtered is None:
            raise ValueError("Failed to extract valid features from one or both samples")

        # Ensure feature dimensions match
        if sample1_filtered.shape[1] != sample2_filtered.shape[1]:
            min_features = min(sample1_filtered.shape[1], sample2_filtered.shape[1])
            sample1_filtered = sample1_filtered[:, :min_features]
            sample2_filtered = sample2_filtered[:, :min_features]
            print(f"Feature mismatch detected! Adjusted to {min_features} features.")

        return sample1_filtered, sample2_filtered

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        messagebox.showerror("Error", f"Could not load data: {str(e)}")
        return None, None

def analyze_dimensionality_methods():
    """Perform PCA, LDA, and other dimensionality methods using Sample 1 and Sample 2 data."""
    print("Starting dimensionality methods analysis...")
    
    sample1, sample2 = get_filtered_data()  # Get data using the updated function

    if sample1 is None or sample2 is None:
        messagebox.showerror("Error", "Could not retrieve data for analysis.")
        return

    if sample1.shape[1] != sample2.shape[1]:
        messagebox.showerror("Error", f"Feature mismatch: Sample 1 has {sample1.shape[1]} columns, Sample 2 has {sample2.shape[1]} columns")
        return

    # Stack data
    data = np.vstack([sample1, sample2])
    labels = np.concatenate([np.zeros(len(sample1)), np.ones(len(sample2))])

    print(f"Data loaded successfully: {len(sample1)} points for Sample 1, {len(sample2)} points for Sample 2")
    print(f"Feature dimensionality: {sample1.shape[1]}")

    # Define dimensionality reduction methods
    methods = {
        "PCA (2D)": PCA(n_components=2),
        "PCA (3D)": PCA(n_components=3),
        "LDA": LinearDiscriminantAnalysis(n_components=1),
        "t-SNE": TSNE(n_components=2, random_state=42),
        "UMAP": umap.UMAP(n_components=2, random_state=42),
        "Kernel PCA": KernelPCA(n_components=2, kernel='rbf'),
        "Isomap": Isomap(n_components=2),
        "Locally Linear Embedding": LocallyLinearEmbedding(n_components=2, eigen_solver="dense") 
    }

    results = {}
    for method_name, method in methods.items():
        print(f"Processing {method_name}...")
        try:
            if method_name == "LDA":
                reduced_data = method.fit_transform(data, labels)
            else:
                reduced_data = method.fit_transform(data)

            # Compute distance metrics
            reduced_sample1 = reduced_data[:len(sample1)]
            reduced_sample2 = reduced_data[len(sample1):]

            mean1 = np.mean(reduced_sample1, axis=0)
            mean2 = np.mean(reduced_sample2, axis=0)
            
            # Handle potential low-dimensionality issues
            if reduced_data.shape[1] == 1:
                # For 1D data, use variance instead of covariance matrix
                var1 = np.var(reduced_sample1) + 1e-6
                var2 = np.var(reduced_sample2) + 1e-6
                pooled_var = (var1 + var2) / 2
                mahal_dist = np.abs(mean1[0] - mean2[0]) / np.sqrt(pooled_var)
            else:
                # For multi-dimensional data
                cov1 = np.atleast_2d(np.cov(reduced_sample1, rowvar=False)) + np.eye(reduced_data.shape[1]) * 1e-6
                cov2 = np.atleast_2d(np.cov(reduced_sample2, rowvar=False)) + np.eye(reduced_data.shape[1]) * 1e-6
                pooled_cov = (cov1 + cov2) / 2
                mahal_dist = mahalanobis(mean1, mean2, np.linalg.pinv(pooled_cov))

            silhouette = silhouette_score(reduced_data, labels)

            results[method_name] = {
                "Mahalanobis Distance": mahal_dist,
                "Silhouette Score": silhouette
            }
            print(f"  Results: Mahalanobis={mahal_dist:.3f}, Silhouette={silhouette:.3f}")
        except Exception as e:
            print(f"Error processing {method_name}: {str(e)}")


def analyze_dimensionality_methods():
    """Perform PCA, LDA, and other dimensionality methods using Sample 1 and Sample 2 data."""
    sample1, sample2 = get_filtered_data()  # Ensure this fetches the correct data

    if sample1 is None or sample2 is None:
        messagebox.showerror("Error", "Could not retrieve data for analysis.")
        return

    if sample1.shape[1] != sample2.shape[1]:
        messagebox.showerror("Error", f"Feature mismatch: Sample 1 has {sample1.shape[1]} columns, Sample 2 has {sample2.shape[1]} columns")
        return

    # Stack data
    data = np.vstack([sample1, sample2])
    labels = np.concatenate([np.zeros(len(sample1)), np.ones(len(sample2))])

    # Define dimensionality reduction methods
    methods = {
        "PCA (2D)": PCA(n_components=2),
        "PCA (3D)": PCA(n_components=3),
        "LDA": LinearDiscriminantAnalysis(n_components=1),
        "t-SNE": TSNE(n_components=2, random_state=42),
        "UMAP": umap.UMAP(n_components=2, random_state=42),
        "Kernel PCA": KernelPCA(n_components=2, kernel='rbf'),
        "Isomap": Isomap(n_components=2),
        "Locally Linear Embedding": LocallyLinearEmbedding(n_components=2, eigen_solver="dense") 
    }


    results = {}
    for method_name, method in methods.items():
        if method_name == "LDA":
            reduced_data = method.fit_transform(data, labels)
        else:
            reduced_data = method.fit_transform(data)

        # Compute distance metrics
        reduced_sample1 = reduced_data[:len(sample1)]
        reduced_sample2 = reduced_data[len(sample1):]

        mean1 = np.mean(reduced_sample1, axis=0)
        mean2 = np.mean(reduced_sample2, axis=0)
        cov1 = np.atleast_2d(np.cov(reduced_sample1, rowvar=False))
        cov2 = np.atleast_2d(np.cov(reduced_sample2, rowvar=False))
        pooled_cov = (cov1 + cov2) / 2

        # Ensure valid Mahalanobis computation
        if pooled_cov.shape[0] == 1 and pooled_cov.shape[1] == 1:
            pooled_cov = pooled_cov + np.eye(1) * 1e-6  # Regularization

        mahal_dist = mahalanobis(mean1, mean2, np.linalg.pinv(pooled_cov))
        silhouette = silhouette_score(reduced_data, labels)

        results[method_name] = {
            "Mahalanobis Distance": mahal_dist,
            "Silhouette Score": silhouette
        }

    # Print results for debugging
    for method, metrics in results.items():
        print(f"{method}: Mahalanobis Distance = {metrics['Mahalanobis Distance']:.3f}, Silhouette Score = {metrics['Silhouette Score']:.3f}")

    # Call functions to display results inside Tkinter
    show_results_in_tkinter(results, root)  # Show the chart
    show_best_method_message(results)  # Show messagebox alert

#------------------------------------------------------------------------------------
#Enhanced LDA Process
#------------------------------------------------------------------------------------


def extract_channel_data(channel, rgb_frame, spectral_frame, sample_name):
    """
    Extracts the appropriate data channel from the image.
    - Applies the same background and border masking as process_lda().
    - Returns only valid pixel data.
    """

    # Background detection (Blue Background)
    white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)

    # Border removal using erosion (Red Border)
    border_size = border_sizes.get(sample_name, 1)
    valid_region = ~white_mask
    for _ in range(border_size):
        valid_region = binary_erosion(valid_region)

    # Define border mask and final valid pixels
    border_mask = (~white_mask) & (~valid_region)
    valid_pixels = ~white_mask & ~border_mask  # Exclude both white and red border pixels

    # Get valid pixel coordinates
    valid_coords = np.where(valid_pixels)

    # Extract channel data only from valid pixels
    if channel == "R":
        return rgb_frame[valid_coords][..., 0]
    elif channel == "G":
        return rgb_frame[valid_coords][..., 1]
    elif channel == "B":
        return rgb_frame[valid_coords][..., 2]
    elif channel == "870":
        return spectral_frame[valid_coords][..., 0]
    elif channel == "1200":
        return spectral_frame[valid_coords][..., 1]
    elif channel == "1550":
        return spectral_frame[valid_coords][..., 2]
    
    # Convert to LAB color space if needed
    elif channel in ["L", "A", "B*"]:
        from skimage.color import rgb2lab
        lab = rgb2lab(rgb_frame)
        idx = ["L", "A", "B*"].index(channel)
        return lab[valid_coords][..., idx]
    
    # If channel isn't recognized, return an empty array
    return np.array([])



def get_all_sample_data():
    """Load and prepare all sample data for LDA."""
    # Placeholder function: Replace with actual image feature extraction
    try:
        sample_data = np.random.rand(100, 6)  # Replace with real extracted features
        labels = np.array([0] * 50 + [1] * 25 + [2] * 25)  # Replace with real sample labels
        return sample_data, labels
    except Exception as e:
        print("Error loading sample data:", e)
        return None, None

from sklearn.preprocessing import LabelEncoder

from scipy.spatial.distance import mahalanobis, euclidean
from sklearn.metrics import silhouette_score

def plot_lda_results(lda_data, labels, title):
    """Display LDA results in a Tkinter-embedded 3D plot with separation metrics."""
    
    plot_window = Toplevel(root)
    plot_window.title(title)
    plot_window.geometry("800x600")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Assign colors to each sample
    unique_labels = np.unique(labels)
    color_map = plt.cm.get_cmap("tab10", len(unique_labels))
    label_colors = {label: color_map(i) for i, label in enumerate(unique_labels)}

    # Compute Centroids for each class
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = np.mean(lda_data[mask], axis=0)

    # Scatter plot
    for label in unique_labels:
        mask = labels == label
        scatter = ax.scatter(
            lda_data[mask, 0], lda_data[mask, 1], lda_data[mask, 2], 
            color=label_colors[label], label=f"{label} (N={np.sum(mask)})", alpha=0.6
        )

        # Plot centroid
        centroid = centroids[label]
        ax.scatter(*centroid, color=label_colors[label], edgecolor='black', s=100, marker='o')

    # Compute Mahalanobis Distance & Euclidean Distance
    mahal_dists = []
    euclidean_dists = []
    label_list = list(centroids.keys())
    pooled_cov = np.cov(lda_data, rowvar=False)
    
    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            c1, c2 = centroids[label_list[i]], centroids[label_list[j]]

            # Mahalanobis Distance
            try:
                inv_cov = np.linalg.pinv(pooled_cov)  # Inverted covariance matrix
                mahal_dist = mahalanobis(c1, c2, inv_cov)
                mahal_dists.append(mahal_dist)
            except:
                mahal_dist = 0  # If computation fails, assume no separation

            # Euclidean Distance
            euclidean_dist = euclidean(c1, c2)
            euclidean_dists.append(euclidean_dist)

    # Compute Silhouette Score
    silhouette = silhouette_score(lda_data, labels)

    # Display the metrics in the plot
    avg_mahal_dist = np.mean(mahal_dists) if mahal_dists else 0
    avg_euclidean_dist = np.mean(euclidean_dists) if euclidean_dists else 0
    ax.text2D(0.05, 0.95, f"Avg Mahalanobis: {avg_mahal_dist:.2f}", transform=ax.transAxes, fontsize=10, color='black')
    ax.text2D(0.05, 0.90, f"Avg Euclidean: {avg_euclidean_dist:.2f}", transform=ax.transAxes, fontsize=10, color='black')
    ax.text2D(0.05, 0.85, f"Silhouette Score: {silhouette:.2f}", transform=ax.transAxes, fontsize=10, color='black')

    # Plot settings
    ax.set_title(title)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3")
    ax.legend()
    
    # Embed in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
def update_sample1_colors_against_all_ellipsoids(sample_plot_dict):
    """
    Checks every point from 'Sample 1' to see if it falls inside any other sample's ellipsoid.
    If inside at least one, color that point yellow. Otherwise, revert to green.
    """
    global filtered_points_by_sample, cached_ellipsoid_params

    # If Sample 1 doesn't exist or hasn't been computed, do nothing
    if "Sample 1" not in filtered_points_by_sample:
        return

    # Retrieve Sample 1's 3D points and the scatter object
    s1_points = filtered_points_by_sample["Sample 1"]
    scatter_obj = sample_plot_dict["Sample 1"]["scatter_points"]

    # Prepare default (green) or any color scheme you prefer
    default_color_rgba = np.array(to_rgba("green"))
    highlight_color_rgba = np.array(to_rgba("yellow"))

    n_points = len(s1_points)
    new_facecolors = np.tile(default_color_rgba, (n_points, 1))

    # We'll build a union "inside_any_ellipsoid" mask
    inside_any_ellipsoid = np.zeros(n_points, dtype=bool)

    # Loop over all ellipsoids EXCEPT Sample 1's own ellipsoid
    for other_sample_name, ell_params in cached_ellipsoid_params.items():
        if other_sample_name == "Sample 1":
            continue  # Skip checking Sample 1's own ellipsoid if you only care about "other" ellipsoids

        # Get center/radii/rotation
        c = ell_params["centroid"]
        rx = ell_params["rx"]
        ry = ell_params["ry"]
        rz = ell_params["rz"]
        R = ell_params["R"]  # 3x3 rotation

        # Transform Sample1 points into that ellipsoid's local coords
        local_coords = (s1_points - c) @ R.T
        vals = (local_coords[:, 0] / rx)**2 + \
               (local_coords[:, 1] / ry)**2 + \
               (local_coords[:, 2] / rz)**2

        # Any point <= 1 is inside this ellipsoid
        inside_mask = (vals <= 1)
        inside_any_ellipsoid |= inside_mask  # accumulate a union

    # Now color any inside points yellow
    new_facecolors[inside_any_ellipsoid] = highlight_color_rgba

    # Update the scatter plot’s facecolors
    scatter_obj.set_facecolors(new_facecolors)
    scatter_obj.set_edgecolors(new_facecolors)

    # Force a re-draw
    scatter_obj.figure.canvas.draw_idle()

def process_pca_lda():
    """
    1) Gather data from each sample (like process_pca/process_lda).
    2) Perform PCA (n_components=4).
    3) Perform LDA (n_components=3) on the PCA outputs.
    4) Plot the 3D results with ellipsoids and interactive controls.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Button as MplButton

    # --- Step 1: Collect data from selected channels, just like process_pca/process_lda ---
    selected_chs = [ch for ch, var in channel_vars.items() if var.get() == 1]
    if len(selected_chs) < 3:
        messagebox.showerror("Error", "At least 3 channels must be selected for PCA→LDA.")
        return

    # Build a dict: sample_name -> [all valid pixel rows from selected channels]
    sample_data_dict = {}
    all_samples_list = [{"name": "Sample 1"}, {"name": "Sample 2"}] + samples
    for sample_entry in all_samples_list:
        sname = sample_entry["name"]
        if sname in file_paths and file_paths[sname]:
            sample_pixels = []
            for filepath in file_paths[sname]:
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                # Possibly the same logic as in process_lda, extracting RGB vs. spectral frames, removing backgrounds:
                rgb_frame = arr[0]
                spectral_frame = arr[1]
                white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
                border_size = border_sizes.get(sname, 1)
                valid_region = ~white_mask
                for _ in range(border_size):
                    valid_region = binary_erosion(valid_region)
                border_mask = (~white_mask) & (~valid_region)
                valid_pixels = ~white_mask & ~border_mask
                valid_coords = np.where(valid_pixels)

                # Extract only chosen channels
                channels_data = []
                for ch in selected_chs:
                    if ch == "R":
                        data = rgb_frame[valid_coords][..., 0]
                    elif ch == "G":
                        data = rgb_frame[valid_coords][..., 1]
                    elif ch == "B":
                        data = rgb_frame[valid_coords][..., 2]
                    elif ch == "870":
                        data = spectral_frame[valid_coords][..., 0]
                    elif ch == "1200":
                        data = spectral_frame[valid_coords][..., 1]
                    elif ch == "1550":
                        data = spectral_frame[valid_coords][..., 2]
                    elif ch in ["L", "A", "B*"]:
                        lab = rgb2lab(rgb_frame)
                        idx = ["L", "A", "B*"].index(ch)
                        data = lab[valid_coords][..., idx]
                    # Normalize if user checked the box
                    if apply_normalization_var.get() == 1:
                        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
                    channels_data.append(data)

                if channels_data:
                    image_data = np.column_stack(channels_data)
                    sample_pixels.append(image_data)
            if sample_pixels:
                sample_data_dict[sname] = np.vstack(sample_pixels)

    if not sample_data_dict:
        messagebox.showerror("Error", "No valid sample data for PCA→LDA.")
        return

    # Combine all data
    all_data = []
    label_list = []
    for sname, data_arr in sample_data_dict.items():
        all_data.append(data_arr)
        label_list.extend([sname] * len(data_arr))
    combined_data = np.vstack(all_data)
    label_array = np.array(label_list)

    # --- Step 2: PCA (4 components) ---
    pca_4 = PCA(n_components=4)
    X_pca_4 = pca_4.fit_transform(combined_data)

    # --- Step 3: LDA (3 components) on the PCA outputs ---
    le = LabelEncoder()
    numeric_labels = le.fit_transform(label_array)
    num_classes = len(le.classes_)
    # LDA can only produce up to (num_classes-1) dims
    n_components_lda = min(3, num_classes - 1)
    lda_3 = LDA(n_components=n_components_lda)
    X_lda_3 = lda_3.fit_transform(X_pca_4, numeric_labels)

    # If we ended up with fewer than 3 dims (because classes < 4), we can pad:
    if X_lda_3.shape[1] < 3:
        needed = 3 - X_lda_3.shape[1]
        X_lda_3 = np.hstack([X_lda_3, np.zeros((X_lda_3.shape[0], needed))])

    # --- Step 4: Compute centroids, filter outliers, compute ellipsoids ---
    centroid_dict = {}
    global filtered_points_by_sample, cached_ellipsoid_params
    filtered_points_by_sample = {}
    for sname in sample_data_dict.keys():
        mask = (label_array == sname)
        points_3d = X_lda_3[mask]
        if len(points_3d) < 2:
            continue
        km = KMeans(n_clusters=1, random_state=0).fit(points_3d)
        c_init = km.cluster_centers_[0]
        dist = np.linalg.norm(points_3d - c_init, axis=1)
        thresh = dist.mean() + 2.0*dist.std()
        inlier_mask = (dist < thresh)
        if np.sum(inlier_mask) > 1:
            # refine centroid
            c_refined = KMeans(n_clusters=1, random_state=0).fit(points_3d[inlier_mask]).cluster_centers_[0]
        else:
            c_refined = c_init
        centroid_dict[sname] = c_refined
        filtered_points_by_sample[sname] = points_3d

    cached_ellipsoid_params = {}
    for sname, pts in filtered_points_by_sample.items():
        (rx_val, ry_val, rz_val), R = compute_ellipsoid(pts, centroid_dict[sname])
        cached_ellipsoid_params[sname] = {
            "centroid": centroid_dict[sname],
            "rx": rx_val, "ry": ry_val, "rz": rz_val,
            "R": R
        }

    # --- Step 5: Plot in 3D (similar to process_lda or process_pca) ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.15, right=0.75)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3")
    ax.set_title("PCA(4) → LDA(3) Projection")

    color_scheme = {"Sample 1": "green", "Sample 2": "red"}
    additional_colors = ["blue", "purple", "orange", "yellow", "cyan", "magenta"]
    color_idx = 0
    sample_plot_dict = {}

    # Helper for creating ellipsoid surfaces
    def create_ellipsoid_surface(rx, ry, rz, rotation_matrix, tx, ty, tz, n=50):
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0, np.pi, n)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        x *= rx; y *= ry; z *= rz
        coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        coords = coords @ rotation_matrix.T
        coords += np.array([tx, ty, tz])
        X = coords[:, 0].reshape(n, n)
        Y = coords[:, 1].reshape(n, n)
        Z = coords[:, 2].reshape(n, n)
        return X, Y, Z

    # Scatter + ellipsoids
    for sname, pts_3d in filtered_points_by_sample.items():
        color = color_scheme[sname] if sname in color_scheme else additional_colors[color_idx % len(additional_colors)]
        color_idx += 1
        sc = ax.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2],
                        color=color, alpha=0.3, s=10, label=f"{sname} (N={len(pts_3d)})")
        ctd = centroid_dict[sname]
        sc_ctd = ax.scatter(ctd[0], ctd[1], ctd[2], color=color, edgecolor='k', s=200, marker='o',
                            label=f"{sname} centroid")
        rx = cached_ellipsoid_params[sname]["rx"]
        ry = cached_ellipsoid_params[sname]["ry"]
        rz = cached_ellipsoid_params[sname]["rz"]
        R = cached_ellipsoid_params[sname]["R"]
        X_e, Y_e, Z_e = create_ellipsoid_surface(rx, ry, rz, R, ctd[0], ctd[1], ctd[2])
        surf = ax.plot_surface(X_e, Y_e, Z_e, color=color, alpha=0.4, edgecolor='none')

        # Axis lines (hidden unless selected)
        extension_factor = 1.2
        cx, cy, cz = ctd
        lx = ax.plot([cx - extension_factor*rx, cx + extension_factor*rx],
                     [cy, cy], [cz, cz],
                     color='red', linestyle='--', linewidth=1.5)[0]
        ly = ax.plot([cx, cx],
                     [cy - extension_factor*ry, cy + extension_factor*ry],
                     [cz, cz],
                     color='green', linestyle='--', linewidth=1.5)[0]
        lz = ax.plot([cx, cx],
                     [cy, cy],
                     [cz - extension_factor*rz, cz + extension_factor*rz],
                     color='blue', linestyle='--', linewidth=1.5)[0]
        txtx = ax.text(cx + extension_factor*rx, cy, cz,  "X", color='red', fontsize=12, fontweight='bold')
        txty = ax.text(cx, cy + extension_factor*ry, cz,  "Y", color='green', fontsize=12, fontweight='bold')
        txtz = ax.text(cx, cy, cz + extension_factor*rz,  "Z", color='blue', fontsize=12, fontweight='bold')
        for ln in (lx, ly, lz):
            ln.set_visible(False)
        for ttt in (txtx, txty, txtz):
            ttt.set_visible(False)

        sample_plot_dict[sname] = {
            "scatter_points": sc,
            "centroid_scatter": sc_ctd,
            "ellipsoid_surface": surf,
            "axes_lines": [lx, ly, lz],
            "axis_labels": [txtx, txty, txtz]
        }

    ax.legend()

    # Radio button to select a sample, show/hide ellipsoid axes, etc.:
    radio_labels = ["All"] + [s for s in sample_plot_dict.keys() if s != "Sample 1"]
    rax = plt.axes([0.8, 0.3, 0.15, 0.4], facecolor=(0.9, 0.9, 0.9))
    radio = RadioButtons(rax, radio_labels, active=0)

    def on_clicked(label):
        global current_sample, adjust_params
        # Show/hide for each sample
        if label == "All":
            for sname, arts in sample_plot_dict.items():
                arts["scatter_points"].set_visible(True)
                arts["centroid_scatter"].set_visible(True)
                arts["ellipsoid_surface"].set_visible(True)
                for line in arts["axes_lines"]:
                    line.set_visible(False)
                for txt in arts["axis_labels"]:
                    txt.set_visible(False)
            current_sample = None
        else:
            for sname, arts in sample_plot_dict.items():
                if sname == "Sample 1" or sname == label:
                    arts["scatter_points"].set_visible(True)
                    arts["centroid_scatter"].set_visible(True)
                    arts["ellipsoid_surface"].set_visible(True)
                    # Make lines/labels visible if exactly the selected sample
                    show_lines = (sname == label)
                    for line in arts["axes_lines"]:
                        line.set_visible(show_lines)
                    for txt in arts["axis_labels"]:
                        txt.set_visible(show_lines)
                else:
                    arts["scatter_points"].set_visible(False)
                    arts["centroid_scatter"].set_visible(False)
                    arts["ellipsoid_surface"].set_visible(False)
                    for line in arts["axes_lines"]:
                        line.set_visible(False)
                    for txt in arts["axis_labels"]:
                        txt.set_visible(False)
            current_sample = label
            # Also load the original parameters into adjust_params for live editing
            if label in cached_ellipsoid_params:
                cparams = cached_ellipsoid_params[label]
                adjust_params["rx"] = cparams["rx"]
                adjust_params["ry"] = cparams["ry"]
                adjust_params["rz"] = cparams["rz"]
                adjust_params["tx"] = cparams["centroid"][0]
                adjust_params["ty"] = cparams["centroid"][1]
                adjust_params["tz"] = cparams["centroid"][2]
                adjust_params["rot_x"], adjust_params["rot_y"], adjust_params["rot_z"] = matrix_to_eulerXYZ(cparams["R"])

        fig.canvas.draw_idle()

    radio.on_clicked(on_clicked)

    # Optionally add the same sliders or +/- buttons for RX, RY, RZ, TX, TY, TZ, ROTX, ROTY, ROTZ, etc.
    # Just copy the same "adjust_buttons" logic from process_lda or process_pca if you need it.

    plt.show()


def process_qda():
    """
    Process the combined sample data using QDA (without any PCA).
    This function:
      1. Builds the sample_data_dict from the selected channels (as in process_lda).
      2. Combines the data from all samples and encodes the labels.
      3. Fits a QDA model on the combined data.
      4. Uses QDA's predicted probabilities as a surrogate transformation.
         (Since probabilities for K classes sum to 1, we drop the last column to yield K–1 dimensions.)
      5. Pads or truncates the resulting data to obtain a 3D representation.
      6. Computes centroids and ellipsoids (using your existing compute_ellipsoid function) for each sample.
      7. Plots the resulting 3D scatter plot with centroids, ellipsoid surfaces, and interactive controls.
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Step 1: Gather sample data (same as in process_lda) ---
    sample_data_dict = {}
    all_samples_list = [{"name": "Sample 1"}, {"name": "Sample 2"}] + samples
    for sample_entry in all_samples_list:
        sname = sample_entry["name"]
        if sname in file_paths and file_paths[sname]:
            sample_pixels = []
            for filepath in file_paths[sname]:
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                # Use your existing logic to extract valid pixels.
                rgb_frame = arr[0]
                spectral_frame = arr[1]
                white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
                border_size = border_sizes.get(sname, 1)
                valid_region = ~white_mask
                for _ in range(border_size):
                    valid_region = binary_erosion(valid_region)
                border_mask = (~white_mask) & (~valid_region)
                valid_pixels = ~white_mask & ~border_mask
                valid_coords = np.where(valid_pixels)
                channels_data = []
                for ch in [ch for ch, var in channel_vars.items() if var.get() == 1]:
                    if ch == "R":
                        data = rgb_frame[valid_coords][..., 0]
                    elif ch == "G":
                        data = rgb_frame[valid_coords][..., 1]
                    elif ch == "B":
                        data = rgb_frame[valid_coords][..., 2]
                    elif ch == "870":
                        data = spectral_frame[valid_coords][..., 0]
                    elif ch == "1200":
                        data = spectral_frame[valid_coords][..., 1]
                    elif ch == "1550":
                        data = spectral_frame[valid_coords][..., 2]
                    elif ch in ["L", "A", "B*"]:
                        lab = rgb2lab(rgb_frame)
                        idx = ["L", "A", "B*"].index(ch)
                        data = lab[valid_coords][..., idx]
                    if apply_normalization_var.get() == 1:
                        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
                    channels_data.append(data)
                if channels_data:
                    image_data = np.column_stack(channels_data)
                    sample_pixels.append(image_data)
            if sample_pixels:
                sample_data_dict[sname] = np.vstack(sample_pixels)
    if not sample_data_dict:
        messagebox.showerror("Error", "No valid sample data for QDA.")
        return

    # --- Step 2: Combine data and encode labels ---
    all_data = []
    label_list = []
    for sname, data_arr in sample_data_dict.items():
        all_data.append(data_arr)
        label_list.extend([sname] * len(data_arr))
    combined_data = np.vstack(all_data)
    label_array = np.array(label_list)
    le = LabelEncoder()
    numeric_labels = le.fit_transform(label_array)

    # --- Step 3: Fit QDA on the combined data ---
    qda = QDA()
    qda.fit(combined_data, numeric_labels)
    # Use QDA's predicted probabilities as a surrogate transformation.
    probas = qda.predict_proba(combined_data)  # shape: (n_samples, n_classes)
    # Since probabilities sum to 1, drop the last column to get a (n_classes-1)-dim representation.
    if probas.shape[1] > 1:
        Q_data = probas[:, :-1]
    else:
        Q_data = probas
    d = Q_data.shape[1]
    # If we have fewer than 3 dimensions, pad with zeros; if more, take the first 3.
    if d < 3:
        pad = np.zeros((Q_data.shape[0], 3 - d))
        Q_data = np.hstack([Q_data, pad])
    elif d > 3:
        Q_data = Q_data[:, :3]

    # --- Step 4: Compute centroids and filter points per sample (using KMeans) ---
    centroid_dict = {}
    global filtered_points_by_sample, cached_ellipsoid_params
    filtered_points_by_sample = {}
    for sname in sample_data_dict.keys():
        mask = (label_array == sname)
        points_3d = Q_data[mask]
        if points_3d.shape[0] < 2:
            continue
        km = KMeans(n_clusters=1, random_state=0).fit(points_3d)
        c_init = km.cluster_centers_[0]
        dist = np.linalg.norm(points_3d - c_init, axis=1)
        thresh = dist.mean() + 2.0 * dist.std()
        inlier_mask = (dist < thresh)
        if np.sum(inlier_mask) > 1:
            km_refined = KMeans(n_clusters=1, random_state=0).fit(points_3d[inlier_mask])
            c_refined = km_refined.cluster_centers_[0]
        else:
            c_refined = c_init
        centroid_dict[sname] = c_refined
        filtered_points_by_sample[sname] = points_3d

    cached_ellipsoid_params = {}
    for sname, pts in filtered_points_by_sample.items():
        (rx_val, ry_val, rz_val), R = compute_ellipsoid(pts, centroid_dict[sname])
        cached_ellipsoid_params[sname] = {"centroid": centroid_dict[sname],
                                          "rx": rx_val, "ry": ry_val, "rz": rz_val,
                                          "R": R}

    # --- Step 5: Plot the 3D QDA results ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.15, right=0.75)
    ax.set_xlabel("QDA Dim 1")
    ax.set_ylabel("QDA Dim 2")
    ax.set_zlabel("QDA Dim 3")
    ax.set_title("QDA (via predicted probabilities) 3D Representation")

    color_scheme = {"Sample 1": "green", "Sample 2": "red"}
    additional_colors = ["blue", "purple", "orange", "yellow", "cyan", "magenta"]
    color_idx = 0
    sample_plot_dict = {}

    def create_ellipsoid_surface(rx, ry, rz, rotation_matrix, tx, ty, tz, n=50):
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0, np.pi, n)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        x *= rx; y *= ry; z *= rz
        coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        coords = coords @ rotation_matrix.T
        coords += np.array([tx, ty, tz])
        X = coords[:, 0].reshape(n, n)
        Y = coords[:, 1].reshape(n, n)
        Z = coords[:, 2].reshape(n, n)
        return X, Y, Z

    for sname, pts_3d in filtered_points_by_sample.items():
        color = color_scheme[sname] if sname in color_scheme else additional_colors[color_idx % len(additional_colors)]
        color_idx += 1
        sc = ax.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2],
                        color=color, alpha=0.3, s=10, label=f"{sname} (N={len(pts_3d)})")
        ctd = centroid_dict[sname]
        sc_ctd = ax.scatter(ctd[0], ctd[1], ctd[2], color=color, edgecolor='k', s=200, marker='o',
                            label=f"{sname} centroid")
        rx = cached_ellipsoid_params[sname]["rx"]
        ry = cached_ellipsoid_params[sname]["ry"]
        rz = cached_ellipsoid_params[sname]["rz"]
        R = cached_ellipsoid_params[sname]["R"]
        X_e, Y_e, Z_e = create_ellipsoid_surface(rx, ry, rz, R, ctd[0], ctd[1], ctd[2])
        surf = ax.plot_surface(X_e, Y_e, Z_e, color=color, alpha=0.4, edgecolor='none')
        # Hide axis lines and labels by default
        sample_plot_dict[sname] = {
            "scatter_points": sc,
            "centroid_scatter": sc_ctd,
            "ellipsoid_surface": surf,
            "axes_lines": [],
            "axis_labels": []
        }
    ax.legend()
    plt.show()
##################################################################
########################### LAB CUBE #############################
##################################################################
# Import required libraries
#import numpy as np
#import cv2
#import tifffile
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from skimage.color import rgb2lab
#from scipy.ndimage import binary_erosion
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
#from tkinter import messagebox

# Fine-tuned scaling parameters (Pre-optimized for LAB transformation)
fine_tuned_params = np.array([
    1.6854,   # Scale factor for L
    1.7500,   # Scale factor for A
    -0.0083,  # Scale factor for B
    91.75,    # Offset for L
    -83.80,   # Offset for A
    147.48    # Offset for B
])

# Function to apply the fine-tuned LAB transformation
def transform_lab(lab_values):
    """
    Applies the fine-tuned scaling transformation to LAB values.
    Converts OpenCV (D50) LAB values to the optimized LAB values.
    """
    scale_L, scale_A, scale_B, offset_L, offset_A, offset_B = fine_tuned_params
    return np.array([
        lab_values[:, 0] * scale_L + offset_L,
        lab_values[:, 1] * scale_A + offset_A,
        lab_values[:, 2] * scale_B + offset_B
    ]).T

# Function to create ellipsoids for visualization
def create_ellipsoid(rx, ry, rz, rotation_matrix, tx, ty, tz, n=50):
    """
    Creates a 3D ellipsoid for plotting.
    """
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    x *= rx
    y *= ry
    z *= rz

    coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    coords = coords @ rotation_matrix.T
    coords += np.array([tx, ty, tz])

    X = coords[:, 0].reshape(n, n)
    Y = coords[:, 1].reshape(n, n)
    Z = coords[:, 2].reshape(n, n)
    
    return X, Y, Z

# Import required libraries
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2lab
from scipy.ndimage import binary_erosion
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tkinter import messagebox

# Fine-tuned scaling parameters (Pre-optimized for LAB transformation)
fine_tuned_params = np.array([
    1.6854,   # Scale factor for L
    1.7500,   # Scale factor for A
    -0.0083,  # Scale factor for B
    91.75,    # Offset for L
    -83.80,   # Offset for A
    147.48    # Offset for B
])

# Function to apply the fine-tuned LAB transformation
def transform_lab(lab_values):
    """
    Applies the fine-tuned scaling transformation to LAB values.
    Converts OpenCV (D50) LAB values to the optimized LAB values.
    """
    scale_L, scale_A, scale_B, offset_L, offset_A, offset_B = fine_tuned_params
    return np.array([
        lab_values[:, 0] * scale_L + offset_L,
        lab_values[:, 1] * scale_A + offset_A,
        lab_values[:, 2] * scale_B + offset_B
    ]).T

# Function to create ellipsoids for visualization
def create_ellipsoid(rx, ry, rz, rotation_matrix, tx, ty, tz, n=50):
    """
    Creates a 3D ellipsoid for plotting.
    """
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    x *= rx
    y *= ry
    z *= rz

    coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    coords = coords @ rotation_matrix.T
    coords += np.array([tx, ty, tz])

    X = coords[:, 0].reshape(n, n)
    Y = coords[:, 1].reshape(n, n)
    Z = coords[:, 2].reshape(n, n)
    
    return X, Y, Z

# Import required libraries
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2lab
from scipy.ndimage import binary_erosion
from sklearn.cluster import KMeans
from tkinter import messagebox

# Fine-tuned scaling parameters (Pre-optimized for LAB transformation)
fine_tuned_params = np.array([
    1.6854,   # Scale factor for L
    1.7500,   # Scale factor for A
    -0.0083,  # Scale factor for B
    91.75,    # Offset for L
    -83.80,   # Offset for A
    147.48    # Offset for B
])

# Function to apply the fine-tuned LAB transformation
def transform_lab(lab_values):
    """
    Applies the fine-tuned scaling transformation to LAB values.
    Converts OpenCV (D50) LAB values to the optimized LAB values.
    """
    scale_L, scale_A, scale_B, offset_L, offset_A, offset_B = fine_tuned_params
    return np.array([
        lab_values[:, 0] * scale_L + offset_L,
        lab_values[:, 1] * scale_A + offset_A,
        lab_values[:, 2] * scale_B + offset_B
    ]).T

# Function to create ellipsoids for visualization
def create_ellipsoid(rx, ry, rz, rotation_matrix, tx, ty, tz, n=50):
    """
    Creates a 3D ellipsoid for plotting.
    """
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    x *= rx
    y *= ry
    z *= rz

    coords = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    coords = coords @ rotation_matrix.T
    coords += np.array([tx, ty, tz])

    X = coords[:, 0].reshape(n, n)
    Y = coords[:, 1].reshape(n, n)
    Z = coords[:, 2].reshape(n, n)
    
    return X, Y, Z

# Function to process and visualize LAB cube without PCA
def process_lab_cube():
    from scipy.ndimage import binary_erosion
    from skimage.color import rgb2lab
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import tifffile
    import time  # Import for timing
    
    # Access the global sample_extracted_data which contains the "used in calculation" pixels
    global sample_extracted_data

    def custom_rgb_to_lab(rgb_frame):
        lab = rgb2lab(rgb_frame)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 2.55, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + 128.0, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + 128.0, 0, 255)
        return lab

    # Initialize the sample data dictionary
    sample_data_dict = {}
    all_samples = [{"name": "Sample 1"}, {"name": "Sample 2"}] + samples

    for sample_entry in all_samples:
        sname = sample_entry["name"]
        if sname in file_paths and file_paths[sname]:
            sample_pixels = []
            mean_colors = []
            total_valid_pixels = 0
            total_eroded_area = 0
            
            # For Sample 1, process using all valid pixels
            if sname == "Sample 1":
                # Regular processing for Sample 1
                for filepath in file_paths[sname]:
                    if filepath.lower().endswith(('.tif', '.tiff')):
                        with tifffile.TiffFile(filepath) as tf:
                            arr = tf.asarray()
                        rgb_frame = arr[0]
                    elif filepath.lower().endswith(('.png', '.bmp')):
                        img = Image.open(filepath).convert("RGB")
                        rgb_frame = np.array(img)
                    else:
                        continue

                    white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
                    border_size = border_sizes.get(sname, 1)
                    valid_region = ~white_mask
                    initial_valid_pixels = np.sum(valid_region)

                    for _ in range(border_size):
                        valid_region = binary_erosion(valid_region)

                    eroded_pixels = np.sum(valid_region)
                    eroded_area = initial_valid_pixels - eroded_pixels
                    total_valid_pixels += eroded_pixels
                    total_eroded_area += eroded_area

                    border_mask = (~white_mask) & (~valid_region)
                    valid_pixels = np.where(~white_mask & ~border_mask)

                    if valid_pixels[0].size > 0:
                        mean_rgb = np.mean(rgb_frame[valid_pixels], axis=0) / 255.0
                        mean_colors.append(mean_rgb)

                    lab_skimage = custom_rgb_to_lab(rgb_frame)
                    lab_values = np.column_stack([
                        lab_skimage[valid_pixels][..., 0],
                        lab_skimage[valid_pixels][..., 1],
                        lab_skimage[valid_pixels][..., 2]
                    ])
                    sample_pixels.append(lab_values)

                if sample_pixels:
                    sample_data_dict[sname] = {
                        "lab": np.vstack(sample_pixels),
                        "mean_rgb": np.mean(mean_colors, axis=0) if mean_colors else [0.5, 0.5, 0.5]
                    }
                    print(f"{sname}: Pixels used = {total_valid_pixels} / Total pixels incl. eroded area = {total_valid_pixels + total_eroded_area}")
            
            # For Sample 2 and beyond, only use "used in calculation" pixels
            else:
                # Check if sample data exists in sample_extracted_data
                if sname in sample_extracted_data:
                    # Process the stored RGB values directly
                    sample_rgb_values = np.array(sample_extracted_data[sname])
                    
                    # Skip if no data
                    if len(sample_rgb_values) == 0:
                        continue
                    
                    # Reshape to ensure 2D array
                    sample_rgb_values = sample_rgb_values.reshape(-1, 3)
                    
                    # Convert from RGB to LAB
                    lab_values = []
                    
                    # Process in batches to avoid memory issues
                    batch_size = 1000
                    for start_idx in range(0, len(sample_rgb_values), batch_size):
                        end_idx = min(start_idx + batch_size, len(sample_rgb_values))
                        batch = sample_rgb_values[start_idx:end_idx]
                        
                        # Create temporary images for each RGB value
                        batch_img = np.zeros((len(batch), 1, 3), dtype=np.uint8)
                        for i, rgb in enumerate(batch):
                            batch_img[i, 0] = rgb
                        
                        # Convert to LAB
                        lab_batch = custom_rgb_to_lab(batch_img)
                        
                        # Extract values
                        for i in range(len(batch)):
                            lab_values.append(lab_batch[i, 0])
                    
                    # Store the LAB values
                    if lab_values:
                        lab_values = np.array(lab_values)
                        mean_colors = [np.mean(sample_rgb_values, axis=0) / 255.0]
                        sample_data_dict[sname] = {
                            "lab": lab_values,
                            "mean_rgb": mean_colors[0]
                        }
                        print(f"{sname}: Using {len(lab_values)} pixels from stored calculation data")
                else:
                    print(f"{sname}: No data in sample_extracted_data, skipping")

    # Now create the visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_xlabel("L (Lightness)")
    ax.set_ylabel("A (Green-Red)")
    ax.set_zlabel("B (Blue-Yellow)")

    # Define lighter sample colors
    sample_colors = {
        "Sample 1": (0.4, 0.9, 0.4),  # Lighter green
        "Sample 2": (1.0, 0.5, 0.5)   # Lighter red
    }
    
    ellipsoid_colors = {
        "Sample 1": (0.0, 0.8, 0.0, 0.3),  # Green ellipsoid
        "Sample 2": (1.0, 0.0, 0.0, 0.3)   # Red ellipsoid
    }
    
    # Define scaling factors - use Sample 2's scaling factor for any additional samples
    default_scaling_factor = 0.7  # Same as Sample 2

    scaling_factors = {
        "Sample 1": 1.5,  # Specific for Sample 1
        # All other samples including Sample 2 will use the default scaling factor
    }

    for sname, info in sample_data_dict.items():
        data_arr = info["lab"]
        point_color = sample_colors.get(sname, info["mean_rgb"])
        
        # Use the mean centroid calculation for ALL samples
        centroid = np.mean(data_arr, axis=0)
        print(f"--- {sname}: Centroid = {centroid}")

        # Plot the data points with lighter colors and no borders
        ax.scatter(data_arr[:, 0], data_arr[:, 1], data_arr[:, 2],
                   color=point_color, alpha=0.4, 
                   s=10, edgecolors='none',
                   label=f"{sname} (N={len(data_arr)})")

        X_e, Y_e, Z_e = None, None, None

        # Get the scaling factor - use default (Sample 2's) value if not specified
        scaling = scaling_factors.get(sname, default_scaling_factor)

        # Start timing ellipsoid calculation
        ellipsoid_start_time = time.time()
        
        try:
            # Use MVEE with the mean centroid
            (rx, ry, rz), rotation_matrix, iterations, eigenvalues, eigenvectors = compute_ellipsoid(data_arr, centroid, max_iter=50)
            
            # Calculate elapsed time for ellipsoid computation
            ellipsoid_time = time.time() - ellipsoid_start_time
            print(f"{sname} Ellipsoid calculation time: {ellipsoid_time:.4f} seconds")
            
            print(f"{sname} Ellipsoid Radii: rx={rx}, ry={ry}, rz={rz}")
            print(f"{sname} Rotation Matrix:\n{rotation_matrix}")
            print(f"{sname} MVEE Iterations: {iterations}")
            print(f"{sname} Eigenvalues: {eigenvalues}")
            print(f"{sname} Eigenvectors:\n{eigenvectors}")
            
            # Scale the radii by the custom scaling factor
            rx *= scaling
            ry *= scaling
            rz *= scaling
            
            print(f"{sname} Scaled Ellipsoid Radii: rx={rx}, ry={ry}, rz={rz}")
            
            # Create ellipsoid with the scaled radii
            X_e, Y_e, Z_e = create_ellipsoid(rx, ry, rz, rotation_matrix,
                                           centroid[0], centroid[1], centroid[2])

        except Exception as e:
            # Calculate elapsed time for fallback method
            ellipsoid_time = time.time() - ellipsoid_start_time
            print(f"{sname} Fallback calculation time: {ellipsoid_time:.4f} seconds")
            
            print(f"MVEE failed for {sname}, fallback: {str(e)}")
            
            # Use the same mean centroid we calculated above
            cov = np.cov(data_arr, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            print(f"{sname} Fallback Eigenvalues: {eigenvalues}")
            print(f"{sname} Fallback Eigenvectors:\n{eigenvectors}")
            
            # Scale the eigenvalues by the scaling factor squared
            scaled_eigenvalues = eigenvalues * (scaling ** 2)
            
            print(f"{sname} Scaled Eigenvalues: {scaled_eigenvalues}")

            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            ellipsoid = np.array([x.flatten(), y.flatten(), z.flatten()]).T
            
            # Use the scaled eigenvalues
            ellipsoid = ellipsoid @ np.diag(np.sqrt(scaled_eigenvalues)) @ eigenvectors.T + centroid

            X_e, Y_e, Z_e = (ellipsoid[:, 0].reshape(30, 30),
                           ellipsoid[:, 1].reshape(30, 30),
                           ellipsoid[:, 2].reshape(30, 30))
        
        # Add time for the entire ellipsoid process
        total_ellipsoid_time = time.time() - ellipsoid_start_time
        print(f"{sname} Total ellipsoid processing time: {total_ellipsoid_time:.4f} seconds")

        if X_e is not None:
            ellipsoid_color = ellipsoid_colors.get(sname, (0.5, 0.5, 0.5, 0.3))
            ax.plot_surface(X_e, Y_e, Z_e, color=ellipsoid_color, alpha=0.3)

    ax.legend()
    plt.title("LAB Cube with Ellipsoids")
    plt.show()


def process_lab_pca():
    """
    Processes samples in LAB color space, applies PCA + LDA, and visualizes in a 3D plot.
    Filters out white background and erosion borders based on erosion settings.
    """
    from skimage.color import rgb2lab
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from scipy.ndimage import binary_erosion
    import numpy as np
    import matplotlib.pyplot as plt

    global cached_ellipsoid_params, filtered_points_by_sample, current_sample, adjust_params

    sample_data_dict = {}
    all_samples = [{"name": "Sample 1"}, {"name": "Sample 2"}] + samples

    for sample_entry in all_samples:
        sname = sample_entry["name"]
        if sname in file_paths and file_paths[sname]:
            sample_pixels = []
            for filepath in file_paths[sname]:
                if filepath.lower().endswith(('.tif', '.tiff')):
                    with tifffile.TiffFile(filepath) as tf:
                        arr = tf.asarray()
                    rgb_frame = arr[0]
                elif filepath.lower().endswith(".png"):
                    img = Image.open(filepath).convert("RGB")
                    rgb_frame = np.array(img)
                else:
                    continue

                # Erosion-aware filtering logic:
                white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
                border_size = border_sizes.get(sname, 1)
                valid_region = ~white_mask
                for _ in range(border_size):
                    valid_region = binary_erosion(valid_region)
                border_mask = (~white_mask) & (~valid_region)
                valid_pixels = np.where(~white_mask & ~border_mask)

                # Convert RGB to LAB
                lab_skimage = rgb2lab(rgb_frame)
                lab_skimage[..., 0] = np.clip((lab_skimage[..., 0] / 100) * 255, 0, 255)
                lab_skimage[..., 1] = np.clip(lab_skimage[..., 1] + 128, 0, 255)
                lab_skimage[..., 2] = np.clip(lab_skimage[..., 2] + 128, 0, 255)

                lab_values = np.column_stack([
                    lab_skimage[valid_pixels][..., 0],
                    lab_skimage[valid_pixels][..., 1],
                    lab_skimage[valid_pixels][..., 2]
                ])

                sample_pixels.append(lab_values)

            if sample_pixels:
                sample_data_dict[sname] = np.vstack(sample_pixels)

    if not sample_data_dict:
        messagebox.showerror("Error", "No valid data for LAB PCA+LDA visualization")
        return

    all_data = []
    all_labels = []
    for sname, data_arr in sample_data_dict.items():
        all_data.append(data_arr)
        all_labels.extend([sname] * len(data_arr))
    combined_data = np.vstack(all_data)
    all_labels = np.array(all_labels)

    # Apply PCA (2 components)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(combined_data)

    # Apply LDA (1 component)
    lda = LDA(n_components=1)
    lda_data = lda.fit_transform(combined_data, all_labels)

    # Combine PCA1, PCA2, and LDA1
    combined_pca_lda = np.column_stack((pca_data, lda_data))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("LD1")

    color_scheme = {"Sample 1": "green", "Sample 2": "red"}
    additional_colors = ["blue", "purple", "orange", "yellow", "cyan", "magenta"]
    color_idx = 0
    
    for sname, data_arr in sample_data_dict.items():
        mask = (all_labels == sname)
        points_3d = combined_pca_lda[mask]
        color = color_scheme.get(sname, additional_colors[color_idx % len(additional_colors)])
        color_idx += 1
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                   color=color, alpha=0.5, s=10, label=f"{sname} (N={len(points_3d)})")

    ax.legend()
    plt.title("PCA1, PCA2, and LDA1 of LAB Space (With Erosion Filtering)")
    plt.show()

##############################################################################
####LAB PLANE#####
##############################################################################
def process_lab_plane():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from sklearn.cluster import KMeans
    from skimage.color import rgb2lab
    from scipy.ndimage import binary_erosion
    from scipy.optimize import differential_evolution
    from PIL import Image
    import tifffile

    def custom_rgb_to_lab(rgb_frame):
        lab = rgb2lab(rgb_frame)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 2.55, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + 128.0, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + 128.0, 0, 255)
        return lab

    def get_cube_faces():
        faces = []
        for axis in range(3):
            for const in [0, 255]:
                p = [np.array([0, 0, 0], dtype=float) for _ in range(4)]
                idx = [(0, 0), (1, 0), (1, 1), (0, 1)]
                for i, (a, b) in enumerate(idx):
                    p[i][axis] = const
                    p[i][(axis + 1) % 3] = 255 * a
                    p[i][(axis + 2) % 3] = 255 * b
                faces.append(p)
        return faces

    def clip_face_against_plane(face_pts, plane_point, plane_normal, tol=1e-5):
        def is_inside(p):
            return np.dot(p - plane_point, plane_normal) <= tol

        def intersect(p1, p2):
            d1 = np.dot(p1 - plane_point, plane_normal)
            d2 = np.dot(p2 - plane_point, plane_normal)
            if abs(d1 - d2) < tol:
                return p1
            t = d1 / (d1 - d2)
            return p1 + t * (p2 - p1)

        clipped = []
        for i in range(4):
            curr = face_pts[i]
            prev = face_pts[i - 1]
            curr_in = is_inside(curr)
            prev_in = is_inside(prev)
            if curr_in:
                if not prev_in:
                    clipped.append(intersect(prev, curr))
                clipped.append(curr)
            elif prev_in:
                clipped.append(intersect(prev, curr))
        return clipped if len(clipped) >= 3 else None

    def refine_centroid(points):
        km = KMeans(n_clusters=1, random_state=0).fit(points)
        c_init = km.cluster_centers_[0]
        dist = np.linalg.norm(points - c_init, axis=1)
        inlier_mask = dist < (dist.mean() + 2 * dist.std())
        if np.sum(inlier_mask) > 1:
            return KMeans(n_clusters=1, random_state=0).fit(points[inlier_mask]).cluster_centers_[0]
        return c_init

    def optimize_separation_plane(sample1, sample2):
        """
        Create a separation plane perpendicular to the line connecting centroids.
        Only optimizes the sensitivity parameter to achieve desired coverage.
        
        Args:
            sample1: Points from the first sample (good)
            sample2: Points from the second sample (defect)
            
        Returns:
            Array containing [azimuth, elevation, sensitivity] parameters
        """
        # Calculate refined centroids of both samples
        c1 = refine_centroid(sample1)
        c2 = refine_centroid(sample2)
        
        # Vector from c1 to c2
        delta = c2 - c1
        
        # Normalize the delta vector
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-6:  # Avoid division by zero
            # Fallback to default if centroids are too close
            return np.array([0, 90, 0])
        
        delta_unit = delta / delta_norm
        
        # Calculate the azimuth and elevation from the delta vector
        # Convert from cartesian to spherical coordinates
        x, y, z = delta_unit
        
        # Calculate elevation in degrees (-90 to 90)
        elevation = np.degrees(np.arcsin(z))
        
        # Calculate azimuth in degrees (0 to 360)
        azimuth = np.degrees(np.arctan2(y, x))
        if azimuth < 0:
            azimuth += 360
        
        # Use 0 as initial sensitivity (midpoint between centroids)
        # The calling function will adjust sensitivity to achieve desired coverage
        sensitivity = 0
        
        # The plane normal should point from c2 to c1 (defect to good)
        # This is the opposite of delta_unit
        # We'll handle this in redraw_cut_plane by checking dot product
        
        return np.array([azimuth, elevation, sensitivity])
    sample_data = {}
    for sname in ["Sample 1", "Sample 2"]:
        if sname in file_paths and file_paths[sname]:
            pixels = []
            for filepath in file_paths[sname]:
                if filepath.lower().endswith(('.tif', '.tiff')):
                    with tifffile.TiffFile(filepath) as tf:
                        arr = tf.asarray()
                    rgb_frame = arr[0]
                elif filepath.lower().endswith(('.png', '.bmp')):
                    img = Image.open(filepath).convert("RGB")
                    rgb_frame = np.array(img)
                else:
                    continue
                white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
                border_size = border_sizes.get(sname, 1)
                valid_region = ~white_mask
                for _ in range(border_size):
                    valid_region = binary_erosion(valid_region)
                border_mask = (~white_mask) & (~valid_region)
                valid_pixels = np.where(~white_mask & ~border_mask)
                lab = custom_rgb_to_lab(rgb_frame)
                lab_valid = np.column_stack([
                    lab[valid_pixels][..., 0],
                    lab[valid_pixels][..., 1],
                    lab[valid_pixels][..., 2]
                ])
                pixels.append(lab_valid)
            sample_data[sname] = np.vstack(pixels)

    sample1 = sample_data["Sample 1"]
    sample2 = sample_data["Sample 2"]

    
    c1 = refine_centroid(sample1)
    c2 = refine_centroid(sample2)
    delta = c2 - c1
    delta /= np.linalg.norm(delta)
    normal_ref = delta.copy()

    def get_plane_params(azimuth, elevation, sensitivity):
        n = np.array([
            np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)),
            np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)),
            np.sin(np.radians(elevation))
        ])
        n /= np.linalg.norm(n)
        if np.dot(n, normal_ref) > 0:
            n = -n
        midpoint = c1 + 0.5 * (c2 - c1)
        offset = -delta * sensitivity * 255
        plane_center = midpoint + offset
        return plane_center, n

    def draw_shaded_half_cube(azimuth, elevation, sensitivity):
        ax.clear()
        bar_ax.clear()
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        ax.set_xlabel("L")
        ax.set_ylabel("A")
        ax.set_zlabel("B")
        ax.set_title("LAB Cube – Shaded Bad Side")

        

        plane_center, normal = get_plane_params(azimuth, elevation, sensitivity)
        dot_products = np.dot(sample1 - plane_center, normal)
        sample1_outside = sample1[dot_products > 0]
        sample1_inside = sample1[dot_products <= 0]

        # Plot green points outside shaded region
        ax.scatter(sample1_outside[:, 0], sample1_outside[:, 1], sample1_outside[:, 2], color='green', alpha=0.4, s=10, label="Sample 1 (outside)")

        # Plot orange points inside shaded region
        ax.scatter(sample1_inside[:, 0], sample1_inside[:, 1], sample1_inside[:, 2], color='orange', alpha=0.6, s=10, label="Sample 1 (inside)")

        ax.scatter(sample2[:, 0], sample2[:, 1], sample2[:, 2], color='red', alpha=0.4, s=10, label="Sample 2")
        if np.dot(normal, normal_ref) > 0:
            normal = -normal

        faces = get_cube_faces()
        for face in faces:
            clipped = clip_face_against_plane(face, plane_center, normal)
            if clipped:
                poly = Poly3DCollection([clipped], facecolor='red', alpha=0.2, edgecolor='black')
                ax.add_collection3d(poly)

        bad_inside = np.sum(np.dot(sample2 - plane_center, normal) <= 0)
        good_inside = np.sum(np.dot(sample1 - plane_center, normal) <= 0)

        ax.text2D(0.75, 0.95, f"Bad Inside: {bad_inside}\nGood Inside: {good_inside}", transform=fig.transFigure)
        bar_ax.bar(["Bad Inside", "Good Inside"], [good_inside, bad_inside], color=["green", "red"])
        
        bar_ax.set_ylim(0, max(sample1.shape[0], sample2.shape[0]) * 1.1)
        bar_ax.set_ylabel("Count")
        bar_ax.set_title("Points Inside Shaded Region")
        ax.legend()
        fig.canvas.draw_idle()

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(121, projection='3d')
    bar_ax = fig.add_subplot(236)

    optimal_params = optimize_separation_plane(sample1, sample2)
    print("Optimal:", optimal_params)

    ax_theta = plt.axes([0.20, 0.03, 0.50, 0.02])
    ax_phi = plt.axes([0.20, 0.00, 0.50, 0.02])
    ax_sens = plt.axes([0.20, 0.06, 0.50, 0.02])
    ax_button = plt.axes([0.75, 0.00, 0.15, 0.05])

    slider_theta = Slider(ax_theta, 'Azimuth (°)', 0, 360, valinit=optimal_params[0])
    slider_phi = Slider(ax_phi, 'Elevation (°)', -90, 90, valinit=optimal_params[1])
    slider_sens = Slider(ax_sens, 'Sensitivity', -2.0, 2.0, valinit=optimal_params[2])
    button_auto = Button(ax_button, 'Auto Calculate')

    def update(val=None):
        draw_shaded_half_cube(slider_theta.val, slider_phi.val, slider_sens.val)

    def auto_calculate(event=None):
        slider_theta.set_val(optimal_params[0])
        slider_phi.set_val(optimal_params[1])
        slider_sens.set_val(optimal_params[2])
        draw_shaded_half_cube(*optimal_params)

    slider_theta.on_changed(update)
    slider_phi.on_changed(update)
    slider_sens.on_changed(update)
    button_auto.on_clicked(auto_calculate)

    draw_shaded_half_cube(*optimal_params)
    plt.show()


###############################
###### Dynamic Plane###########
def launch_dynamic_plane():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.gridspec as gridspec
    from matplotlib.widgets import Slider, CheckButtons, Button
    from skimage.color import rgb2lab
    from scipy.ndimage import binary_erosion
    from sklearn.cluster import KMeans
    from scipy.optimize import differential_evolution
    from PIL import Image
    import tifffile
    global anomaly_sensitivities

    sample1_data = []
    if "Sample 1" in file_paths:
        for filepath in file_paths["Sample 1"]:
            try:
                # Load image
                if filepath.lower().endswith(('.tif', '.tiff')):
                    with tifffile.TiffFile(filepath) as tf:
                        arr = tf.asarray()
                        rgb_frame = arr[0]
                elif filepath.lower().endswith(('.png', '.bmp')):
                    img = Image.open(filepath).convert("RGB")
                    rgb_frame = np.array(img)
                else:
                    continue
                
                # Create masks for background and borders
                white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
                border_size = border_sizes.get("Sample 1", 1)
                valid_region = ~white_mask
                for _ in range(border_size):
                    valid_region = binary_erosion(valid_region)
                border_mask = (~white_mask) & (~valid_region)
                valid_pixels = ~white_mask & ~border_mask
                
                # Extract all valid RGB values
                valid_rgb = rgb_frame[valid_pixels]
                sample1_data.extend(valid_rgb)
                print(f"Added {len(valid_rgb)} pixels from {filepath}")
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
        
        # Replace sample_extracted_data for Sample 1
        sample_extracted_data["Sample 1"] = sample1_data
        print(f"Loaded {len(sample1_data)} total Sample 1 pixels")
    

    # Define helper functions first (so they're available in all code paths)
    def custom_rgb_to_lab(rgb_frame):
        """Original custom LAB conversion function"""
        lab = rgb2lab(rgb_frame)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 2.55, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + 128.0, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + 128.0, 0, 255)
        return lab

    def get_lab_data(sample_name):
        """Get LAB data from the given sample name"""
        pixels = []
        if sample_name in file_paths and file_paths[sample_name]:
            for filepath in file_paths[sample_name]:
                if filepath.lower().endswith(('.tif', '.tiff')):
                    with tifffile.TiffFile(filepath) as tf:
                        arr = tf.asarray()
                    rgb_frame = arr[0]
                elif filepath.lower().endswith(('.png', '.bmp')):
                    img = Image.open(filepath).convert("RGB")
                    rgb_frame = np.array(img)
                else:
                    continue
                white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
                border_size = border_sizes.get(sample_name, 1)
                valid_region = ~white_mask
                for _ in range(border_size):
                    valid_region = binary_erosion(valid_region)
                border_mask = (~white_mask) & (~valid_region)
                valid_pixels = np.where(~white_mask & ~border_mask)
                lab = custom_rgb_to_lab(rgb_frame)
                lab_valid = np.column_stack([
                    lab[valid_pixels][..., 0],
                    lab[valid_pixels][..., 1],
                    lab[valid_pixels][..., 2]
                ])
                pixels.append(lab_valid)
        return np.vstack(pixels) if pixels else np.array([])

    def load_image(sample_name):
        """Helper to load the first image for a sample"""
        if sample_name in file_paths and file_paths[sample_name]:
            img_path = file_paths[sample_name][0]
            if img_path.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(img_path) as tf:
                    return tf.asarray()[0]
            else:
                return np.array(Image.open(img_path).convert("RGB"))
        return np.zeros((100, 100, 3), dtype=np.uint8)  # Fallback empty image

    def process_stored_rgb(rgb_array):
        """Process stored RGB values to LAB using batches"""
        if len(rgb_array) == 0:
            return np.array([])
            
        # Create a temporary 1-pixel image for each RGB value and convert it
        lab_values = []
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        for start_idx in range(0, len(rgb_array), batch_size):
            end_idx = min(start_idx + batch_size, len(rgb_array))
            batch = rgb_array[start_idx:end_idx]
            
            # Create a multi-pixel image for the batch
            batch_img = np.zeros((len(batch), 1, 3), dtype=np.uint8)
            for i, rgb in enumerate(batch):
                batch_img[i, 0] = rgb
                
            # Apply the custom LAB conversion
            lab_batch = custom_rgb_to_lab(batch_img)
            
            # Extract the LAB values
            for i in range(len(batch)):
                lab_values.append(lab_batch[i, 0])
        
        return np.array(lab_values)

    def refine_centroid(points):
        """Optimized version that caches results"""
        # Check if points shape/identity matches the cached version
        cache_key = id(points)
        if hasattr(refine_centroid, 'cache') and cache_key in refine_centroid.cache:
            return refine_centroid.cache[cache_key]
        
        # Regular KMeans implementation
        km = KMeans(n_clusters=1, random_state=0).fit(points)
        c_init = km.cluster_centers_[0]
        
        # Use vectorized operations for distance calculation
        dist = np.linalg.norm(points - c_init, axis=1)
        inlier_mask = dist < (dist.mean() + 2 * dist.std())
        
        # Only perform second KMeans if needed
        if np.sum(inlier_mask) > 1:
            result = KMeans(n_clusters=1, random_state=0).fit(points[inlier_mask]).cluster_centers_[0]
        else:
            result = c_init
        
        # Cache the result
        if not hasattr(refine_centroid, 'cache'):
            refine_centroid.cache = {}
        refine_centroid.cache[cache_key] = result
        
        return result

    def optimize_separation_plane(sample1, sample2, target_initial_count=20, target_final_coverage=90.0):
        """
        Iterative optimization of plane orientation and sensitivity:
        1. Start with initial plane based on centroids
        2. Set sensitivity to include ~20 defect points
        3. Iteratively rotate plane to minimize good points inside
        4. Finally, adjust sensitivity to cover 90% of defect points
        
        Args:
            sample1: Points from sample 1 (good)
            sample2: Points from sample 2/3 (defect)
            target_initial_count: Target number of defect points for optimization (~20)
            target_final_coverage: Final coverage percentage of defect points (90%)
            
        Returns:
            Array containing [azimuth, elevation, sensitivity] parameters
        """
        import time
        start_time = time.time()
        
        # Calculate centroids
        c1 = refine_centroid(sample1)
        c2 = refine_centroid(sample2)
        
        # Initial direction based on centroids
        delta = c2 - c1
        delta_norm = np.linalg.norm(delta)
        
        if delta_norm < 1e-6:  # Avoid division by zero
            # Fallback to default if centroids are too close
            return np.array([0, 90, 0])
            
        delta_unit = delta / delta_norm
        
        # Convert to spherical coordinates for easier rotation
        x, y, z = delta_unit
        initial_elevation = np.degrees(np.arcsin(z))
        initial_azimuth = np.degrees(np.arctan2(y, x))
        if initial_azimuth < 0:
            initial_azimuth += 360
            
        print(f"Initial direction: Az: {initial_azimuth:.2f}°, El: {initial_elevation:.2f}°")
        
        # Find sensitivity that includes approximately target_initial_count defect points
        count_sensitivity = find_sensitivity_for_count(sample1, sample2, c1, c2, delta_unit, target_initial_count)
        
        # Start iteration
        current_az = initial_azimuth
        current_el = initial_elevation
        current_normal = delta_unit
        
        # Track the best plane parameters
        best_az = current_az
        best_el = current_el
        best_normal = current_normal
        best_good_inside = len(sample1)  # Worst case
        
        # Parameters for grid search
        max_iterations = 10
        initial_step = 15.0  # Degrees
        min_step = 1.0  # Minimum step size
        
        # Iterative refinement loop
        for iteration in range(max_iterations):
            improved = False
            step_size = initial_step / (2 ** iteration)  # Reduce step size each iteration
            
            if step_size < min_step:
                print(f"Step size {step_size:.2f}° below minimum, stopping")
                break
                
            print(f"\nIteration {iteration+1}, step size: {step_size:.2f}°")
            
            # Try rotating in different directions
            for d_az in [-step_size, 0, step_size]:
                for d_el in [-step_size, 0, step_size]:
                    # Skip the center point (no change)
                    if d_az == 0 and d_el == 0:
                        continue
                        
                    # Calculate new angles
                    test_az = current_az + d_az
                    if test_az < 0:
                        test_az += 360
                    if test_az >= 360:
                        test_az -= 360
                        
                    test_el = current_el + d_el
                    if test_el < -90:
                        test_el = -90
                    if test_el > 90:
                        test_el = 90
                    
                    # Convert to normal vector
                    test_el_rad = np.radians(test_el)
                    test_az_rad = np.radians(test_az)
                    test_normal = np.array([
                        np.cos(test_el_rad) * np.cos(test_az_rad),
                        np.cos(test_el_rad) * np.sin(test_az_rad),
                        np.sin(test_el_rad)
                    ])
                    
                    # Ensure normal points from good to defect
                    if np.dot(test_normal, delta_unit) < 0:
                        test_normal = -test_normal
                    
                    # Find midpoint and adjust by count_sensitivity
                    midpoint = (c1 + c2) / 2
                    offset = delta_unit * count_sensitivity * delta_norm
                    plane_center = midpoint + offset
                    
                    # Count points inside
                    s1_inside = np.sum(np.dot(sample1 - plane_center, test_normal) <= 0)
                    s2_inside = np.sum(np.dot(sample2 - plane_center, test_normal) <= 0)
                    
                    # Check if rotation improved (reduced good points while keeping ~target defect points)
                    # Allow some flexibility in defect count to prioritize reducing good count
                    defect_tolerance = 0.1 * target_initial_count  # Allow 10% tolerance
                    
                    if s1_inside < best_good_inside and abs(s2_inside - target_initial_count) <= defect_tolerance:
                        best_good_inside = s1_inside
                        best_az = test_az
                        best_el = test_el
                        best_normal = test_normal
                        improved = True
                        
            # Update current parameters to best from this iteration
            if improved:
                current_az = best_az
                current_el = best_el
                current_normal = best_normal
                print(f"Improved plane: Az: {current_az:.2f}°, El: {current_el:.2f}°, Good inside: {best_good_inside}, Defect inside: ~{target_initial_count}")
            else:
                print(f"No improvement in this iteration")
                break
        
        # Final plane orientation from iterative process
        final_az = current_az
        final_el = current_el
        final_normal = current_normal
        
        # Now find sensitivity to cover target_final_coverage% of defect points
        final_sensitivity = find_sensitivity_for_coverage(sample1, sample2, c1, c2, final_normal, target_final_coverage)
        
        # Convert sensitivity to slider value (0-400)
        slider_sensitivity = 200 + int(final_sensitivity * 100 / delta_norm)
        slider_sensitivity = max(0, min(400, slider_sensitivity))
        
        completion_time = time.time() - start_time
        print(f"\nOptimization complete in {completion_time:.4f} seconds")
        print(f"Final plane: Az: {final_az:.2f}°, El: {final_el:.2f}°, Sens: {slider_sensitivity}")
        
        # Calculate final coverage
        midpoint = (c1 + c2) / 2
        offset = delta_unit * final_sensitivity
        plane_center = midpoint + offset
        
        s1_inside = np.sum(np.dot(sample1 - plane_center, final_normal) <= 0)
        s2_inside = np.sum(np.dot(sample2 - plane_center, final_normal) <= 0)
        
        s1_inside_pct = s1_inside / len(sample1) * 100
        s2_inside_pct = s2_inside / len(sample2) * 100
        
        print(f"Final coverage: Sample 1: {s1_inside_pct:.2f}%, Sample 2: {s2_inside_pct:.2f}%")
        
        return np.array([final_az, final_el, slider_sensitivity])

    def increment_sensitivity(event):
        """Increment sensitivity and force plane re-optimization"""
        current_sens = slider_sens.val
        new_sens = min(400, current_sens + 1)
        
        # Update the slider value (without triggering normal callbacks)
        slider_sens.eventson = False
        slider_sens.set_val(new_sens)
        slider_sens.eventson = True
        
        # Update current state
        current_state = sample3_state if selected_sample[0] == "Sample 3" else sample2_state
        current_state['sensitivity'] = new_sens
        
        print(f"Sensitivity incremented to {new_sens}, running optimization...")
        
        # FORCE reoptimization by directly calling optimize_separation_plane
        # This is independent of the auto_calculate function
        try:
            points = get_current_sample2_points()
            if points is not None and len(points) >= 3:
                # Run differential_evolution with the new sensitivity value
                
                # Define a local separation score function with the new sensitivity
                def local_separation_score(params):
                    # Only use azimuth and elevation from params
                    azimuth, elevation = params
                    normal = np.array([
                        np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)),
                        np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)),
                        np.sin(np.radians(elevation))
                    ])
                    normal /= np.linalg.norm(normal)
                    
                    c1 = refine_centroid(sample1)
                    c2 = refine_centroid(points)
                    
                    delta = c2 - c1
                    delta /= np.linalg.norm(delta)
                    
                    if np.dot(normal, delta) > 0:
                        normal = -normal
                    
                    midpoint = c1 + 0.5 * (c2 - c1)
                    
                    # Use the current sensitivity value
                    sensitivity_adjusted = (new_sens - 200) / 200.0  # Normalize to [-1, 1]
                    offset = delta * sensitivity_adjusted * 255
                    plane_center = midpoint + offset
                    
                    # Calculate points inside the plane
                    bad_inside = np.sum(np.dot(points - plane_center, normal) <= 0)
                    good_inside = np.sum(np.dot(sample1 - plane_center, normal) <= 0)
                    
                    # Score based on maximizing defect coverage while minimizing good coverage
                    # Adjust weights as needed (0.1 penalty for good points inside)
                    return -(bad_inside - 0.1 * good_inside)
                
                # Define narrower bounds for a faster search (just azimuth and elevation)
                bounds = [(0, 360), (-90, 90)]
                
                # Run a quick optimization with reduced parameters
                print("Starting differential evolution...")
                result = differential_evolution(
                    local_separation_score, 
                    bounds, 
                    strategy='best1bin', 
                    popsize=10,  # Smaller population 
                    maxiter=10,  # Fewer iterations
                    polish=False
                )
                
                # Extract optimal parameters
                optimal_azimuth = result.x[0]
                optimal_elevation = result.x[1]
                
                print(f"Optimization complete: Az: {optimal_azimuth:.1f}°, El: {optimal_elevation:.1f}°")
                
                # Update sliders without triggering callbacks
                slider_azim.eventson = False
                slider_elev.eventson = False
                
                slider_azim.set_val(optimal_azimuth)
                elevation_slider_val = optimal_elevation + 90
                slider_elev.set_val(elevation_slider_val)
                
                slider_azim.eventson = True
                slider_elev.eventson = True
                
                # Update current state
                current_state['azimuth'] = optimal_azimuth
                current_state['elevation'] = elevation_slider_val
                
                # Redraw the cut plane with new parameters
                redraw_cut_plane()
                
                # Update title
                ax_3d.set_title(
                    f"Optimized (Sens: {new_sens}, Az: {optimal_azimuth:.1f}°, El: {elevation_slider_val:.1f}°)"
                )
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            # On error, just redraw with new sensitivity
            redraw_cut_plane()


    def decrement_sensitivity(event):
        """Decrement sensitivity and force plane re-optimization"""
        current_sens = slider_sens.val
        new_sens = max(0, current_sens - 1)
        
        # Update the slider value (without triggering normal callbacks)
        slider_sens.eventson = False
        slider_sens.set_val(new_sens)
        slider_sens.eventson = True
        
        # Update current state
        current_state = sample3_state if selected_sample[0] == "Sample 3" else sample2_state
        current_state['sensitivity'] = new_sens
        
        print(f"Sensitivity decremented to {new_sens}, running optimization...")
        
        # FORCE reoptimization by directly calling optimize_separation_plane
        # This is independent of the auto_calculate function
        try:
            points = get_current_sample2_points()
            if points is not None and len(points) >= 3:
                # Run differential_evolution with the new sensitivity value
                
                # Define a local separation score function with the new sensitivity
                def local_separation_score(params):
                    # Only use azimuth and elevation from params
                    azimuth, elevation = params
                    normal = np.array([
                        np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)),
                        np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)),
                        np.sin(np.radians(elevation))
                    ])
                    normal /= np.linalg.norm(normal)
                    
                    c1 = refine_centroid(sample1)
                    c2 = refine_centroid(points)
                    
                    delta = c2 - c1
                    delta /= np.linalg.norm(delta)
                    
                    if np.dot(normal, delta) > 0:
                        normal = -normal
                    
                    midpoint = c1 + 0.5 * (c2 - c1)
                    
                    # Use the current sensitivity value
                    sensitivity_adjusted = (new_sens - 200) / 200.0  # Normalize to [-1, 1]
                    offset = delta * sensitivity_adjusted * 255
                    plane_center = midpoint + offset
                    
                    # Calculate points inside the plane
                    bad_inside = np.sum(np.dot(points - plane_center, normal) <= 0)
                    good_inside = np.sum(np.dot(sample1 - plane_center, normal) <= 0)
                    
                    # Score based on maximizing defect coverage while minimizing good coverage
                    # Adjust weights as needed (0.1 penalty for good points inside)
                    return -(bad_inside - 0.1 * good_inside)
                
                # Define narrower bounds for a faster search (just azimuth and elevation)
                bounds = [(0, 360), (-90, 90)]
                
                # Run a quick optimization with reduced parameters
                print("Starting differential evolution...")
                result = differential_evolution(
                    local_separation_score, 
                    bounds, 
                    strategy='best1bin', 
                    popsize=10,  # Smaller population 
                    maxiter=10,  # Fewer iterations
                    polish=False
                )
                
                # Extract optimal parameters
                optimal_azimuth = result.x[0]
                optimal_elevation = result.x[1]
                
                print(f"Optimization complete: Az: {optimal_azimuth:.1f}°, El: {optimal_elevation:.1f}°")
                
                # Update sliders without triggering callbacks
                slider_azim.eventson = False
                slider_elev.eventson = False
                
                slider_azim.set_val(optimal_azimuth)
                elevation_slider_val = optimal_elevation + 90
                slider_elev.set_val(elevation_slider_val)
                
                slider_azim.eventson = True
                slider_elev.eventson = True
                
                # Update current state
                current_state['azimuth'] = optimal_azimuth
                current_state['elevation'] = elevation_slider_val
                
                # Redraw the cut plane with new parameters
                redraw_cut_plane()
                
                # Update title
                ax_3d.set_title(
                    f"Optimized (Sens: {new_sens}, Az: {optimal_azimuth:.1f}°, El: {elevation_slider_val:.1f}°)"
                )
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            # On error, just redraw with new sensitivity
            redraw_cut_plane()


    # Add this after your existing sliders and buttons setup
    

    def find_sensitivity_for_count(sample1, sample2, c1, c2, normal, target_count):
        """
        Find sensitivity value that includes approximately target_count defect points
        
        Args:
            sample1: Sample 1 points (good)
            sample2: Sample 2 points (defect)
            c1, c2: Centroids of samples
            normal: Plane normal vector
            target_count: Target number of defect points to include
        
        Returns:
            Sensitivity value
        """
        # Distance between centroids for normalization
        centroid_distance = np.linalg.norm(c2 - c1)
        
        # Start at midpoint between centroids
        midpoint = (c1 + c2) / 2
        
        # Compute dot products once
        s2_dots = np.dot(sample2 - midpoint, normal)
        
        # Sort dot products to allow binary search
        sorted_dots = np.sort(s2_dots)
        
        # Target index (we want target_count points inside)
        target_idx = len(sorted_dots) - target_count
        
        # If target_idx out of bounds, adjust
        if target_idx < 0:
            target_idx = 0
        if target_idx >= len(sorted_dots):
            target_idx = len(sorted_dots) - 1
            
        # Get threshold that gives this count
        threshold = sorted_dots[target_idx]
        
        # Convert threshold to offset from midpoint
        sensitivity = threshold / np.dot(normal, normal)
        
        print(f"Sensitivity for ~{target_count} defect points: {sensitivity:.4f}")
        
        return sensitivity


    def find_sensitivity_for_coverage(sample1, sample2, target_coverage=80.0):
        """
        Find the sensitivity value that results in a specific coverage percentage
        of Sample 2/3 points inside the shaded area.
        
        Args:
            sample1: Sample 1 points (good)
            sample2: Sample 2 points (defect)
            target_coverage: Target percentage of defect points to include (default: 80%)
            
        Returns:
            Sensitivity value (0-400 range for slider)
        """
        # Get current plane orientation
        current_state = sample3_state if selected_sample[0] == "Sample 3" else sample2_state
        azimuth = current_state['azimuth']
        elevation = current_state['elevation'] - 90  # Convert to -90 to 90 range
        
        # Calculate normal vector
        elev_rad = np.deg2rad(elevation)
        azim_rad = np.deg2rad(azimuth)
        normal = np.array([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad)
        ])
        normal /= np.linalg.norm(normal)
        
        # Get centroids
        c1 = refine_centroid(sample1)
        c2 = refine_centroid(sample2)
        
        # Calculate midpoint
        midpoint = (c1 + c2) / 2
        
        # Calculate centroid distance for scaling
        delta = c2 - c1
        delta_norm = np.linalg.norm(delta)
        
        # Compute dot products once (optimization)
        all_dot_products = np.dot(sample2 - midpoint, normal)
        
        # Target number of points to include
        target_count = int(len(sample2) * target_coverage / 100)
        
        # Sort dot products for binary search
        sorted_dots = np.sort(all_dot_products)
        
        # We want target_count points inside, which means we need the
        # threshold to be at the appropriate percentile
        if target_count > 0 and target_count <= len(sorted_dots):
            idx = len(sorted_dots) - target_count
            idx = max(0, min(idx, len(sorted_dots) - 1))
            threshold = sorted_dots[idx]
            
            # Convert threshold to sensitivity
            # sensitivity_adjusted = (slider_val - 200) / 100.0
            raw_sensitivity = threshold / np.dot(normal, normal)
            slider_value = 200 + raw_sensitivity * 100 / delta_norm
            
            # Clip to valid range
            slider_value = max(0, min(400, slider_value))
            
            print(f"Sensitivity for {target_coverage:.1f}% coverage: {slider_value:.1f}")
            return int(slider_value)
        else:
            # Fallback if target count is invalid
            print(f"Invalid target count, using default sensitivity")
            return 200  # Default middle value

    def get_cube_faces():
        """Get the faces of the LAB cube"""
        faces = []
        for axis in range(3):
            for const in [0, 255]:
                p = [np.array([0, 0, 0], dtype=float) for _ in range(4)]
                idx = [(0, 0), (1, 0), (1, 1), (0, 1)]
                for i, (a, b) in enumerate(idx):
                    p[i][axis] = const
                    p[i][(axis + 1) % 3] = 255 * a
                    p[i][(axis + 2) % 3] = 255 * b
                faces.append(p)
        return faces

    def clip_face_against_plane(face_pts, plane_point, plane_normal, tol=1e-5):
        """Clip a face against a plane"""
        def is_inside(p):
            return np.dot(p - plane_point, plane_normal) <= tol
        def intersect(p1, p2):
            d1 = np.dot(p1 - plane_point, plane_normal)
            d2 = np.dot(p2 - plane_point, plane_normal)
            if abs(d1 - d2) < tol:
                return p1
            t = d1 / (d1 - d2)
            return p1 + t * (p2 - p1)
        clipped = []
        for i in range(4):
            curr = face_pts[i]
            prev = face_pts[i - 1]
            curr_in = is_inside(curr)
            prev_in = is_inside(prev)
            if curr_in:
                if not prev_in:
                    clipped.append(intersect(prev, curr))
                clipped.append(curr)
            elif prev_in:
                clipped.append(intersect(prev, curr))
        return clipped if len(clipped) >= 3 else None

    def make_valid_mask(img, sample_name):
        """Create valid pixel mask for an image"""
        white_mask = np.all(img == [255, 255, 255], axis=-1)
        border_size = border_sizes.get(sample_name, 1)
        valid_region = ~white_mask
        for _ in range(border_size):
            valid_region = binary_erosion(valid_region)
        border_mask = (~white_mask) & (~valid_region)
        return ~white_mask & ~border_mask

    # MAIN CODE FLOW STARTS HERE
    
    # Check if we have stored RGB data from sample analysis
    if "Sample 1" in sample_extracted_data and "Sample 2" in sample_extracted_data:
        print(f"Using stored data from Sample Analysis:")
        print(f"Sample 1: {len(sample_extracted_data['Sample 1'])} pixels")
        print(f"Sample 2: {len(sample_extracted_data['Sample 2'])} pixels")
        
        # Process stored RGB data to LAB format
        sample1 = process_stored_rgb(sample_extracted_data["Sample 1"])
        sample2 = process_stored_rgb(sample_extracted_data["Sample 2"])
        
        # Load images for display (these aren't used for analysis)
        sample1_img = load_image("Sample 1")
        sample2_img = load_image("Sample 2")
        
        # Process Sample 3 if available
        if "Sample 3" in sample_extracted_data:
            sample3_rgb = sample_extracted_data["Sample 3"]
            sample3 = process_stored_rgb(sample3_rgb)
            print(f"Sample 3: {len(sample_extracted_data['Sample 3'])} pixels")
            sample3_img = load_image("Sample 3")
        else:
            # Check if Sample 3 images exist even if no analysis data
            if "Sample 3" in file_paths and file_paths["Sample 3"]:
                sample3_img = load_image("Sample 3")
                sample3 = np.array([])  # Empty if no analysis data
            else:
                sample3_img = np.zeros_like(sample2_img)
                sample3 = np.array([])
    else:
        # No stored analysis data, fall back to loading data directly
        print("No stored data from Sample Analysis, using default loading method")
        
        # Load data and images for all samples
        sample1 = get_lab_data("Sample 1")
        sample1_img = load_image("Sample 1")
        
        sample2 = get_lab_data("Sample 2")
        sample2_img = load_image("Sample 2")
        
        # Handle Sample 3 if available
        if "Sample 3" in file_paths and file_paths["Sample 3"]:
            sample3 = get_lab_data("Sample 3")
            sample3_img = load_image("Sample 3")
        else:
            sample3 = np.array([])
            sample3_img = np.zeros_like(sample2_img)

    # Create valid masks for Sample 2 and 3
    valid_mask_sample2 = make_valid_mask(sample2_img, "Sample 2")
    valid_mask_sample3 = make_valid_mask(sample3_img, "Sample 3")

    # Setup GUI state and defaults
    selected_pixels2 = {}
    selected_pixels3 = {}
    
    # Sample-specific state dictionaries
    sample2_state = {
        'manual_override': False,
        'use_all_pixels': True,
        'azimuth': 0,
        'elevation': 90,
        'sensitivity': 200,
        'calculated': False  # Flag to indicate if auto-calculation has been done
    }

    sample3_state = {
        'manual_override': False,
        'use_all_pixels': True,
        'azimuth': 0,
        'elevation': 90,
        'sensitivity': 200,
        'calculated': False  # Flag to indicate if auto-calculation has been done
    }

    # Current active state (references to avoid duplication)
    current_state = sample2_state
    manual_override = [current_state['manual_override']]
    use_all_pixels = [current_state['use_all_pixels']]
    calculating = [False]
    selected_sample = ["Sample 2"]
    
    def get_current_sample2_points():
        if selected_sample[0] == "Sample 2":
            if use_all_pixels[0]:
                return sample2
            else:
                selected_lab_points = [data['lab'] for data in selected_pixels2.values()]
        else:
            if use_all_pixels[0]:
                return sample3
            else:
                selected_lab_points = [data['lab'] for data in selected_pixels3.values()]

        if len(selected_lab_points) < 3:
            return None
        return np.array(selected_lab_points)
    
    def get_selected_pixels():
        return selected_pixels2 if selected_sample[0] == "Sample 2" else selected_pixels3

    def update_sample1_with_plane_overlay(ax, midpoint, normal):
        """
        Function to update Sample 1 images with orange overlay for pixels inside the plane.
        This displays all loaded Sample 1 images in a grid in figure 1.
        
        Args:
            ax: The matplotlib axis for Sample 1 (ax_img1)
            midpoint: The midpoint of the plane
            normal: The normal vector of the plane
        """
        # Clear current image
        ax.clear()
        
        # Check if we have Sample 1 images
        if "Sample 1" not in file_paths or not file_paths["Sample 1"]:
            ax.set_title("No Sample 1 images loaded")
            ax.axis("off")
            return
        
        # Get all Sample 1 image files
        sample1_files = file_paths["Sample 1"]
        
        # If there's only one image, display it normally
        if len(sample1_files) == 1:
            # Create a copy of Sample 1 image for overlay
            overlay_img = sample1_img.copy()
            
            # Process and display with overlay
            overlay_img = apply_overlay_to_image(overlay_img, "Sample 1", midpoint, normal)
            
            # Display the image
            ax.imshow(overlay_img)
            ax.set_title("Sample 1 (Orange: Inside Region)")
            ax.axis("off")
            return
        
        # For multiple images, we'll create a subplot grid within the existing axis
        # Determine grid dimensions
        n_images = len(sample1_files)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        # Create a figure-level suptitle
        ax.set_title("Sample 1 Images (Orange: Inside Region)", pad=20)
        
        # Remove regular axes
        ax.axis("off")
        
        # Create a figure within the axis
        fig_inner = plt.figure(figsize=(8, 8))
        
        # Process each Sample 1 image
        for i, filepath in enumerate(sample1_files):
            if i >= grid_size * grid_size:  # Limit to grid capacity
                break
                
            # Load image
            try:
                if filepath.lower().endswith(('.tif', '.tiff')):
                    with tifffile.TiffFile(filepath) as tf:
                        arr = tf.asarray()
                        img = arr[0]
                elif filepath.lower().endswith(('.png', '.bmp')):
                    img = np.array(Image.open(filepath).convert("RGB"))
                else:
                    continue
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
                continue
            
            # Apply overlay
            overlay_img = apply_overlay_to_image(img.copy(), "Sample 1", midpoint, normal)
            
            # Add to grid
            plt.subplot(grid_size, grid_size, i+1)
            plt.imshow(overlay_img)
            plt.title(f"Image {i+1}", fontsize=8)
            plt.axis("off")
        
        # Adjust layout
        plt.tight_layout()
        
        # Draw the inner figure on the original axis
        ax.figure.canvas.draw()
        
        # Get the renderer
        renderer = ax.figure.canvas.renderer
        
        # Draw the inner figure to a temporary canvas
        fig_inner.canvas.draw()
        
        # Convert to image
        buf = fig_inner.canvas.buffer_rgba()
        image_array = np.frombuffer(buf, dtype=np.uint8).reshape(
            fig_inner.canvas.get_width_height()[::-1] + (4,))
        
        # Close inner figure to avoid memory leaks
        plt.close(fig_inner)
        
        # Display the image array on the original axis
        ax.imshow(image_array)
        

    def apply_overlay_to_image(img, sample_name, midpoint, normal):
        """
        Helper function to apply the appropriate overlay to an image.
        For Sample 1, it applies orange overlay to pixels inside the plane.
        For other samples, it first marks used pixels as green, then ANY pixels
        inside the plane as red. Properly excludes background and eroded border pixels.
        
        Args:
            img: The image to overlay
            sample_name: The name of the sample (e.g., "Sample 1")
            midpoint: The midpoint of the plane
            normal: The normal vector of the plane
                
        Returns:
            The image with overlay applied
        """
        # Create a copy of the image for overlay
        overlay_img = img.copy()
        
        # Create masks for background and borders
        white_mask = np.all(img == [255, 255, 255], axis=-1)
        border_size = border_sizes.get(sample_name, 1)
        valid_region = ~white_mask
        for _ in range(border_size):
            valid_region = binary_erosion(valid_region)
        
        # Border mask includes eroded border pixels
        border_mask = (~white_mask) & (~valid_region)
        
        # Valid pixels exclude both background and border
        valid_pixels = ~white_mask & ~border_mask
        
        # Create empty mask for highlighting
        inside_region_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        used_for_calc_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        
        # Get the coordinates of valid pixels
        y_indices, x_indices = np.where(valid_pixels)
        
        # Mark pixels used in calculation
        if sample_name in sample_extracted_data:
            sample_rgb_values = sample_extracted_data[sample_name]
            
            # Create lookup dictionary
            rgb_lookup = {}
            for i, rgb in enumerate(sample_rgb_values):
                key = f"{rgb[0]},{rgb[1]},{rgb[2]}"
                if key not in rgb_lookup:
                    rgb_lookup[key] = []
                rgb_lookup[key].append(i)
            
            # Mark pixels used in calculation (only within valid pixels)
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                rgb = img[y, x]
                key = f"{rgb[0]},{rgb[1]},{rgb[2]}"
                if key in rgb_lookup:
                    used_for_calc_mask[y, x] = True
        
        # Now check ALL valid pixels to see if they fall inside the plane
        # This automatically excludes border and background
        for i in range(len(y_indices)):
            y, x = y_indices[i], x_indices[i]
            rgb = img[y, x]
            
            # Convert this pixel to LAB
            lab_pixel = custom_rgb_to_lab(np.array([[[rgb[0], rgb[1], rgb[2]]]])).flatten()
            
            # Check if this pixel falls inside the plane
            if np.dot(lab_pixel - midpoint, normal) <= 0:
                inside_region_mask[y, x] = True
        
        # First mark background and borders appropriately
        overlay_img[white_mask] = [0, 0, 255]  # Blue for background
        overlay_img[border_mask] = [255, 0, 0]  # Red for border
        
        # Then apply highlighting based on sample type
        if sample_name == "Sample 1":
            # For Sample 1, just highlight inside pixels orange
            overlay_img[inside_region_mask] = [255, 165, 0]  # Orange
        else:
            # For other samples:
            # First mark calculation pixels as green
            overlay_img[used_for_calc_mask] = [0, 255, 0]  # Green
            
            # Then mark inside pixels as red (overwrites green if inside)
            overlay_img[inside_region_mask] = [255, 0, 0]  # Red
        
        return overlay_img

    def update_image_with_plane_overlay(ax, img, points, midpoint, normal):
        """
        Create a dual-color overlay showing:
        - Green: Pixels that were extracted during sample analysis (excluding borders)
        - Red: ANY valid pixels that fall inside the shaded region (excluding borders)
        
        Args:
            ax: The matplotlib axis to update
            img: The original RGB image
            points: The 3D points that were used in calculation (already in LAB space)
            midpoint: The midpoint of the plane
            normal: The normal vector of the plane
        """
        # Clear current image
        ax.clear()
        
        # Get the current selected sample
        current_sample_name = selected_sample[0]
        
        # Create a copy of the image for overlay
        overlay_img = img.copy()
        
        # Create masks for background and borders exactly as in analyze_sample
        white_mask = np.all(img == [255, 255, 255], axis=-1)
        border_size = border_sizes.get(current_sample_name, 1)
        valid_region = ~white_mask
        for _ in range(border_size):
            valid_region = binary_erosion(valid_region)
        
        # Border mask includes white background and eroded borders
        # valid_pixels are ONLY the pixels neither in background nor in border
        border_mask = (~white_mask) & (~valid_region)
        valid_pixels = ~white_mask & ~border_mask
        
        # Create empty masks for highlighting
        used_in_calc_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        inside_region_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        
        # First, identify pixels that were used in the calculation (for green highlighting)
        if current_sample_name in sample_extracted_data:
            # Get the stored RGB values for this sample
            sample_rgb_values = sample_extracted_data[current_sample_name]
            
            # Create a dictionary for fast lookup of RGB values
            rgb_lookup = {}
            for i, rgb in enumerate(sample_rgb_values):
                # Convert RGB tuple to string key
                key = f"{rgb[0]},{rgb[1]},{rgb[2]}"
                if key not in rgb_lookup:
                    rgb_lookup[key] = []
                rgb_lookup[key].append(i)
            
            # Mark pixels used in calculation (for green highlight)
            y_indices, x_indices = np.where(valid_pixels)
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                rgb = img[y, x]
                key = f"{rgb[0]},{rgb[1]},{rgb[2]}"
                
                # Check if this RGB value is in our extracted data
                if key in rgb_lookup:
                    # This pixel was used in the sample analysis
                    used_in_calc_mask[y, x] = True
        
        # Only check VALID pixels (this excludes background and eroded borders)
        y_indices, x_indices = np.where(valid_pixels)
        for i in range(len(y_indices)):
            y, x = y_indices[i], x_indices[i]
            rgb = img[y, x]
            
            # Convert this pixel to LAB
            lab_pixel = custom_rgb_to_lab(np.array([[[rgb[0], rgb[1], rgb[2]]]])).flatten()
            
            # Check if this pixel falls inside the plane
            if np.dot(lab_pixel - midpoint, normal) <= 0:
                inside_region_mask[y, x] = True
        
        # Mark background and border regions with appropriate colors
        overlay_img[white_mask] = [0, 0, 255]  # Blue for background
        overlay_img[border_mask] = [255, 0, 0]  # Red for border
        
        # Apply the dual-color highlighting ONLY to valid pixels:
        # First green for all pixels used in sample analysis
        overlay_img[used_in_calc_mask] = [0, 255, 0]  # Green
        
        # Then red for ALL pixels inside the plane (not just those used in calculation)
        # This won't affect border areas as they're not in the inside_region_mask
        overlay_img[inside_region_mask] = [255, 0, 0]  # Red
        
        # Display the image
        ax.imshow(overlay_img)
        title = f"{current_sample_name} (Green: Used in calc, Red: Inside plane)"
        ax.set_title(title)
        ax.axis("off")

    def toggle_manual_control(label):
        if label == 'Manual Control':
            # Update current state
            manual_override[0] = check.get_status()[0]
            current_state = sample3_state if selected_sample[0] == "Sample 3" else sample2_state
            current_state['manual_override'] = manual_override[0]
            
            # Enable or disable sliders based on manual control status
            if manual_override[0]:
                # Entering manual mode - keep current slider values
                slider_sens.set_active(True)
                slider_azim.set_active(True)
                slider_elev.set_active(True)
            else:
                # Exiting manual mode - restore calculated values if available
                slider_sens.set_active(False)
                slider_azim.set_active(False)
                slider_elev.set_active(False)
                
                # Use cached values if they exist, otherwise calculate
                if current_state['calculated']:
                    # Restore the calculated values without triggering events
                    slider_azim.eventson = False
                    slider_elev.eventson = False
                    slider_sens.eventson = False
                    
                    slider_azim.set_val(current_state['azimuth'])
                    slider_elev.set_val(current_state['elevation'])
                    slider_sens.set_val(current_state['sensitivity'])
                    
                    slider_azim.eventson = True
                    slider_elev.eventson = True
                    slider_sens.eventson = True
                    
                    redraw_cut_plane()
                elif use_all_pixels[0]:
                    # Only auto-calculate if no calculated values exist
                    auto_calculate(None)
            
            fig.canvas.draw_idle()

    def toggle_all_pixels(label):
        if label == 'Use All Pixels':
            # Update current state
            use_all_pixels[0] = check_all.get_status()[0]
            current_state = sample3_state if selected_sample[0] == "Sample 3" else sample2_state
            current_state['use_all_pixels'] = use_all_pixels[0]
            
            # Auto-calculate if switching to all pixels and not in manual mode
            if use_all_pixels[0] and not manual_override[0]:
                auto_calculate(None)
            else:
                # If we're turning off "Use All Pixels", clear the image overlay
                # and update the plot to show only selected pixels
                if not use_all_pixels[0]:
                    # Reset the image without overlay
                    if selected_sample[0] == "Sample 3":
                        ax_img2.clear()
                        ax_img2.imshow(sample3_img)
                        ax_img2.set_title("Select Pixels on Sample 3")
                    else:
                        ax_img2.clear()
                        ax_img2.imshow(sample2_img)
                        ax_img2.set_title("Select Pixels on Sample 2")
                    ax_img2.axis("off")
                    
                    # Update selected pixels display
                    update_selected_pixel_display()
            
            redraw_cut_plane()

    

    def toggle_sample(label):
        if label == 'Select Sample 3':
            # Get current state before switching
            prev_sample = selected_sample[0]
            prev_state = sample2_state if prev_sample == "Sample 2" else sample3_state
            
            # Save current UI state to the previous sample's state
            prev_state['manual_override'] = manual_override[0]
            prev_state['use_all_pixels'] = use_all_pixels[0]
            prev_state['azimuth'] = slider_azim.val
            
            # CRITICAL: Only update elevation if manual override is on
            if manual_override[0]:
                prev_state['elevation'] = slider_elev.val
            
            prev_state['sensitivity'] = slider_sens.val
            
            # Update selected sample
            selected_sample[0] = "Sample 3" if check_sample_toggle.get_status()[0] else "Sample 2"
            
            # Switch to new sample state
            new_state = sample3_state if selected_sample[0] == "Sample 3" else sample2_state
            
            # Update references
            manual_override[0] = new_state['manual_override']
            use_all_pixels[0] = new_state['use_all_pixels']
            
            # Debug print to verify values
            print(f"\nSwitching to {selected_sample[0]}")
            print("New state values:")
            print(f"Azimuth: {new_state['azimuth']}")
            print(f"Elevation: {new_state['elevation']}")
            print(f"Sensitivity: {new_state['sensitivity']}")
            
            # Update sliders (without triggering callbacks)
            slider_azim.eventson = False
            slider_elev.eventson = False
            slider_sens.eventson = False
            
            slider_azim.set_val(new_state['azimuth'])
            slider_elev.set_val(new_state['elevation'])
            slider_sens.set_val(new_state['sensitivity'])
            
            slider_azim.eventson = True
            slider_elev.eventson = True
            slider_sens.eventson = True
            
            # Update slider active state based on manual override
            slider_azim.set_active(manual_override[0])
            slider_elev.set_active(manual_override[0])
            slider_sens.set_active(manual_override[0])
            
            # Update image display
            if selected_sample[0] == "Sample 3":
                ax_img2.clear()
                ax_img2.imshow(sample3_img)
                ax_img2.set_title("Select Pixels on Sample 3")
                ax_img2.axis("off")
            else:
                ax_img2.clear()
                ax_img2.imshow(sample2_img)
                ax_img2.set_title("Select Pixels on Sample 2")
                ax_img2.axis("off")
            
            # Update selected pixels display
            update_selected_pixel_display()
            
            # Update the plot with the new sample data
            redraw_cut_plane()
            
            # Update title bar with the new sample's saved values
            ax_3d.set_title(
                f"Optimized (Az: {new_state['azimuth']:.1f}°, El: {new_state['elevation']:.1f}°, Sens: {new_state['sensitivity']:.1f})"
            )
            
            # Re-enable pixel selection
            setup_pixel_selection()
            
            # Auto calculate if needed (first time viewing this sample with all pixels)
            if use_all_pixels[0] and not manual_override[0] and not new_state['calculated']:
                auto_calculate(None)
            
            fig.canvas.draw_idle()

    def setup_pixel_selection():
        # Disconnect any existing events
        if hasattr(fig, 'pixel_selection_cid'):
            fig.canvas.mpl_disconnect(fig.pixel_selection_cid)
        
        # Connect new event handler
        fig.pixel_selection_cid = fig.canvas.mpl_connect('button_press_event', on_click)

    def on_click(event):
        if event.inaxes != ax_img2 or use_all_pixels[0]:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        # Determine which mask to use based on selected sample
        if selected_sample[0] == "Sample 2":
            valid_mask = valid_mask_sample2
            img = sample2_img
            pixels_dict = selected_pixels2
        else:  # Sample 3
            valid_mask = valid_mask_sample3
            img = sample3_img
            pixels_dict = selected_pixels3
        
        if y < img.shape[0] and x < img.shape[1] and valid_mask[y, x]:
            key = f"{x},{y}"
            if key in pixels_dict:
                # If pixel already selected, remove it
                del pixels_dict[key]
            else:
                # Add pixel to selected points
                rgb = img[y, x]
                lab = custom_rgb_to_lab(np.array([[[rgb[0], rgb[1], rgb[2]]]]))
                lab_flat = lab[0, 0]
                pixels_dict[key] = {
                    'xy': (x, y),
                    'rgb': rgb,
                    'lab': [lab_flat[0], lab_flat[1], lab_flat[2]]
                }
        
        # Update display of selected pixels
        update_selected_pixel_display()
        
        # Update 3D plot if in manual selection mode
        if not use_all_pixels[0]:
            redraw_cut_plane()

    def update_selected_pixel_display():
        # Clear previous pixel markers
        for artist in ax_img2.collections:
            if hasattr(artist, '_is_pixel_marker'):
                artist.remove()
        
        # Get active selection dictionary
        pixels_dict = selected_pixels2 if selected_sample[0] == "Sample 2" else selected_pixels3
        
        # Plot markers for selected pixels
        xy_list = [data['xy'] for data in pixels_dict.values()]
        if xy_list:
            scatter = ax_img2.scatter(
                [p[0] for p in xy_list], 
                [p[1] for p in xy_list], 
                color='cyan', 
                s=30, 
                marker='o'
            )
            scatter._is_pixel_marker = True
        
        fig.canvas.draw_idle()

    
    def auto_calculate(event):
        if calculating[0]:
            return

        calculating[0] = True
        ax_3d.set_title("Calculating optimal plane... Please wait")
        fig.canvas.draw_idle()

        points = get_current_sample2_points()

        if points is None or len(points) < 3:
            ax_3d.set_title("Not enough points selected")
            calculating[0] = False
            fig.canvas.draw_idle()
            return

        try:
            # Start timing the entire calculation process
            import time
            total_start_time = time.time()
            
            # Set the random seed for reproducibility
            np.random.seed(42)
            
            # Get optimal plane parameters with iterative optimization
            # Target ~20 points initially, then 90% coverage for final sensitivity
            optimal_params = optimize_separation_plane(sample1, points, 
                                                    target_initial_count=20, 
                                                    target_final_coverage=90.0)
            
            # Extract parameters
            azimuth = optimal_params[0]
            elevation = optimal_params[1]
            sensitivity = optimal_params[2]
            
            optimize_duration = time.time() - total_start_time
            optimal_sens = find_sensitivity_for_coverage(sample1, points, target_coverage=80.0)
            # Determine the current sample's state
            current_state = sample3_state if selected_sample[0] == "Sample 3" else sample2_state

            # Update sliders without triggering callbacks
            slider_azim.eventson = False
            slider_elev.eventson = False
            slider_sens.eventson = False

            # Set slider values
            slider_azim.set_val(azimuth)
            elevation_slider_val = elevation + 90
            slider_elev.set_val(elevation_slider_val) 
            slider_sens.set_val(optimal_sens)

            slider_azim.eventson = True
            slider_elev.eventson = True
            slider_sens.eventson = True

            # Store calculated values in the current sample's state
            current_state['azimuth'] = azimuth
            current_state['elevation'] = elevation_slider_val
            current_state['sensitivity'] = optimal_sens
            current_state['calculated'] = True

            # Now redraw the cut plane with the newly calculated parameters
            redraw_cut_plane()

            # Update title bar with final values and computation time
            ax_3d.set_title(
                f"Optimized (Sens: {sensitivity}, Az: {azimuth:.1f}°, El: {elevation_slider_val:.1f}°, Time: {optimize_duration:.2f}s)"
            )

        except Exception as e:
            ax_3d.set_title(f"Error: {str(e)}")
            print(f"Error in auto_calculate: {e}")
            import traceback
            traceback.print_exc()

        calculating[0] = False
        fig.canvas.draw_idle()

    def redraw_cut_plane(*_):
        nonlocal plane_patches
        for p in plane_patches:
            p.remove()
        plane_patches.clear()

        points = get_current_sample2_points()
        if points is None or len(points) < 3:
            scatter2._offsets3d = ([], [], [])
            fig.canvas.draw_idle()
            return

        # Update legend
        active_sel = get_selected_pixels()
        if use_all_pixels[0]:
            legend_text.set_text(f"Selected: ALL")
        else:
            legend_text.set_text(f"Selected: {len(active_sel)}")

        # Get the azimuth and elevation from sliders if in manual mode
        if manual_override[0]:
            azimuth = np.deg2rad(slider_azim.val)
            elevation = np.deg2rad(slider_elev.val - 90)
            # Calculate normal vector from azimuth and elevation
            dx = np.cos(elevation) * np.cos(azimuth)
            dy = np.cos(elevation) * np.sin(azimuth)
            dz = np.sin(elevation)
            normal = np.array([dx, dy, dz])
        else:
            # In automatic mode, use the values from auto-calculate
            current_state = sample3_state if selected_sample[0] == "Sample 3" else sample2_state
            
            azimuth = np.deg2rad(current_state['azimuth'])
            elevation = np.deg2rad(current_state['elevation'] - 90)
            
            # Calculate normal vector from azimuth and elevation
            dx = np.cos(elevation) * np.cos(azimuth)
            dy = np.cos(elevation) * np.sin(azimuth)
            dz = np.sin(elevation)
            normal = np.array([dx, dy, dz])
                
            # Update sliders without triggering callbacks (for display only)
            slider_azim.eventson = False
            slider_elev.eventson = False
            
            slider_azim.set_val(np.degrees(azimuth))
            slider_elev.set_val(np.degrees(elevation) + 90)
            
            slider_azim.eventson = True
            slider_elev.eventson = True

        # Get centroids for both samples
        c1 = refine_centroid(sample1)
        c2 = refine_centroid(points)
        
        # Calculate midpoint between centroids
        midpoint = (c1 + c2) / 2

        # Ensure normal points from good to defect
        if np.dot(c2 - midpoint, normal) > 0:
            normal = -normal

        # Apply sensitivity to move the plane along the normal
        # Convert sensitivity slider (0-400) to a normalized range
        sensitivity_factor = (slider_sens.val - 200) / 100.0
        sensitivity_distance = sensitivity_factor * np.linalg.norm(c2 - c1)
        midpoint += normal * sensitivity_distance

        # Clear previous overlays
        for artist in ax_3d.collections[:]:
            if getattr(artist, '_is_sample1', False) or getattr(artist, '_is_defect', False):
                artist.remove()

        # Sample 1 (Good sample)
        dot_products_good = np.dot(sample1 - midpoint, normal)
        sample1_outside = sample1[dot_products_good > 0]  # Good outside (visible)
        sample1_inside = sample1[dot_products_good <= 0]  # Good inside

        # Sample 2/3 (Defect sample)
        dot_products_defect = np.dot(points - midpoint, normal)
        points_outside = points[dot_products_defect > 0]  # Defect outside
        points_inside = points[dot_products_defect <= 0]  # Defect inside (shaded)

        # Plot Sample 1 points
        scatter_good_outside = ax_3d.scatter(
            sample1_outside[:, 0], sample1_outside[:, 1], sample1_outside[:, 2],
            color='green', alpha=0.4, s=10
        )
        scatter_good_outside._is_sample1 = True

        scatter_good_inside = ax_3d.scatter(
            sample1_inside[:, 0], sample1_inside[:, 1], sample1_inside[:, 2],
            color='orange', edgecolors='orange', alpha=0.9, s=10
        )
        scatter_good_inside._is_sample1 = True
        
        # Plot Sample 2/3 inside points with different color to highlight them
        scatter_color = 'red' if selected_sample[0] == "Sample 2" else 'magenta'
        scatter_defect_inside = ax_3d.scatter(
            points_inside[:, 0], points_inside[:, 1], points_inside[:, 2],
            color=scatter_color, alpha=0.9, s=20
        )
        scatter_defect_inside._is_defect = True

        # Shaded cut plane
        shade_color = 'red' if selected_sample[0] == "Sample 2" else 'grey'
        for face in get_cube_faces():
            clipped = clip_face_against_plane(face, midpoint, normal)
            if clipped:
                patch = Poly3DCollection([clipped], facecolor=shade_color, alpha=0.2, edgecolor='black')
                ax_3d.add_collection3d(patch)
                plane_patches.append(patch)

        # Plot active sample points
        scatter2._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        scatter2.set_color(scatter_color)

        # Count inside region and calculate percentages
        bad_inside = np.sum(dot_products_defect <= 0)
        good_inside = np.sum(dot_products_good <= 0)
        
        bad_inside_pct = bad_inside / len(points) * 100
        good_inside_pct = good_inside / len(sample1) * 100

        # Update bar chart
        ax_bar.clear()
        bars = ax_bar.bar(["Defect", "Good"], [bad_inside, good_inside], color=[scatter_color, "orange"])
        ax_bar.set_ylim(0, (bad_inside + good_inside) * 1.1)
        ax_bar.set_ylabel("Count")
        ax_bar.set_title(f"Points Inside (Defect: {bad_inside_pct:.1f}%, Good: {good_inside_pct:.1f}%)")

        threshold = len(sample2) if selected_sample[0] == "Sample 2" else len(sample3)
        if not use_all_pixels[0]:
            threshold = len(get_selected_pixels())
        bar_center = bars[0].get_x() + bars[0].get_width() / 2
        ax_bar.plot([bar_center - 0.15, bar_center + 0.15], [threshold, threshold], 'k-', linewidth=2)

        # Sensitivity direction line
        if direction_line[0] is not None:
            direction_line[0].remove()
            direction_line[0] = None
        if check_show_dir.get_status()[0]:
            direction_line[0] = ax_3d.plot(
                [c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                color='blue', linewidth=2, label="Sens Direction"
            )[0]
        
        # Update the Sample 1 image with orange overlay
        update_sample1_with_plane_overlay(ax_img1, midpoint, normal)
        
        # Update the active sample image overlay
        if selected_sample[0] == "Sample 2":
            update_image_with_plane_overlay(ax_img2, sample2_img, points, midpoint, normal)
        else:  # Sample 3
            update_image_with_plane_overlay(ax_img2, sample3_img, points, midpoint, normal)
        
        fig.canvas.draw_idle()

    # Handle slider changes
    def on_slider_changed(*args):
        current_state = sample3_state if selected_sample[0] == "Sample 3" else sample2_state
        current_state['azimuth'] = slider_azim.val
        
        # Only update elevation if in manual override mode
        if manual_override[0]:
            current_state['elevation'] = slider_elev.val
        
        current_state['sensitivity'] = slider_sens.val
        redraw_cut_plane()

    # Create the figure and subplots
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[0.25, 2, 1, 0.3])
    ax_bar = fig.add_subplot(gs[:, 0])
    ax_bar.set_position([0.01, 0.1, 0.15, 0.8])  # [left, bottom, width, height]
    ax_3d = fig.add_subplot(gs[:, 1], projection='3d')
    ax_img1 = fig.add_subplot(gs[0, 2])
    ax_img2 = fig.add_subplot(gs[1, 2])
    ax_legend = fig.add_subplot(gs[1, 3])

    ax_show_dir = plt.axes([0.14, 0.90, 0.12, 0.05])
    check_show_dir = CheckButtons(ax_show_dir, ['Show Sens Direction'], [False])
    direction_line = [None]

    ax_sample_toggle = plt.axes([0.8, 0.90, 0.12, 0.05])
    check_sample_toggle = CheckButtons(ax_sample_toggle, ['Select Sample 3'], [False])

    ax_img1.imshow(sample1_img)
    ax_img1.set_title("Sample 1 Image")
    ax_img1.axis("off")

    ax_img2.imshow(sample2_img)
    ax_img2.set_title("Select Pixels on Sample 2")
    ax_img2.axis("off")

    ax_3d.set_box_aspect([1, 1, 1])
    ax_3d.set_xlim(0, 255)
    ax_3d.set_ylim(0, 255)
    ax_3d.set_zlim(0, 255)
    ax_3d.set_xlabel("L")
    ax_3d.set_ylabel("A")
    ax_3d.set_zlabel("B")
    ax_3d.set_title("Dynamic LAB Cube")

    ax_3d.scatter(sample1[:, 0], sample1[:, 1], sample1[:, 2], color='green', alpha=0.4, s=10, label="Sample 1")
    scatter2 = ax_3d.scatter([], [], [], color='red', alpha=0.8, s=20, label="Selected Defect")
    plane_patches = []

    # Slider and widget layout
    ax_sens = plt.axes([0.2, 0.06, 0.2, 0.03])
    ax_azim = plt.axes([0.2, 0.03, 0.2, 0.03])
    ax_elev = plt.axes([0.2, 0.005, 0.2, 0.03])
    ax_check = plt.axes([0.05, 0.02, 0.1, 0.05])
    ax_use_all = plt.axes([0.6, 0.03, 0.1, 0.05])
    ax_auto_calc = plt.axes([0.75, 0.04, 0.1, 0.03])

    ax_sens_plus = plt.axes([0.45, 0.06, 0.03, 0.03])
    ax_sens_minus = plt.axes([0.15, 0.06, 0.03, 0.03])

    button_sens_plus = Button(ax_sens_plus, '+')
    button_sens_minus = Button(ax_sens_minus, '-')

    button_sens_plus.on_clicked(increment_sensitivity)
    button_sens_minus.on_clicked(decrement_sensitivity)

    slider_sens = Slider(ax_sens, 'Sensitivity', 0, 400, valinit=200, valstep=1)
    slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=0)
    slider_elev = Slider(ax_elev, 'Elevation', 0, 180, valinit=90)

    check = CheckButtons(ax_check, ['Manual Control'], [False])
    check_all = CheckButtons(ax_use_all, ['Use All Pixels'], [True])
    button_auto_calc = Button(ax_auto_calc, 'Auto Calculate')

    slider_sens.set_active(False)
    slider_azim.set_active(False)
    slider_elev.set_active(False)

    legend_text = ax_legend.text(0.5, 0.5, "Selected: 0", ha='center', va='center', fontsize=12)
    ax_legend.axis('off')

    # Connect event handlers
    button_auto_calc.on_clicked(auto_calculate)
    slider_sens.on_changed(on_slider_changed)
    slider_azim.on_changed(on_slider_changed)
    slider_elev.on_changed(on_slider_changed)
    check_show_dir.on_clicked(redraw_cut_plane)
    check_all.on_clicked(toggle_all_pixels)
    check_sample_toggle.on_clicked(toggle_sample)
    check.on_clicked(toggle_manual_control)

    # Initial drawing of all pixels
    points = sample2  # initially show sample2
    scatter2._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    # Setup initial pixel selection
    setup_pixel_selection()

    # Calculate initial plane for Sample 2
    if use_all_pixels[0]:
        auto_calculate(None)
        
        # Then switch to Sample 3 temporarily, calculate, and switch back
        selected_sample[0] = "Sample 3"
        auto_calculate(None)
        selected_sample[0] = "Sample 2"
        
        # Make sure to restore Sample 2 display and sliders
        slider_azim.set_val(sample2_state['azimuth'])
        slider_elev.set_val(sample2_state['elevation']) 
        slider_sens.set_val(sample2_state['sensitivity'])
        redraw_cut_plane()

    plt.subplots_adjust(top=0.95)
    plt.show()

#def update_sensitivity_label():
    #"""Updates the label displaying the current anomaly sensitivity value"""
    #sensitivity_value_label.config(text=f"{anomaly_sensitivity:.1f}")

def increase_anomaly_sensitivity():
    global anomaly_sensitivity
    anomaly_sensitivity += 0.1
    update_sensitivity_labels()
    # Use optimized update instead of full reprocessing
    update_anomaly_masks_fast()

def decrease_anomaly_sensitivity():
    global anomaly_sensitivity
    anomaly_sensitivity = max(0.5, anomaly_sensitivity - 0.1)  # Don't go below 0.5
    update_sensitivity_labels()
    # Use optimized update instead of full reprocessing
    update_anomaly_masks_fast()



# New function to update the masks without redisplaying the entire analysis
def update_anomaly_masks():
    """
    Update anomaly masks for all samples based on current sensitivity values.
    This is a slower, more complete update function that reprocesses everything.
    """
    global sample_extracted_data, cached_reference_model
    
    # Clear existing extracted data - we'll rebuild it with the new sensitivity
    sample_extracted_data = {}
    
    # Check if Sample 1 has been loaded
    if "Sample 1" not in file_paths or not file_paths["Sample 1"]:
        messagebox.showerror("Error", "Please load Sample 1 first")
        return
    
    # Extract reference features from Sample 1
    reference_model = extract_reference_features("Sample 1")
    if not reference_model:
        messagebox.showerror("Error", "Failed to extract features from reference")
        return
    
    # Store the reference model
    cached_reference_model = reference_model
    
    print(f"Updating masks with per-sample sensitivities...")
    
    # Process Sample 1 (for reference data)
    highlight_sample("Sample 1", img_frame_s1)
    
    # Process Sample 2
    if "Sample 2" in file_paths and file_paths["Sample 2"]:
        highlight_sample("Sample 2", img_frame_s2)
    
    # Process other samples
    for sample in samples:
        sample_name = sample['name']
        if sample_name in file_paths and file_paths[sample_name]:
            highlight_sample(sample_name, sample['image_container'])
    
    print(f"Masks updated for all samples")

def process_sample_masks(sample_name, reference_model):
    if sample_name not in file_paths:
        return
    
    for filepath in file_paths[sample_name]:
        try:
            # Load image
            if filepath.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                    rgb_frame = arr[0]
            elif filepath.lower().endswith(('.png', '.bmp')):
                img = Image.open(filepath).convert("RGB")
                rgb_frame = np.array(img)
            else:
                continue
            
            # Create masks for background and borders
            white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
            border_size = border_sizes.get(sample_name, 1)
            valid_region = ~white_mask
            for _ in range(border_size):
                valid_region = binary_erosion(valid_region)
            border_mask = (~white_mask) & (~valid_region)
            valid_pixels = ~white_mask & ~border_mask
            
            # For Sample 1, store all valid pixels regardless of anomaly status
            if sample_name == "Sample 1":
                store_sample_data(sample_name, rgb_frame, np.where(valid_pixels), None)
                continue  # No need to check for anomalies in the reference sample
            
            # For other samples, detect anomalies
            # Convert to LAB for analysis
            lab = rgb2lab(rgb_frame)
            
            # Initialize anomaly mask
            anomaly_mask = np.zeros_like(white_mask)
            
            # Get reference statistics
            ref_mean = reference_model['lab_mean']
            ref_std = reference_model['lab_std'] + 1e-6
            
            # Check each valid pixel for anomalies
            y_indices, x_indices = np.where(valid_pixels)
            for i in range(len(y_indices)):
                y, x = y_indices[i], x_indices[i]
                pixel_lab = lab[y, x]
                
                # Calculate Mahalanobis-like distance
                delta = pixel_lab - ref_mean
                distance = np.sqrt(np.sum((delta / ref_std) ** 2))
                
                # Mark as anomaly if distance exceeds threshold - use global sensitivity
                if distance > anomaly_sensitivity:
                    anomaly_mask[y, x] = True
            
            # Refine with morphological operations
            anomaly_mask = binary_dilation(anomaly_mask, iterations=1)
            anomaly_mask = binary_erosion(anomaly_mask, iterations=1)
            
            # Store only the anomalous pixels for non-reference samples
            store_sample_data(sample_name, rgb_frame, np.where(valid_pixels), anomaly_mask)
            
        except Exception as e:
            print(f"Error processing masks for {filepath}: {str(e)}")      

def display_with_masks(sample_name, container):
    """Display image with background, border, and anomaly masks"""
    global anomaly_sensitivities
    
    if sample_name not in file_paths:
        return
    
    # Get sample-specific sensitivity
    sensitivity = anomaly_sensitivities.get(sample_name, 2.5)
    
    # Clear container
    for widget in container.winfo_children():
        widget.destroy()
    
    for filepath in file_paths[sample_name]:
        try:
            # Load image
            if filepath.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                    rgb_frame = arr[0]
            elif filepath.lower().endswith(('.png', '.bmp')):
                img = Image.open(filepath).convert("RGB")
                rgb_frame = np.array(img)
            else:
                continue
            
            # Create masks as before
            white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
            border_size = border_sizes.get(sample_name, 1)
            valid_region = ~white_mask
            for _ in range(border_size):
                valid_region = binary_erosion(valid_region)
            border_mask = (~white_mask) & (~valid_region)
            valid_pixels = ~white_mask & ~border_mask
            
            # Convert the background to blue for visualization
            rgb_display = rgb_frame.copy()
            rgb_display[white_mask] = [0, 0, 255]   # Blue for background
            rgb_display[border_mask] = [255, 0, 0]  # Red for border
            
            # If we have anomaly data for this sample, display anomalies
            if sample_name != "Sample 1" and sample_name in sample_extracted_data:
                # Check each valid pixel for match with sample_extracted_data
                y_indices, x_indices = np.where(valid_pixels)
                for i in range(len(y_indices)):
                    y, x = y_indices[i], x_indices[i]
                    rgb = rgb_frame[y, x]
                    
                    # Check if this RGB value exists in extracted data
                    for extracted_rgb in sample_extracted_data[sample_name]:
                        if np.array_equal(rgb, extracted_rgb):
                            rgb_display[y, x] = [255, 0, 0]  # Mark anomaly as red
                            break
            
            # Resize for display
            img_display = Image.fromarray(rgb_display).resize((100, 100))
            photo = ImageTk.PhotoImage(img_display)
            
            # Display in container
            label = Label(container, image=photo)
            label.image = photo
            label.pack(side=tk.LEFT, padx=2)
            
        except Exception as e:
            print(f"Error displaying masks for {filepath}: {str(e)}")

def display_with_masks_fast(sample_name, container):
    """Update display with current masks - optimized version with per-sample sensitivity"""
    global anomaly_sensitivities
    
    if sample_name not in file_paths:
        return
    
    # Get sample-specific sensitivity
    sensitivity = anomaly_sensitivities.get(sample_name, 2.5)
    
    # Clear container
    for widget in container.winfo_children():
        widget.destroy()
    
    for filepath in file_paths[sample_name]:
        try:
            # Load image
            if filepath.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(filepath) as tf:
                    arr = tf.asarray()
                    rgb_frame = arr[0]
            elif filepath.lower().endswith(('.png', '.bmp')):
                img = Image.open(filepath).convert("RGB")
                rgb_frame = np.array(img)
            else:
                continue
            
            # Create masks
            white_mask = np.all(rgb_frame == [255, 255, 255], axis=-1)
            border_size = border_sizes.get(sample_name, 1)
            valid_region = ~white_mask
            for _ in range(border_size):
                valid_region = binary_erosion(valid_region)
            border_mask = (~white_mask) & (~valid_region)
            valid_pixels = ~white_mask & ~border_mask
            
            # Display background in blue and border in red
            rgb_display = rgb_frame.copy()
            rgb_display[white_mask] = [0, 0, 255]  # Blue background
            rgb_display[border_mask] = [255, 0, 0]  # Red border
            
            # Create a unique key for this image
            image_key = f"{sample_name}_{filepath}"
            
            # If we have cached distances, use them to highlight anomalies
            if image_key in cached_distances:
                pixel_distances = cached_distances[image_key]
                
                # Apply current threshold to distances
                y_indices, x_indices = np.where(valid_pixels)
                for i in range(len(y_indices)):
                    y, x = y_indices[i], x_indices[i]
                    # Check if it's an anomaly based on sample-specific sensitivity
                    if not np.isnan(pixel_distances[y, x]) and pixel_distances[y, x] > sensitivity:
                        rgb_display[y, x] = [255, 0, 0]  # Red for anomaly
            
            # Resize for display
            img_display = Image.fromarray(rgb_display).resize((100, 100))
            photo = ImageTk.PhotoImage(img_display)
            
            # Display in container
            label = Label(container, image=photo)
            label.image = photo
            label.pack(side=tk.LEFT, padx=2)
            
        except Exception as e:
            print(f"Error displaying masks for {filepath}: {str(e)}")
# -------------------------------------------------------------------
# TKINTER GUI
# -------------------------------------------------------------------
root.title("Dynamic Sample Loader")
root.geometry("800x500")

# SAMPLE 1 FRAME (STATIC)
frame_s1 = Frame(root, bg="lightgreen")
frame_s1.pack(side=tk.TOP, fill=tk.X, pady=5)
btn_s1 = TkButton(frame_s1, text="Load Sample 1", command=lambda: load_sample("Sample 1", img_frame_s1))
btn_s1.pack(side=tk.LEFT, padx=5)
erosion_frame_s1 = Frame(frame_s1, bg="lightcyan")
erosion_frame_s1.pack(side=tk.LEFT, padx=5)
TkButton(erosion_frame_s1, text="Erosion +", command=lambda: increase_border("Sample 1", img_frame_s1)).pack(side=tk.TOP, pady=2)
TkButton(erosion_frame_s1, text="Erosion -", command=lambda: decrease_border("Sample 1", img_frame_s1)).pack(side=tk.TOP, pady=2)
img_frame_s1 = Frame(frame_s1, bg="pink")
img_frame_s1.pack(side=tk.LEFT, padx=5)

# Add highlight button to Sample 1
highlight_btn_s1 = TkButton(frame_s1, text="Highlight", command=lambda: highlight_sample("Sample 1", img_frame_s1))
highlight_btn_s1.pack(side=tk.LEFT, padx=5)

# SAMPLE 2 FRAME (STATIC)
frame_s2 = Frame(root, bg="lightblue")
frame_s2.pack(side=tk.TOP, fill=tk.X, pady=5)
btn_s2 = TkButton(frame_s2, text="Load Sample 2", command=lambda: load_sample("Sample 2", img_frame_s2))
btn_s2.pack(side=tk.LEFT, padx=5)
erosion_frame_s2 = Frame(frame_s2, bg="lightcyan")
erosion_frame_s2.pack(side=tk.LEFT, padx=5)
TkButton(erosion_frame_s2, text="Erosion +", command=lambda: increase_border("Sample 2", img_frame_s2)).pack(side=tk.TOP, pady=2)
TkButton(erosion_frame_s2, text="Erosion -", command=lambda: decrease_border("Sample 2", img_frame_s2)).pack(side=tk.TOP, pady=2)
img_frame_s2 = Frame(frame_s2, bg="pink")
img_frame_s2.pack(side=tk.LEFT, padx=5)

# Add highlight button to Sample 2
highlight_btn_s2 = TkButton(frame_s2, text="Highlight", command=lambda: highlight_sample("Sample 2", img_frame_s2))
highlight_btn_s2.pack(side=tk.LEFT, padx=5)

# ADD SAMPLE BUTTON AND CHECKBOXES
frame_add_sample = Frame(root, bg="lightgray")
frame_add_sample.pack(side=tk.TOP, fill=tk.X, pady=10)
btn_add_sample = TkButton(frame_add_sample, text="Add Sample", command=add_sample)
btn_add_sample.config(command=lambda: [add_sample(), create_sample_sensitivity_controls()])
btn_add_sample.pack(side=tk.LEFT, padx=5)
channel_vars = {ch: IntVar() for ch in channels}
for ch in channels:
    chk = Checkbutton(frame_add_sample, text=ch, variable=channel_vars[ch])
    chk.pack(side=tk.LEFT)

# Create a frame to hold the buttons
frame_buttons = Frame(root)
frame_buttons.pack(side=tk.TOP, pady=10)

# Process PCA Button (Left)
btn_process_pca = TkButton(frame_buttons, text="Process PCA", command=process_pca)
btn_process_pca.pack(side=tk.LEFT, padx=5)

# Process LDA Button (Middle)
btn_process_lda = TkButton(frame_buttons, text="Process LDA", command=process_lda)
btn_process_lda.pack(side=tk.LEFT, padx=5)

# Process PCA->LDA
btn_process_pca_lda = TkButton(frame_buttons, text="Process PCA->LDA", command=process_pca_lda)
btn_process_pca_lda.pack(side=tk.LEFT, padx=5)

# Analyze Dimensionality Methods Button (Right)
btn_analyze_dimensionality = TkButton(frame_buttons, text="Analyze Dimensionality Methods", command=analyze_dimensionality_methods)
btn_analyze_dimensionality.pack(side=tk.LEFT, padx=5)

# Process QDA Button (for investigating QDA without PCA)
btn_process_qda = TkButton(frame_buttons, text="Process QDA", command=process_qda)
btn_process_qda.pack(side=tk.LEFT, padx=5)

btn_process_lab = TkButton(frame_buttons, text="Process LAB Cube", command=process_lab_cube)
btn_process_lab.pack(side=tk.LEFT, padx=5)

# Add new button to execute LAB PCA processing
btn_process_lab_pca = TkButton(root, text="Process LAB PCA", command=process_lab_pca)
btn_process_lab_pca.pack(side=tk.TOP, pady=5)

btn_process_lab_plane = TkButton(root, text="LAB Plane", command=process_lab_plane)
btn_process_lab_plane.pack(side=tk.TOP, pady=5)

chk_normalize = Checkbutton(root, text="Normalize channels to [0,1]", variable=apply_normalization_var)
chk_normalize.pack(side=tk.TOP, pady=5)

btn_dynamic_plane = TkButton(frame_buttons, text="Dynamic Plane", command=launch_dynamic_plane)
btn_dynamic_plane.pack(side=tk.LEFT, padx=5, pady=5)

# Add sample analysis button to main buttons frame
btn_sample_analysis = TkButton(frame_buttons, text="Sample Analysis", command=perform_sample_analysis)
btn_sample_analysis.pack(side=tk.LEFT, padx=5)

# Create a frame for sensitivity controls
sensitivity_frame = Frame(frame_buttons, bg="lightgray", bd=1, relief=tk.RAISED)
sensitivity_frame.pack(side=tk.LEFT, padx=10, pady=5)

# Add a label for the sensitivity controls
sensitivity_label = Label(sensitivity_frame, text="Anomaly Sensitivity:")
sensitivity_label.pack(side=tk.LEFT, padx=5)

# Add decrease button
sensitivity_decrease_btn = TkButton(sensitivity_frame, text="-", width=2, 
                                command=decrease_anomaly_sensitivity)
sensitivity_decrease_btn.pack(side=tk.LEFT, padx=2)

# Add a label to display the current sensitivity value
#sensitivity_value_label = Label(sensitivity_frame, text=f"{anomaly_sensitivity:.1f}", width=4)
#sensitivity_value_label.pack(side=tk.LEFT, padx=2)

# Add increase button
sensitivity_increase_btn = TkButton(sensitivity_frame, text="+", width=2,
                                command=increase_anomaly_sensitivity)
sensitivity_increase_btn.pack(side=tk.LEFT, padx=2)

def show_results_in_tkinter(results, root):
    """
    Display the best method in a Tkinter popup and embed a bar chart inside a new Tkinter window.
    """
    root.after(0, lambda: show_best_method_message(results))  # Ensure it runs on Tkinter's main thread

    # Create a new Tkinter window for the chart
    chart_window = Toplevel(root)
    chart_window.title("Dimensionality Reduction Comparison")
    chart_window.geometry("700x500")

    # Extract method names and scores
    methods = list(results.keys())
    mahalanobis_distances = [results[m]["Mahalanobis Distance"] for m in methods]
    silhouette_scores = [results[m]["Silhouette Score"] for m in methods]

    # Create a Matplotlib figure
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Mahalanobis Distance as bars
    ax1.barh(methods, mahalanobis_distances, color="skyblue", label="Mahalanobis Distance")
    ax1.set_xlabel("Mahalanobis Distance (Higher is Better)")
    ax1.set_title("Comparison of Dimensionality Reduction Methods")

    # Create another x-axis for Silhouette Score
    ax2 = ax1.twiny()
    ax2.plot(silhouette_scores, methods, "ro", label="Silhouette Score")
    ax2.set_xlabel("Silhouette Score (Higher is Better)")

    # Add legends
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")

    # Embed the figure in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def show_best_method_message(results):
    """
    Show the best method based on Mahalanobis Distance and Silhouette Score.
    """
    # Find the best method based on Mahalanobis Distance and Silhouette Score
    best_mahalanobis = max(results.items(), key=lambda x: x[1]["Mahalanobis Distance"])
    best_silhouette = max(results.items(), key=lambda x: x[1]["Silhouette Score"])

    # Format the result message
    result_msg = f"=== Best Dimensionality Reduction Method ===\n"
    result_msg += f"🔹 Best by Mahalanobis Distance: {best_mahalanobis[0]} ({best_mahalanobis[1]['Mahalanobis Distance']:.3f})\n"
    result_msg += f"🔹 Best by Silhouette Score: {best_silhouette[0]} ({best_silhouette[1]['Silhouette Score']:.3f})\n"

    # If the best method is the same in both metrics
    if best_mahalanobis[0] == best_silhouette[0]:
        result_msg += f"\n🏆 Overall Best Method: **{best_mahalanobis[0]}** (Highest Separation & Clustering Score!) 🎯"

    # Show a popup message in Tkinter GUI
    messagebox.showinfo("Best Dimensionality Method", result_msg)

create_sample_sensitivity_controls()

root.mainloop()                        