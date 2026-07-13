import os
import sys
import zarr
import numpy as np
import matplotlib.pyplot as plt

# Make numpy output readable (limit decimal places, suppress scientific notation)
np.set_printoptions(precision=4, suppress=True, edgeitems=3)

def inspect_zarr(path):
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        return

    try:
        data = zarr.open(path, mode='r')
    except Exception as e:
        print(f"Could not open Zarr file: {e}")
        return

    print(f"\n{'='*20} ZARR INSPECTION {'='*20}")
    print(f"Path: {path}")
    
    # 1. Visual Tree Structure
    print("\n--- Hierarchy Tree ---")
    try:
        print(data.tree())
    except AttributeError:
        print("Root is a single Array (no tree structure).")

    # 2. Detailed Iteration
    print("\n--- Detailed Content ---")
    print(f"{'Path':<40} | {'Type':<10} | {'Shape':<20} | {'Dtype'}")
    print("-" * 85)

    def visitor(name, obj):
        is_group = isinstance(obj, zarr.hierarchy.Group)
        obj_type = "Group" if is_group else "Array"
        shape = str(obj.shape) if not is_group else "-"
        dtype = str(obj.dtype) if not is_group else "-"
        print(f"{name:<40} | {obj_type:<10} | {shape:<20} | {dtype}")

    if hasattr(data, 'visititems'):
        data.visititems(visitor)
    else:
        print(f"{'root':<40} | Array      | {data.shape:<20} | {data.dtype}")

    # 3. Attributes (Metadata)
    if len(data.attrs) > 0:
        print("\n--- Root Attributes (Metadata) ---")
        for key, val in data.attrs.items():
            print(f"{key}: {val}")


def analyze_dataset_statistics(zarr_path):
    """Computes stats (min, max, mean, std, jitter) and shows examples for non-image arrays."""
    try:
        z = zarr.open(zarr_path, mode='r')
    except Exception as e:
        return
        
    print(f"\n{'='*20} DATA ANALYSIS {'='*20}")

    def stat_visitor(name, obj):
        if not isinstance(obj, zarr.core.Array):
            return
            
        # Skip string/object arrays
        if not np.issubdtype(obj.dtype, np.number):
            return

        # Heuristic: Skip arrays that look like images (saves RAM & calculation time)
        if obj.ndim >= 3 and (obj.dtype == np.uint8 or 'image' in name.lower()):
            print(f"\n[{name}] (Identified as Image/Video data - skipping numerical stats)")
            return

        # Load data into memory for analysis
        data = obj[:]
        
        if data.size == 0:
            print(f"\n[{name}] is empty.")
            return

        print(f"\n--- Analysis: {name} ---")
        print(f"Example Data (first 3 rows):\n{data[:3]}")
        
        # If it's a 1D or 2D array (e.g., shape (N,) or (N, D)), compute stats per dimension
        axis_to_reduce = 0 if data.ndim > 1 else None
        
        print(f"Min:  {np.min(data, axis=axis_to_reduce)}")
        print(f"Max:  {np.max(data, axis=axis_to_reduce)}")
        print(f"Mean: {np.mean(data, axis=axis_to_reduce)}")
        print(f"Std:  {np.std(data, axis=axis_to_reduce)}")
        
        # Calculate Jitter (Mean Absolute Difference between consecutive time steps)
        if data.shape[0] > 1:
            diffs = np.diff(data, axis=0)
            jitter = np.mean(np.abs(diffs), axis=axis_to_reduce)
            print(f"Jitter (Mean Abs Step Diff): {jitter}")

    if hasattr(z, 'visititems'):
        z.visititems(stat_visitor)


def visualize_images(zarr_path, num_samples=3):
    """Finds image arrays and plots sample frames using matplotlib."""
    try:
        z = zarr.open(zarr_path, mode='r')
    except Exception as e:
        return

    image_arrays = []
    
    def find_images(name, obj):
        if isinstance(obj, zarr.core.Array):
            # Heuristic to detect images: 4D (T, H, W, C), uint8, or named appropriately
            if obj.ndim >= 3 and (obj.dtype == np.uint8 or 'image' in name.lower() or 'rgb' in name.lower()):
                image_arrays.append((name, obj))
                
    if hasattr(z, 'visititems'):
        z.visititems(find_images)
        
    if not image_arrays:
        print("\nNo image arrays found to visualize.")
        return
        
    print(f"\n{'='*20} IMAGE VISUALIZATION {'='*20}")
    
    for name, arr in image_arrays:
        print(f"Preparing to show samples from '{name}'...")
        total_frames = arr.shape[0]
        
        # Pick evenly spaced indices across the dataset
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        if num_samples == 1:
            axes = [axes]
            
        fig.suptitle(f"Image Samples from: {name}", fontsize=14)
        
        for ax, idx in zip(axes, indices):
            img = arr[idx]
            
            # Handle PyTorch style (Channels First: C, H, W) -> Matplotlib style (H, W, C)
            if img.shape[0] == 3 and img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
                
            ax.imshow(img)
            ax.set_title(f"Frame {idx}")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()


def print_episode_ends(zarr_path):
    try:
        z = zarr.open(zarr_path, mode='r')
        if 'meta/episode_ends' in z:
            print(f"\n--- Episode Ends ---")
            print(z['meta/episode_ends'][:])
        else:
            print(f"\nKey 'meta/episode_ends' not found in {zarr_path}")
    except Exception as e:
        print(f"Error reading Zarr: {e}")


if __name__ == "__main__":
    # Get path from command line argument or default
    zarr_path = sys.argv[1] if len(sys.argv) > 1 else "data.zarr"
    
    # 1. Structural Overview
    inspect_zarr(zarr_path)
    
    # 2. Episode Metadata
    print_episode_ends(zarr_path)
    
    # 3. Statistical Analysis (Stats, Jitter, Examples)
    analyze_dataset_statistics(zarr_path)
    
    # 4. Visualizing Image Data
    visualize_images(zarr_path, num_samples=4)
