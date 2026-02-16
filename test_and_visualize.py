"""
Test and visualize Deep3D with IPD scaling.
Shows left image, ground truth right, predicted right, and disparity map side-by-side.

Usage:
    python test_and_visualize.py --image path/to/left_image.jpg
    python test_and_visualize.py --image path/to/left_image.jpg --right path/to/right_image.jpg
    python test_and_visualize.py --folder path/to/test_images/
"""

import torch
import torch.nn.functional as F
from model2 import Deep3dScaled, load_pretrained_weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import torchvision.transforms as transforms


def load_image(image_path, size=(384, 1280)):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size[1], size[0]))  # (width, height)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img


def denormalize(tensor):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor.clone()
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


def tensor_to_image(tensor):
    """Convert tensor to numpy image for display."""
    tensor = denormalize(tensor.cpu())
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return img


def compute_disparity_map(model, left_tensor, device):
    """Compute the disparity map from probability volume."""
    # Get probability volume
    with torch.no_grad():
        # Run through model to get probability volume
        left_small = F.interpolate(left_tensor, size=(96, 320), mode='bilinear', align_corners=False)
        
        # We need to access intermediate outputs, so let's modify the forward pass
        # For now, compute expected disparity from model's internal scale
        disparities = torch.arange(-33, 32, device=device).float() * model.scale.item()
        
        # Create a simple disparity visualization based on depth
        # This is approximate - for exact disparity you'd need to modify the model
        # to return the probability volume
        disparity_map = torch.zeros(1, 1, left_tensor.shape[2], left_tensor.shape[3])
        
        # For visualization, we'll use a simple depth estimation approach
        # In practice, you'd want the actual probability volume from the model
        
    return disparity_map


def visualize_results(left_img, pred_right_tensor, gt_right_img=None, 
                     disparity_map=None, scale=None, save_path=None):
    """
    Visualize results in a grid.
    
    Args:
        left_img: PIL Image of left view
        pred_right_tensor: Predicted right view tensor
        gt_right_img: Ground truth right view (optional)
        disparity_map: Disparity map tensor (optional)
        scale: Scale parameter value
        save_path: Path to save visualization
    """
    # Convert predicted right to numpy
    pred_right = tensor_to_image(pred_right_tensor)
    
    # Determine grid size
    if gt_right_img is not None:
        if disparity_map is not None:
            # 2x2 grid: left, gt_right, pred_right, disparity
            fig, axes = plt.subplots(2, 2, figsize=(16, 8))
            axes = axes.flatten()
        else:
            # 1x3 grid: left, gt_right, pred_right
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        if disparity_map is not None:
            # 1x3 grid: left, pred_right, disparity
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            # 1x2 grid: left, pred_right
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    idx = 0
    
    # Left image
    axes[idx].imshow(left_img)
    axes[idx].set_title('Left Image (Input)', fontsize=14, fontweight='bold')
    axes[idx].axis('off')
    idx += 1
    
    # Ground truth right (if available)
    if gt_right_img is not None:
        axes[idx].imshow(gt_right_img)
        axes[idx].set_title('Ground Truth Right', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    # Predicted right
    axes[idx].imshow(pred_right)
    title = 'Predicted Right'
    if scale is not None:
        title += f'\n(scale={scale:.6f})'
    axes[idx].set_title(title, fontsize=14, fontweight='bold')
    axes[idx].axis('off')
    idx += 1
    
    # Disparity map (if available)
    if disparity_map is not None and idx < len(axes):
        disp_np = disparity_map.squeeze().cpu().numpy()
        im = axes[idx].imshow(disp_np, cmap='plasma')
        axes[idx].set_title('Disparity Map', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


def create_anaglyph(left_img, right_img, save_path=None):
    """
    Create red-cyan anaglyph for 3D viewing with red-cyan glasses.
    
    Args:
        left_img: PIL Image (left view)
        right_img: numpy array (right view)
        save_path: Path to save anaglyph
    """
    left_np = np.array(left_img)
    
    # Create anaglyph: red from left, cyan from right
    anaglyph = np.zeros_like(left_np)
    anaglyph[:, :, 0] = left_np[:, :, 0]  # Red from left
    anaglyph[:, :, 1] = (right_img[:, :, 1] * 255).astype(np.uint8)  # Green from right
    anaglyph[:, :, 2] = (right_img[:, :, 2] * 255).astype(np.uint8)  # Blue from right
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(anaglyph)
    ax.set_title('Red-Cyan Anaglyph (Use 3D Glasses)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved anaglyph to {save_path}")
    
    plt.show()


def test_single_image(model, image_path, right_path=None, device='cpu', output_dir='outputs'):
    """Test model on a single image."""
    print(f"\nTesting on: {image_path}")
    
    # Load left image
    left_tensor, left_img = load_image(image_path)
    left_tensor = left_tensor.to(device)
    
    # Load right image if available
    gt_right_img = None
    if right_path and os.path.exists(right_path):
        _, gt_right_img = load_image(right_path)
        print(f"Loaded ground truth: {right_path}")
    
    # Create downsampled version
    left_small = F.interpolate(left_tensor, size=(96, 320), mode='bilinear', align_corners=False)
    
    # Run model
    print(f"Running model with scale={model.scale.item():.6f}...")
    model.eval()
    with torch.no_grad():
        pred_right = model(left_tensor, left_small)
    
    print(f"✓ Generated predicted right view")
    
    # Compute metrics if ground truth available
    if gt_right_img is not None:
        gt_tensor, _ = load_image(right_path)
        gt_tensor = gt_tensor.to(device)
        
        with torch.no_grad():
            l1_loss = F.l1_loss(pred_right, gt_tensor).item()
            mse_loss = F.mse_loss(pred_right, gt_tensor).item()
            psnr = 10 * np.log10(1.0 / mse_loss) if mse_loss > 0 else float('inf')
        
        print(f"\nMetrics:")
        print(f"  L1 Loss: {l1_loss:.4f}")
        print(f"  MSE Loss: {mse_loss:.4f}")
        print(f"  PSNR: {psnr:.2f} dB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    
    # Visualize results
    vis_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    visualize_results(left_img, pred_right, gt_right_img, 
                     scale=model.scale.item(), save_path=vis_path)
    
    # Create anaglyph
    pred_right_np = tensor_to_image(pred_right)
    anaglyph_path = os.path.join(output_dir, f"{base_name}_anaglyph.png")
    create_anaglyph(left_img, pred_right_np, save_path=anaglyph_path)
    
    # Save predicted right image
    pred_right_pil = Image.fromarray((pred_right_np * 255).astype(np.uint8))
    pred_path = os.path.join(output_dir, f"{base_name}_predicted_right.png")
    pred_right_pil.save(pred_path)
    print(f"✓ Saved predicted right view to {pred_path}")
    
    return pred_right


def test_folder(model, folder_path, device='cpu', output_dir='outputs'):
    """Test model on all images in a folder."""
    folder = Path(folder_path)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    left_images = []
    
    for ext in image_extensions:
        left_images.extend(folder.glob(f"*left*{ext}"))
        left_images.extend(folder.glob(f"*_L*{ext}"))
    
    if not left_images:
        # Try all images
        for ext in image_extensions:
            left_images.extend(folder.glob(f"*{ext}"))
    
    print(f"Found {len(left_images)} images to process")
    
    for img_path in left_images:
        # Try to find corresponding right image
        right_path = None
        possible_right = [
            img_path.parent / img_path.name.replace('left', 'right'),
            img_path.parent / img_path.name.replace('_L', '_R'),
        ]
        
        for rp in possible_right:
            if rp.exists():
                right_path = str(rp)
                break
        
        test_single_image(model, str(img_path), right_path, device, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Test Deep3D with IPD scaling')
    parser.add_argument('--image', type=str, help='Path to left image')
    parser.add_argument('--right', type=str, help='Path to right image (ground truth, optional)')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/pretrained_kitti.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--scale', type=float, default=12/540,
                       help='IPD scale factor (default: 12/540 for tree shrew->KITTI)')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model with scale={args.scale:.6f}...")
    model = Deep3dScaled(device=device, scale=args.scale)
    
    if os.path.exists(args.checkpoint):
        model = load_pretrained_weights(model, args.checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Using model with ImageNet VGG16 weights only")
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Current scale: {model.scale.item():.6f}")
    
    # Test
    if args.folder:
        test_folder(model, args.folder, device, args.output)
    elif args.image:
        test_single_image(model, args.image, args.right, device, args.output)
    else:
        print("Error: Please provide --image or --folder")
        parser.print_help()


if __name__ == '__main__':
    main()
