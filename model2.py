"""
Modified Deep3D model with IPD scaling for tree shrew adaptation.
Based on anishmadan23/deep3d-pytorch with added learnable scale parameter.

Changes from original:
1. Added scale parameter (initialized to 12/540 = 0.0222 for tree shrew -> KITTI scaling)
2. Modified selection layer to use scaled disparities
3. Uses sub-pixel accurate grid_sample instead of integer pixel shifts
"""

import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import torch.nn.functional as F

cfg = [64, 128, 256, 512, 512]

class Deep3dScaled(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, device=torch.device('cpu'), scale=12/540):
        """
        Deep3D model with IPD scaling.
        
        Args:
            in_channels: Input image channels (default: 3 for RGB)
            out_channels: Output image channels (default: 3 for RGB)
            device: Device to run on (cpu or cuda)
            scale: IPD scaling factor (default: 12/540 for tree_shrew/KITTI)
        """
        super(Deep3dScaled, self).__init__()
        self.device = device
        
        # Learnable scale parameter for IPD adaptation
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=True)
        
        # Original VGG16 feature extraction
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        modules = []
        layer = []
        for l in vgg16.features:
            if isinstance(l, nn.MaxPool2d):
                layer.append(l)
                modules.append(layer)
                layer = []
            else:
                layer.append(l)
        
        # Deconvolution branches
        scale_factor = 1
        deconv = []
        layer = []
        for m in range(len(modules)):
            layer.append(nn.Conv2d(cfg[m], cfg[m], kernel_size=3, stride=1, padding=1))
            layer.append(nn.ReLU(inplace=True))
            layer.append(nn.Conv2d(cfg[m], cfg[m], kernel_size=3, stride=1, padding=1))
            layer.append(nn.ReLU(inplace=True))
            if(m==0):
                layer.append(nn.ConvTranspose2d(cfg[m], 65, kernel_size=1, stride=1, padding=(0,0)))
            else:
                scale_factor *=2
                layer.append(nn.ConvTranspose2d(cfg[m], 65, kernel_size=scale_factor*2, stride=scale_factor, padding=(scale_factor//2, scale_factor//2)))
            deconv.append(layer)
            layer = []
        
        self.module_1 = nn.Sequential(*modules[0])
        self.module_2 = nn.Sequential(*modules[1])
        self.module_3 = nn.Sequential(*modules[2])
        self.module_4 = nn.Sequential(*modules[3])
        self.module_5 = nn.Sequential(*modules[4])
        
        self.deconv_1 = nn.Sequential(*deconv[0])
        self.deconv_2 = nn.Sequential(*deconv[1])
        self.deconv_3 = nn.Sequential(*deconv[2])
        self.deconv_4 = nn.Sequential(*deconv[3])
        self.deconv_5 = nn.Sequential(*deconv[4])
        
        self.linear_module = nn.Sequential(*[nn.Linear(15360,4096),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=0.5),
                                            nn.Linear(4096,1950)])
        
        self.deconv_6 = nn.Sequential(*[nn.ConvTranspose2d(65,65,kernel_size=scale_factor*2,stride=scale_factor,padding=(scale_factor//2,scale_factor//2))])
        
        self.upconv_final = nn.Sequential(*[nn.ConvTranspose2d(65,65,kernel_size=(4,4),stride=2,padding=(1,1)),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(65,65,kernel_size=(3,3),stride=1,padding=(1,1)),
                                            nn.Softmax(dim=1)])
        
        # Initialize new layers
        for block in [self.deconv_1,self.deconv_2,self.deconv_3,self.deconv_4,self.deconv_5,self.deconv_6,self.linear_module,self.upconv_final]:
            for m in block:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, orig_x, x):
        """
        Forward pass with scaled disparity selection.
        
        Args:
            orig_x: Original full-resolution left image (B, 3, H, W)
            x: Downsampled left image for VGG processing (B, 3, H/4, W/4)
            
        Returns:
            Synthesized right image (B, 3, H, W)
        """
        x_copy = orig_x
        pred = []
        
        # VGG feature extraction
        out_1 = self.module_1(x)
        out_2 = self.module_2(out_1)
        out_3 = self.module_3(out_2)
        out_4 = self.module_4(out_3)
        out_5 = self.module_5(out_4)
        
        # FC layers
        out_5_flatten = out_5.view(x_copy.shape[0],-1)
        out_6 = self.linear_module(out_5_flatten)
        
        # Deconvolution branches
        p1 = self.deconv_1(out_1)
        p2 = self.deconv_2(out_2)
        p3 = self.deconv_3(out_3)
        p4 = self.deconv_4(out_4)
        p5 = self.deconv_5(out_5)
        p6 = self.deconv_6(out_6.view(x_copy.shape[0],65,3,10))
        
        pred.append(p1)
        pred.append(p2)
        pred.append(p3)
        pred.append(p4)
        pred.append(p5)
        pred.append(p6)
        
        # Feature fusion
        out = torch.zeros(pred[0].shape).to(self.device)
        for p in pred:
            out = torch.add(out, p)
        
        # Final upsampling and softmax
        out = self.upconv_final(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear')
        
        # --- MODIFIED SELECTION LAYER WITH IPD SCALING ---
        B, C, H, W = x_copy.shape
        
        # Create coordinate grids for sub-pixel shifting
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing="ij"
        )
        base_grid = torch.stack((x_coords, y_coords), dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
        
        right_img = torch.zeros_like(x_copy)
        
        # Iterate through disparity levels with scaling
        for depth_map_idx in range(-33, 32):
            # Apply IPD scaling to disparity
            scaled_disp = depth_map_idx * self.scale
            
            # Convert disparity to normalized coordinates [-1, 1]
            shift = 2 * scaled_disp / W
            
            # Create shifted grid
            grid = base_grid.clone()
            grid[..., 0] -= shift  # Shift horizontally (negative for right view)
            
            # Sample from left image at shifted locations (sub-pixel accurate)
            shifted = F.grid_sample(
                x_copy,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            
            # Weight by disparity probability
            weight = out[:, depth_map_idx+33:depth_map_idx+34, :, :]
            right_img += shifted * weight
        
        return right_img


def load_pretrained_weights(model, pretrained_path, strict=False):
    """
    Load pretrained weights for custom layers only.
    VGG16 backbone is already loaded from ImageNet in __init__.
    """
    print(f"Loading custom layer weights from {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Get model's current state
    model_dict = model.state_dict()
    
    # Filter to only matching keys and shapes
    pretrained_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    
    # Update model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Report what was loaded
    loaded_keys = len(pretrained_dict)
    total_keys = len(model_dict)
    checkpoint_keys = len(state_dict)
    
    print(f"\n✓ Loaded {loaded_keys} custom layer parameters from checkpoint")
    print(f"✓ VGG16 backbone ({total_keys - loaded_keys - 1} params) loaded from ImageNet")
    print(f"✓ Scale parameter initialized to {model.scale.item():.6f}")
    print(f"\nTotal model parameters: {total_keys}")
    print(f"  - From checkpoint (KITTI): {loaded_keys}")
    print(f"  - From ImageNet (VGG16): {total_keys - loaded_keys - 1}")  
    print(f"  - Initialized (scale): 1")
    
    return model


# Example usage
if __name__ == '__main__':
    # Create model with tree shrew -> KITTI scaling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Deep3dScaled(device=device, scale=12/540)
    
    print(f"Model created on {device}")
    print(f"Initial scale parameter: {model.scale.item():.6f}")
    print(f"Scale is learnable: {model.scale.requires_grad}")
    
    # Load pretrained KITTI weights
    model = load_pretrained_weights(model, 'checkpoints/pretrained_kitti.pth')
    
    # Test forward pass
    left_orig = torch.randn(2, 3, 384, 1280).to(device)
    left_small = torch.randn(2, 3, 96, 320).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(left_orig, left_small)
    
    print(f"Output shape: {output.shape}")
    print("✓ Model test successful!")
