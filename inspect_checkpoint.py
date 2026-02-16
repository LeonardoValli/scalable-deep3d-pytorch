import torch

# Load and inspect
checkpoint = torch.load('checkpoints/pretrained_kitti.pth', map_location='cpu')

print("Checkpoint type:", type(checkpoint))
print("\nTop-level keys:")
if isinstance(checkpoint, dict):
    print(checkpoint.keys())
    
    # Try to find the state dict
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    elif 'model' in checkpoint:
        sd = checkpoint['model']
    else:
        sd = checkpoint
    
    print(f"\nNumber of parameters: {len(sd)}")
    print("\nFirst 10 parameter names:")
    for i, key in enumerate(list(sd.keys())[:10]):
        print(f"  {key}: {sd[key].shape}")