"""
Visualization: Why Independent Keyframe Encoding Matters

This script demonstrates the difference between:
1. WRONG: Encoding all keyframes together with 3D convolution
2. CORRECT: Encoding each keyframe independently as 1-frame video
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

def simulate_3d_conv_receptive_field():
    """
    Simulate how 3D convolution creates artificial temporal correlation
    """
    # Keyframe indices (temporally distant)
    keyframe_indices = [0, 20, 40, 60, 80]
    
    # Simulate 3D conv kernel size (T, H, W) = (3, 3, 3)
    temporal_kernel_size = 3
    
    print("="*60)
    print("3D Convolution Receptive Field Analysis")
    print("="*60)
    
    print(f"\nKeyframe indices: {keyframe_indices}")
    print(f"Temporal distances: {[keyframe_indices[i+1] - keyframe_indices[i] for i in range(len(keyframe_indices)-1)]}")
    
    print("\n❌ WRONG: Concatenating keyframes for 3D convolution")
    print("   Keyframes arranged as: [K0, K1, K2, K3, K4]")
    print("   3D Conv sees them as: [Frame 0, Frame 1, Frame 2, Frame 3, Frame 4]")
    print("   Problem: Conv kernel assumes K0 and K1 are adjacent frames!")
    print("   Reality: K0 (frame 0) and K1 (frame 20) are 20 frames apart")
    
    print("\n✓ CORRECT: Encoding each keyframe independently")
    print("   K0 encoded as: [Frame 0] (1-frame video)")
    print("   K1 encoded as: [Frame 20] (1-frame video)")
    print("   K2 encoded as: [Frame 40] (1-frame video)")
    print("   No artificial temporal correlation!")
    
    # Visualize receptive field
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Wrong approach
    ax1 = axes[0]
    wrong_positions = np.arange(len(keyframe_indices))
    ax1.scatter(wrong_positions, [1]*len(keyframe_indices), s=200, c='red', marker='o', label='Keyframes')
    
    # Show 3D conv receptive field
    for i in range(len(keyframe_indices) - temporal_kernel_size + 1):
        ax1.plot([i, i+temporal_kernel_size-1], [0.5, 0.5], 'r-', linewidth=3, alpha=0.3)
        ax1.fill_between([i, i+temporal_kernel_size-1], 0.3, 0.7, color='red', alpha=0.2)
    
    ax1.set_xlim(-0.5, len(keyframe_indices)-0.5)
    ax1.set_ylim(0, 1.5)
    ax1.set_xlabel('Position in Concatenated Sequence', fontsize=12)
    ax1.set_title('❌ WRONG: 3D Conv Creates Artificial Temporal Correlation', fontsize=14, color='red')
    ax1.set_xticks(wrong_positions)
    ax1.set_xticklabels([f'K{i}\n(frame {keyframe_indices[i]})' for i in range(len(keyframe_indices))])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Correct approach
    ax2 = axes[1]
    correct_positions = keyframe_indices
    ax2.scatter(correct_positions, [1]*len(keyframe_indices), s=200, c='green', marker='o', label='Keyframes (independent)')
    
    # Show independent encoding (no cross-frame receptive field)
    for i, pos in enumerate(correct_positions):
        ax2.plot([pos-1, pos+1], [0.5, 0.5], 'g-', linewidth=3, alpha=0.5)
        ax2.fill_between([pos-1, pos+1], 0.3, 0.7, color='green', alpha=0.2)
        ax2.text(pos, 0.5, f'1-frame\nvideo', ha='center', va='center', fontsize=8)
    
    ax2.set_xlim(-5, 85)
    ax2.set_ylim(0, 1.5)
    ax2.set_xlabel('Actual Frame Index in Original Video', fontsize=12)
    ax2.set_title('✓ CORRECT: Independent Encoding (No Artificial Correlation)', fontsize=14, color='green')
    ax2.set_xticks(correct_positions)
    ax2.set_xticklabels([f'K{i}\n(frame {keyframe_indices[i]})' for i in range(len(keyframe_indices))])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('docs/keyframe_encoding_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: docs/keyframe_encoding_comparison.png")


def demonstrate_latent_quality():
    """
    Demonstrate how encoding method affects latent quality
    """
    print("\n" + "="*60)
    print("Latent Quality Comparison")
    print("="*60)
    
    # Simulate latent statistics
    np.random.seed(42)
    
    # Wrong: Concatenated encoding
    # 3D conv creates artificial smoothing between keyframes
    wrong_latents = []
    for i in range(5):
        if i == 0:
            latent = np.random.randn(16, 32, 32)
        else:
            # Artificial correlation from 3D conv
            latent = 0.7 * wrong_latents[-1] + 0.3 * np.random.randn(16, 32, 32)
        wrong_latents.append(latent)
    
    # Correct: Independent encoding
    correct_latents = [np.random.randn(16, 32, 32) for _ in range(5)]
    
    # Compute correlation between consecutive keyframes
    wrong_corr = [np.corrcoef(wrong_latents[i].flatten(), wrong_latents[i+1].flatten())[0,1] 
                  for i in range(4)]
    correct_corr = [np.corrcoef(correct_latents[i].flatten(), correct_latents[i+1].flatten())[0,1] 
                    for i in range(4)]
    
    print("\nCorrelation between consecutive keyframe latents:")
    print(f"❌ Wrong (concatenated):  {np.mean(wrong_corr):.4f} (artificially high!)")
    print(f"✓ Correct (independent): {np.mean(correct_corr):.4f} (properly independent)")
    
    print("\nWhy this matters:")
    print("- High correlation → Model thinks keyframes are temporally close")
    print("- Low correlation → Model correctly treats keyframes as independent")
    print("- Independent encoding preserves true temporal structure")


def show_code_comparison():
    """
    Show code comparison
    """
    print("\n" + "="*60)
    print("Code Comparison")
    print("="*60)
    
    print("\n❌ WRONG APPROACH:")
    print("""
# Concatenate all keyframes
keyframes_tensor = torch.stack(keyframes, dim=2)  # (C, K, H, W)
z_edit = vae.encode(keyframes_tensor)  # 3D conv assumes temporal continuity!

# Problem: VAE's 3D convolution will create artificial correlation
# between frame 0 and frame 20, treating them as adjacent frames
""")
    
    print("\n✓ CORRECT APPROACH:")
    print("""
# Encode each keyframe independently
z_edit_list = []
for keyframe in keyframes:
    keyframe_tensor = keyframe.unsqueeze(2)  # (C, 1, H, W)
    z_k = vae.encode(keyframe_tensor)  # Treat as 1-frame video
    z_edit_list.append(z_k)
z_edit = torch.cat(z_edit_list, dim=2)  # (B, C, K, H, W)

# Benefit: Each keyframe is encoded independently
# No artificial temporal correlation
""")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  Why Independent Keyframe Encoding is Critical")
    print("="*70)
    
    simulate_3d_conv_receptive_field()
    demonstrate_latent_quality()
    show_code_comparison()
    
    print("\n" + "="*70)
    print("Summary:")
    print("- Keyframes are temporally distant (e.g., 20 frames apart)")
    print("- 3D convolution assumes temporal continuity")
    print("- Independent encoding preserves true temporal structure")
    print("- This is critical for Flow Matching to work correctly")
    print("="*70 + "\n")
