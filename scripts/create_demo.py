import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

ct_path = "./test_input/RIA_17-010C_000_000070_0000.nii.gz"
mask_path = "./output_mask/RIA_17-010C_000_000070.nii.gz"

output_dir = "demo"
os.makedirs(output_dir, exist_ok=True)

ct = nib.load(ct_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

slice_idx = ct.shape[2] // 2

ct_slice = ct[:, :, slice_idx]
mask_slice = mask[:, :, slice_idx]

# ----------------------------
# Orientation fix
# ----------------------------
ct_slice = np.rot90(ct_slice, k=-1)
mask_slice = np.rot90(mask_slice, k=-1)

# ----------------------------
# Normalize CT
# ----------------------------
ct_slice = np.clip(ct_slice, -200, 300)
ct_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)

# ----------------------------
# Base RGB
# ----------------------------
rgb = np.stack([ct_slice]*3, axis=-1)

liver = mask_slice == 1
tumor = mask_slice == 2

# ----------------------------
# Liver (soft green)
# ----------------------------
liver_color = np.array([0.0, 1.0, 0.0])
liver_alpha = 0.20

rgb[liver] = (1 - liver_alpha) * rgb[liver] + liver_alpha * liver_color

# ----------------------------
# Tumor (strong red on top)
# ----------------------------
tumor_color = np.array([1.0, 0.0, 0.0])
tumor_alpha = 0.65

rgb[tumor] = (1 - tumor_alpha) * rgb[tumor] + tumor_alpha * tumor_color

# ----------------------------
# Save
# ----------------------------
plt.figure(figsize=(6, 6))
plt.imshow(rgb)
plt.axis("off")
plt.tight_layout()

plt.savefig(
    f"{output_dir}/example_result.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print("Demo images generated in demo/")