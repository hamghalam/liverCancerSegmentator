import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

ct_path = "./test_input/RIA_17-010C_000_000070_0000.nii.gz"
mask_path = "./output_mask/RIA_17-010C_000_000070.nii.gz"

output_dir = "demo"
os.makedirs(output_dir, exist_ok=True)


# Load images
ct = nib.load(ct_path).get_fdata()
mask = nib.load(mask_path).get_fdata()


# Select middle axial slice
slice_idx = ct.shape[2] // 2

ct_slice = ct[:, :, slice_idx]
mask_slice = mask[:, :, slice_idx]


# Normalize CT for visualization
ct_slice = np.clip(ct_slice, -200, 300)
ct_slice = (ct_slice - ct_slice.min()) / (
    ct_slice.max() - ct_slice.min()
)


# Save CT image
plt.figure(figsize=(6,6))
plt.imshow(ct_slice, cmap="gray")
plt.axis("off")
plt.savefig(
    f"{output_dir}/ct_slice.png",
    bbox_inches="tight",
    dpi=300
)
plt.close()


# Save overlay
plt.figure(figsize=(6,6))

plt.imshow(ct_slice, cmap="gray")

# liver label
plt.imshow(
    mask_slice == 1,
    alpha=0.25
)

# tumor label
plt.imshow(
    mask_slice == 2,
    alpha=0.6
)

plt.axis("off")

plt.savefig(
    f"{output_dir}/example_result.png",
    bbox_inches="tight",
    dpi=300
)

plt.close()


print("Demo images generated in demo/")