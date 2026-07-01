from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nnunet.inference.predict import predict_from_folder

try:
    from .metrics import dice_coefficient, pairwise_dice_confidence, tumor_volume_mm3
except ImportError:
    from metrics import dice_coefficient, pairwise_dice_confidence, tumor_volume_mm3


DEFAULT_FOLDS = (0, 1, 2, 3, 4)


def output_name_from_case_id(case_id: str) -> str:
    return f"{case_id}.nii.gz"


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, ...], tuple[int, ...]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing = tuple(float(value) for value in image.GetSpacing())
    size = tuple(int(value) for value in image.GetSize())
    return array, spacing, size


def run_nnunet_prediction(
    model: Path,
    input_dir: Path,
    output_dir: Path,
    folds: tuple[int, ...],
    tta: bool = True,
    overwrite_existing: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predict_from_folder(
        model=str(model),
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        folds=folds,
        save_npz=False,
        num_threads_preprocessing=2,
        num_threads_nifti_save=2,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=tta,
        mixed_precision=True,
        overwrite_existing=overwrite_existing,
    )


def compute_prediction_metadata(
    case_id: str,
    ensemble_prediction: Path,
    fold_predictions: dict[int, Path],
    label_value: int = 2,
) -> dict:
    ensemble_array, spacing, size = read_mask(ensemble_prediction)
    fold_arrays = {}
    fold_metadata = {}

    for fold, prediction_path in sorted(fold_predictions.items()):
        fold_array, fold_spacing, fold_size = read_mask(prediction_path)
        fold_arrays[fold] = fold_array
        fold_metadata[f"fold_{fold}"] = {
            "prediction_path": str(prediction_path),
            "spacing_mm": list(fold_spacing),
            "size_voxels": list(fold_size),
            "tumor_volume_mm3": tumor_volume_mm3(
                fold_array,
                fold_spacing,
                label_value=label_value,
            ),
        }

    pairwise_scores = {}
    folds = sorted(fold_predictions)
    for left_index, left_fold in enumerate(folds):
        for right_fold in folds[left_index + 1 :]:
            score = dice_coefficient(
                fold_arrays[left_fold],
                fold_arrays[right_fold],
                label_value=label_value,
            )
            pairwise_scores[f"fold_{left_fold}_vs_fold_{right_fold}"] = score

    return {
        "case_id": case_id,
        "label_value": label_value,
        "ensemble_prediction_path": str(ensemble_prediction),
        "fold_predictions": fold_metadata,
        "confidence_pairwise_dice": pairwise_dice_confidence(
            fold_arrays.values(),
            label_value=label_value,
        ),
        "pairwise_dice_scores": pairwise_scores,
        "spacing_mm": list(spacing),
        "size_voxels": list(size),
        "tumor_volume_mm3": tumor_volume_mm3(
            ensemble_array,
            spacing,
            label_value=label_value,
        ),
    }


def predict_with_confidence(
    model: Path,
    input_dir: Path,
    output_dir: Path,
    case_id: str,
    folds: tuple[int, ...] = DEFAULT_FOLDS,
    label_value: int = 2,
    tta: bool = True,
    overwrite_existing: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_nnunet_prediction(
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        folds=folds,
        tta=tta,
        overwrite_existing=overwrite_existing,
    )

    fold_prediction_paths = {}
    for fold in folds:
        fold_output_dir = output_dir / "fold_predictions" / f"fold_{fold}"
        run_nnunet_prediction(
            model=model,
            input_dir=input_dir,
            output_dir=fold_output_dir,
            folds=(fold,),
            tta=tta,
            overwrite_existing=overwrite_existing,
        )
        fold_prediction_paths[fold] = fold_output_dir / output_name_from_case_id(case_id)

    ensemble_prediction = output_dir / output_name_from_case_id(case_id)
    metadata = compute_prediction_metadata(
        case_id=case_id,
        ensemble_prediction=ensemble_prediction,
        fold_predictions=fold_prediction_paths,
        label_value=label_value,
    )

    metadata_path = output_dir / f"{case_id}_confidence.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    metadata["metadata_path"] = str(metadata_path)

    return metadata


def parse_folds(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nnUNet ensemble prediction and save confidence metadata."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input folder with nnUNet *_0000.nii.gz files.")
    parser.add_argument("--output", required=True, type=Path, help="Output folder for ensemble, fold masks, and metadata.")
    parser.add_argument("--model", required=True, type=Path, help="nnUNet model folder.")
    parser.add_argument("--case-id", required=True, help="Case identifier without _0000 or .nii.gz suffix.")
    parser.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated folds to run.")
    parser.add_argument("--label-value", default=2, type=int, help="Segmentation label used for tumor confidence.")
    parser.add_argument("--disable-tta", action="store_true", help="Disable test-time augmentation.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing predictions.")
    parser.add_argument(
        "--copy-metadata-to",
        type=Path,
        default=None,
        help="Optional extra path where the metadata JSON should be copied.",
    )

    args = parser.parse_args()

    metadata = predict_with_confidence(
        model=args.model,
        input_dir=args.input,
        output_dir=args.output,
        case_id=args.case_id,
        folds=parse_folds(args.folds),
        label_value=args.label_value,
        tta=not args.disable_tta,
        overwrite_existing=not args.no_overwrite,
    )

    if args.copy_metadata_to is not None:
        args.copy_metadata_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata["metadata_path"], args.copy_metadata_to)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
