import os
from nnunet.inference.predict import predict_from_folder


class LiverSegmentator:

    def __init__(self, model_path):
        self.model_path = model_path

    def predict(
        self,
        input_folder,
        output_folder
    ):

        os.makedirs(output_folder, exist_ok=True)

        predict_from_folder(
            model=self.model_path,
            input_folder=input_folder,
            output_folder=output_folder,
            folds=(0, 1, 2, 3, 4),
            save_npz=False,
            num_threads_preprocessing=2,
            num_threads_nifti_save=2,
            lowres_segmentations=None,
            part_id=0,
            num_parts=1,
            tta=True,
            mixed_precision=True,
            overwrite_existing=True
        )