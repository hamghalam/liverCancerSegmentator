from fastapi import FastAPI
from pydantic import BaseModel

import os
import shutil
from pathlib import Path
from nnunet.inference.predict import predict_from_folder

from failureDetection.confidence_cli import predict_with_confidence



app = FastAPI()


INPUT_DIR="/data/input"
OUTPUT_DIR="/data/output"
WORK_DIR="/data/work"


MODEL="/models/nnUNetTrainerV2__nnUNetPlansv2.1"



class Request(BaseModel):
    case_id:str

class ConfidenceRequest(BaseModel):
    case_id:str
    label_value:int=2



@app.post("/predict")
def predict(req:Request):


    predict_from_folder(
        model=MODEL,
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        folds=(0,1,2,3,4),
        save_npz=False,
        num_threads_preprocessing=2,
        num_threads_nifti_save=2,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=True
    )


    return {
        "status":"completed",
        "case":req.case_id
    }


@app.post("/predict-with-confidence")
def predict_with_confidence_endpoint(req:ConfidenceRequest):


    case_input_dir = Path(WORK_DIR) / req.case_id / "input"
    case_input_dir.mkdir(parents=True, exist_ok=True)
    source_file = Path(INPUT_DIR) / f"{req.case_id}_0000.nii.gz"
    case_file = case_input_dir / source_file.name
    shutil.copyfile(source_file, case_file)


    metadata = predict_with_confidence(
        model=Path(MODEL),
        input_dir=case_input_dir,
        output_dir=Path(OUTPUT_DIR),
        case_id=req.case_id,
        label_value=req.label_value,
    )


    return {
        "status":"completed",
        "case":req.case_id,
        "confidence":metadata
    }



if __name__=="__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000
    )
