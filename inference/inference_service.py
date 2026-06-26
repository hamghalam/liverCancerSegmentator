from fastapi import FastAPI
from pydantic import BaseModel

import os
from nnunet.inference.predict import predict_from_folder



app = FastAPI()


INPUT_DIR="/data/input"
OUTPUT_DIR="/data/output"


MODEL="/models/nnUNetTrainerV2__nnUNetPlansv2.1"



class Request(BaseModel):
    case_id:str



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



if __name__=="__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000
    )