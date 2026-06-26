from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import shutil
import requests
import uuid


app = FastAPI(
    title="Liver Cancer Segmentator API",
    description="nnUNet based liver and tumor segmentation service"
)


INPUT_DIR = "/data/input"
OUTPUT_DIR = "/data/output"

NNUNET_SERVICE = "http://nnunet:9000/predict"


os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)



@app.get("/")
def home():
    return {
        "service": "Liver Cancer Segmentator",
        "status": "running"
    }



@app.post("/segment")
async def segment(file: UploadFile = File(...)):

    case_id = str(uuid.uuid4())


    # nnUNet requires _0000.nii.gz
    input_name = f"{case_id}_0000.nii.gz"


    input_path = os.path.join(
        INPUT_DIR,
        input_name
    )


    # save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )


    # call nnUNet container
    response = requests.post(
        NNUNET_SERVICE,
        json={
            "case_id": case_id
        }
    )


    if response.status_code != 200:
        return {
            "status": "failed",
            "message": response.text
        }


    output_file = os.path.join(
        OUTPUT_DIR,
        f"{case_id}.nii.gz"
    )


    return FileResponse(
        output_file,
        media_type="application/gzip",
        filename="segmentation.nii.gz"
    )