FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04


WORKDIR /app


RUN apt update && apt install -y python3 python3-pip


COPY requirements.txt .


RUN pip3 install --no-cache-dir -r requirements.txt


COPY inference_service.py .


CMD ["python3","inference_service.py"]