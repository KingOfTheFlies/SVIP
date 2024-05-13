FROM nvidia/cuda:12.3.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app_rec

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN ln -s /usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib/libcudnn.so.8 /usr/lib/libcudnn.so
RUN ln -s /usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/libcublas.so.12 /usr/lib/libcublas.so
RUN pip install paddlepaddle-gpu==2.6.0
RUN pip install paddleocr==2.7.0.3
RUN pip install torch==2.2.1
RUN pip install httpx
COPY . /app_rec
EXPOSE 8912
CMD ["python", "rec_script.py"]