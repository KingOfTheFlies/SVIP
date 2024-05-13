FROM nvidia/cuda:12.3.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app_det

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install python-multipart==0.0.9
RUN pip install uvicorn==0.29.0
RUN python -m pip install pip~=19.0
RUN pip install -U openmim
RUN pip install six --upgrade
RUN pip install setuptools --upgrade
COPY . /app_det

RUN cd mmocr && mim install -v -e .
EXPOSE 8156
CMD ["python", "det_script.py"]
