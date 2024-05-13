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
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
COPY . /app_det

EXPOSE 8873
CMD ["python", "detect.py"]