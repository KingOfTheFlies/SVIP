ARG USE_CUDA

FROM pytorch/torchserve:latest-gpu

WORKDIR /image_captioning_model

EXPOSE 3220

# RUN useradd -ms /bin/bash nonrootuser
# USER nonrootuser

RUN pip install --upgrade pip
RUN pip install python-multipart==0.0.9

# USER nonrootuser
COPY . /image_captioning_model
RUN pip install -r requirements.txt

CMD [ "python", "./image_captioning_model.py" ]