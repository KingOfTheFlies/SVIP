ARG USE_CUDA

FROM pytorch/torchserve:latest-gpu

WORKDIR /cartoon_or_not

EXPOSE 3219

USER root
RUN pip install --upgrade pip
RUN pip install python-multipart==0.0.9

RUN mkdir -p /var/lib/apt/lists/partial
RUN apt update
RUN apt install -y git

COPY . /cartoon_or_not
RUN pip install -r requirements.txt
RUN pip install -U git+https://github.com/openai/CLIP.git

CMD [ "python", "./cartoon_or_not_model.py" ]