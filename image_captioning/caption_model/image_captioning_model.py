from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from zipfile import ZipFile
import torch
import time
import shutil
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os 
from io import BytesIO

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"model's device: {device}")

print('Downloading preprocessor: ', end='')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
print('Finished')

print('Downloading model: ', end='')
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
print('Finished')

model.to(device)

folder_with_imgs = "Image/"

from PIL import Image
import numpy as np

def arafee_deleter(sentence):
    sentence = sentence.replace('arafed ', '')
    sentence = sentence.replace('araffe ', '')
    sentence = sentence.replace(' arafed', '')
    sentence = sentence.replace(' araffe', '')
    return sentence

def extract_images_only(zip_path, target_folder):
    with ZipFile(zip_path, 'r') as zip_file:
        all_files = zip_file.namelist()
        images = [f for f in all_files if f.startswith(folder_with_imgs) and not f.endswith('/')]
        for image in images:
            zip_file.extract(image, target_folder)

@app.post("/caption")
async def get_caption(file: UploadFile = File(...)):
    st_time = time.time()
    # ---------------------------
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Сохранение zip file локально
    temp_zip_path = f"{temp_dir}/{file.filename}"
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # images dir
    images_dir = f"{temp_dir}/input_images/"

    os.makedirs(images_dir, exist_ok=True)  
    
    extract_images_only(temp_zip_path, images_dir)
    
    # ---------------------------
    # opened_image = Image.open(BytesIO(file.file.read())).convert('RGB')
    # ---------------------------

    if 'Image' not in os.listdir(images_dir):
        print('No images')
        shutil.rmtree(temp_dir)
        return JSONResponse(content={'caption_result': {}})
    dirs = os.listdir(images_dir + 'Image/')
    res_dict = {}

    images = [Image.open(images_dir + 'Image/' + image_name).convert('RGB') for image_name in dirs]
    preprocessed_images = processor(images, return_tensors="pt").to(device)
    predicted_tensors = model.generate(**preprocessed_images, max_new_tokens=300)
    predicted_captions = processor.batch_decode(predicted_tensors, skip_special_tokens=True)

    for image_name, caption in zip(dirs, predicted_captions):
        res_dict["res/Image/" + image_name] = arafee_deleter(caption)
        print(f"Caption: {caption}")

    shutil.rmtree(temp_dir)

    print(f"inference time (caption): {time.time() - st_time}")
    return JSONResponse(content={'caption_result': res_dict})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3220)