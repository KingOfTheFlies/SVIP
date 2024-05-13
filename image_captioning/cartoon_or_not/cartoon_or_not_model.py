from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from zipfile import ZipFile
import torch
import time
import shutil
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os 
from io import BytesIO

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"model's device: {device}")

print('Downloading clip: ', end='')
clip_model, clip_preprocess = clip.load('ViT-B/16')
print('Finished')

clip_model = clip_model.to(device)
folder_with_imgs = "Image/"

import clip
from PIL import Image
import numpy as np

def get_rc_clip_score(clip_model, 
                      clip_preprocess,
                      images,
                      descriptions={'cartoon': 'It is an animated series', 'real': 'It is a photo, it is a photograph'}):
    # Load the pre-trained CLIP model and the image
    res_scores = {
        'cartoon': [],
        'real': []
    }
    # for image, desc in zip(image, descriptions):
    for image in images:
        # Preprocess the image and tokenize the desc
        image_input = clip_preprocess(image).unsqueeze(0)
        desc_input = clip.tokenize(descriptions.values())

        # Move the inputs to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_input = image_input.to(device)
        desc_input = desc_input.to(device)
        clip_model = clip_model.to(device)

        # Generate embeddings for the image and desc
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            desc_features = clip_model.encode_text(desc_input)

        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        desc_features = desc_features / desc_features.norm(dim=-1, keepdim=True)

        rc_scores = torch.matmul(image_features, desc_features.T).tolist()[0]
        # Calculate the cosine similarity to get the CLIP score
        res_scores['cartoon'].append(rc_scores[0])
        res_scores['real'].append(rc_scores[1])
        # clip_scores.append(torch.matmul(image_features, desc_features.T).item())

    return res_scores

def extract_images_only(zip_path, target_folder):
    with ZipFile(zip_path, 'r') as zip_file:
        all_files = zip_file.namelist()
        images = [f for f in all_files if f.startswith(folder_with_imgs) and not f.endswith('/')]
        for image in images:
            zip_file.extract(image, target_folder)

@app.post("/cartoon_or_not")
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
        return JSONResponse(content={'cartoon_or_not_result': {}})
    dirs = os.listdir(images_dir + 'Image/')
    res_dict = {}

    images = [Image.open(images_dir + 'Image/' + image_name).convert('RGB') for image_name in dirs]
    descriptions = {
        'cartoon': 'It is an animated series',
        'real': 'It is a photo, it is a photograph'
    }
    rc_result = get_rc_clip_score(clip_model, clip_preprocess, images, descriptions)
    for image_name, cartoon_score, real_score in zip(dirs, rc_result['cartoon'], rc_result['real']):
        res_dict["res/Image/" + image_name] = int(cartoon_score > real_score)
    # ------------------------

    shutil.rmtree(temp_dir)

    print(f"inference time (cartoon_or_not): {time.time() - st_time}")
    return JSONResponse(content={'cartoon_or_not_result': res_dict})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3219)