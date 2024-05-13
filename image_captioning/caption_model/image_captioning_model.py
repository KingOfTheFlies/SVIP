from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from zipfile import ZipFile
import torch
import time
import shutil
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
import os 
from io import BytesIO
from scipy import stats

app = FastAPI()
config_path = "./config.json"
config = BlipConfig.from_pretrained(config_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"model's device: {device}")

print('Downloading preprocessor: ', end='')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", config=config)
print('Finished')

print('Downloading model: ', end='')
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", config=config)
print('Finished')

model.to(device)

folder_with_imgs = "Image/"

from PIL import Image
import numpy as np

softmax = torch.nn.Softmax()
def perplexity(p):
    return np.exp(stats.entropy(p))

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
    # predicted_captions = processor.batch_decode(predicted_tensors, skip_special_tokens=True)

    # print(predicted_tensors['sequences'], predicted_tensors['scores']) 
    for image_name, image in zip(dirs, images):
        preprocessed_image = processor(image, 'an image of', return_tensors="pt").to(device)
        tokens = model.generate(**preprocessed_image, max_new_tokens=300, return_dict_in_generate=True)
        caption = processor.decode(tokens['sequences'][0], skip_special_tokens=True)
        res_dict["res/Image/" + image_name] = {}
        res_dict["res/Image/" + image_name]['caption'] = caption
        print(f"Caption: {caption}")

        score = 0
        for i in range(len(tokens['scores'])): #!!!
            score += perplexity(softmax(tokens['scores'][i].cpu().detach()[0])) #!!!
        score /= len(tokens['scores'])

        res_dict["res/Image/" + image_name]['perplexity'] = score

    # print(res_dict)

    shutil.rmtree(temp_dir)

    print(f"inference time (caption): {time.time() - st_time}")
    return JSONResponse(content={'caption_result': res_dict})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3220)