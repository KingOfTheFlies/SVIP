import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from mmocr.mmocr.apis import MMOCRInferencer
import torch
import cv2 as cv
import numpy as np
from matplotlib.patches import Polygon as patches_Polygon
import matplotlib.pyplot as plt
from contextlib import asynccontextmanager
from zipfile import ZipFile 
import io

@asynccontextmanager
async def lifespan(app: FastAPI):
    detection_config_path = "./mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py"
    detection_weights_path = './mmocr/weights/DBNetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015_20220829_230108-f289bd20.pth'
    model = MMOCRInferencer(det = detection_config_path, det_weights = detection_weights_path,device = torch.device('cuda'))
    app.state.model = model
    yield

app = FastAPI(lifespan = lifespan)

def get_det_result(image_path):
    if not os.path.exists(image_path):
        return dict()
    
    res = app.state.model(image_path)
    images_res = dict()
    
    for i,img_name in enumerate(os.listdir(image_path)): 
        confidences = np.array(res['predictions'][i]['det_scores'])
        if len(confidences) == 0:
            continue
        ind = np.where(confidences>0.5)[0]
        if len(ind) == 0:
            continue
        rec_points = np.array(res['predictions'][i]['det_polygons'])[ind,:]
        confidences = confidences[ind]
        images_res[img_name] = {'points':rec_points.tolist(),'confidences':confidences.tolist()}
    return images_res

def draw_boxes(image_path,rec_points):
    image = cv.imread(image_path)
    rects = []
    for rec in rec_points:
        x = np.array(rec).reshape((-1,2))
        x = np.vstack([x,x[0,:]]).astype('float32')
        rects.append(x)
    rects = np.array(rects)
    fig, ax = plt.subplots()
    ax.imshow(image[:,:,::-1])
    for rec in rects:
        a = patches_Polygon(rec,fill = False,edgecolor='red', linewidth = 1.5)
        ax.add_patch(a)
    plt.savefig('result_det.jpg')


def extract_from_folder(data_bytes, target_folder,folder_with_imgs):
    with ZipFile(io.BytesIO(data_bytes), 'r') as zip_file:
        all_files = zip_file.namelist()
        images = [f for f in all_files if f.startswith(folder_with_imgs) and not f.endswith('/')]
        for image in images:
            zip_file.extract(image, target_folder)
             
@app.post("/detect")
async def detect_text(path = 'Image',file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        os.makedirs('./tmp/input_images', exist_ok=True)       
        extract_from_folder(file_content, './tmp/input_images',path + '/')
        det_res = get_det_result('./tmp/input_images/' + path)  
        shutil.rmtree('tmp')
        return JSONResponse(content={'type':path,'det_result':det_res})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/draw/")
def draw(file: UploadFile = File(...)):
    try:
        with open("temp_image.jpg", "wb") as buffer:  
            shutil.copyfileobj(file.file, buffer)
        rec_points,_ = get_det_result('temp_image.jpg')
        draw_boxes('temp_image.jpg',rec_points)
        return JSONResponse(content={'result':'ok'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
            
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8156)