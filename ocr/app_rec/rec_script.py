from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from PIL import Image
import io
from PaddleOCR.tools.infer_rec import inference,prepare_model
import cv2 as cv
import numpy as np
import logging
from zipfile import ZipFile
import os
import json
import httpx
import bisect
import asyncio
from contextlib import asynccontextmanager
import math
from collections import defaultdict
import logging

config_path = 'PaddleOCR/configs/rec/rec_svtrnet.yml'
opt = {'Global.checkpoints': 'models/rec/model (6)','Global.character_dict_path':'advance_dict_little.txt','Global.use_space_char':'True'}
@asynccontextmanager

async def lifespan(app: FastAPI):
    logging.getLogger().setLevel(logging.ERROR)
    prepare_model(config_path,opt)
    yield

app = FastAPI(lifespan = lifespan)

def recognize(det_res_json,image_path,data_json):
    result_dict = dict()
    data_confidence = dict()
    if det_res_json is None :
        opt['Global.infer_img'] = image_path
        info,time = inference(config_path,opt)  
        for data in info:
            if  ((len(data['text']) != 0)) and data['confidence'] > 0.6:
                short_img_name = data['img_name'].split('/')[-1]
                result_dict[short_img_name] = defaultdict(list)
                result_dict[short_img_name]['points'].append(data_json[data['img_name']])
                result_dict[short_img_name]['texts'].append(data['text'])
                result_dict[short_img_name]['confidence'] = data['confidence']    
        return result_dict   
    else:
        os.makedirs('./tmp/cropped/', exist_ok=True)
        k = 0
        for img_name in det_res_json:
            img_bbox = data_json[image_path + img_name]
            img_x = min(img_bbox[0],img_bbox[2])
            img_y = min(img_bbox[1],img_bbox[3])
            image = cv.imread(image_path + img_name)
            rec_points = np.array(det_res_json[img_name]['points'])
            confidences = det_res_json[img_name]['confidences']
            if len(rec_points) > 0:
                result_dict[img_name] = defaultdict(list)
            for p,conf in zip(rec_points,confidences):
                os.makedirs(f'./tmp/cropped/{img_name[:-4]}', exist_ok=True)
                p = p.reshape((-1,2)).astype(int)
                x_min = max(0,min([point[0] for point in p]))
                y_min = max(0,min([point[1] for point in p]))
                x_max = max([point[0] for point in p])
                y_max = max([point[1] for point in p])
                
                new_points = [int(x_min + img_x), int(y_min + img_y), int(x_max + img_x), int(y_max + img_y)]
                file_name = f'./tmp/cropped/{img_name[:-4]}/cropped_image_{k}.jpg'
                data_json[file_name] = new_points
                data_confidence[file_name] = np.round(conf,4)
                cropped_image = image[y_min:y_max, x_min:x_max]
                cv.imwrite(file_name, cropped_image)
                k+=1
            if os.path.exists(f'./tmp/cropped/{img_name[:-4]}'):
                opt['Global.infer_img'] = f'./tmp/cropped/{img_name[:-4]}'  
                info,time = inference(config_path,opt)
                for data in info:
                    if (len(data['text']) != 0) and (data['confidence'] > 0.6): 
                        result_dict[img_name]['points'].append(data_json[data['img_name']])
                        result_dict[img_name]['texts'].append(data['text']) 
                        result_dict[img_name]['confidence'].append(data['confidence'])
                        result_dict[img_name]['det_confidence'].append(data_confidence[data['img_name']])
                if 'confidence' in result_dict[img_name]:
                    result_dict[img_name]['confidence'] = float(np.mean(result_dict[img_name]['confidence']) )
                else:
                    del result_dict[img_name]
        return result_dict

def extract_from_folder(data_bytes, target_folder,folder_with_imgs = 'Text/'):
    with ZipFile(io.BytesIO(data_bytes), 'r') as zip_file:
        all_files = zip_file.namelist()
        images = [f for f in all_files if f.startswith(folder_with_imgs) and not f.endswith('/')]
        if len(images) == 0:
             return False
        for image in images:
            zip_file.extract(image, target_folder)
    return True

def extract_json_from_zip(data_bytes, target_file='data.json'):
        with ZipFile(io.BytesIO(data_bytes), 'r') as z:
            if target_file in z.namelist():
                with z.open(target_file) as json_file:
                    json_data = json_file.read()
                    data_dict = json.loads(json_data)
                    return data_dict
            
def sort_text(res_dict):
    boxes = res_dict['points']
    texts = res_dict['texts']
    rec_confidence = np.round(res_dict['confidence'],4)
    det_confidence = res_dict.get('det_confidence',[-1])
    ind_sort = sorted(range(len(boxes)), key=lambda k: boxes[k][1])
    boxes_ymin_sorted = np.array([box[1] for box in boxes])[ind_sort]  
    l = 0
    while l < len(boxes):
        j = ind_sort[l]
        h = boxes[j][1] + (boxes[j][3] - boxes[j][1]) * 0.5  
        r = bisect.bisect_left(boxes_ymin_sorted,h)
        ind_sort[l:r] = sorted(ind_sort[l:r], key=lambda k: boxes[k][0])
        l = r
    sorted_texts = [texts[ind] for ind in ind_sort]
    sorted_confidences = [det_confidence[ind] for ind in ind_sort]
    full_text = ' '.join(sorted_texts)
    sorted_boxes = [boxes[ind] for ind in ind_sort]
    return {'text': full_text, 'text_boxes': list(zip(sorted_texts,sorted_boxes,sorted_confidences)),'confidence':rec_confidence}

def process_fields(det_res_image_json,file_content,bbox_dict, path = 'Image'):
    os.makedirs('./tmp/input_images', exist_ok=True) 
    is_success = extract_from_folder(file_content, './tmp/input_images',path + '/')
    if not is_success:
        return dict()
    result_dict = recognize(det_res_image_json,'./tmp/input_images/' + path + '/',bbox_dict)
    text_from_img_fields = dict()
    for img_name in result_dict:
        text_from_img_fields['res/' + path + '/' + img_name] = sort_text(result_dict[img_name])    
    shutil.rmtree('tmp')
    return text_from_img_fields

def choose_best_preds(data1,data2):
    best_preds = dict()
    for img_name in set(data1.keys()) | set(data2.keys()):
        if (img_name in data1) and (img_name not in data2 or data1[img_name]['confidence'] >= data2[img_name]['confidence']):
            best_preds[img_name] = data1[img_name]
        else:
            best_preds[img_name] = data2[img_name]
    return best_preds

async def get_det_res(url,files):
    async with httpx.AsyncClient() as client:
        response = await client.post(url,files = files, timeout = 200)
        if response.status_code == 200:
                det_res_json = response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=str(response.json()))
        return det_res_json

@app.post("/recognize")
async def recognize_text(file: UploadFile = File(...)):
    
    #try:  
        text_from_img_fields,text_from_text_fields = [],[]
        file_content = await file.read()
        files = {'file': (file.filename, file_content, file.content_type)}
        
        bbox_dict = dict()
        data_json = extract_json_from_zip(file_content)[0]
        #for patch in data_json:
        for data in data_json:
                bbox_dict[data['filename'].replace('res','./tmp/input_images')] = [int(point) for point in data['bnd_box']]
                
        task1 = asyncio.create_task(get_det_res('http://detector:8156/detect?path=Image',files))
        task2 = asyncio.create_task(get_det_res('http://detector:8156/detect?path=Text',files))

        done, pending = await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)
        queue = list(done)
        result = dict()
        while len(queue) > 0:
            task = queue.pop(0)
            if task.exception() is not None:
                raise HTTPException(status_code = 500, detail= str(task.exception(),'utf-8'))
            else:
                if task.result()['type'] == 'Image':
                    data_from_img_fields = process_fields(task.result()['det_result'],file_content,bbox_dict, path = 'Image')
                    data_from_text_fields_wo_det = process_fields(None,file_content,bbox_dict, path = 'Text')
                    result.update(data_from_img_fields,flush = True)
                else:
                    data_from_text_fields = process_fields(task.result()['det_result'],file_content,bbox_dict, path = 'Text')
            if pending:
                task2,_ = await asyncio.wait(pending)
                pending = False
                for t in task2:
                    queue.append(t)
        data_from_text_fields = choose_best_preds(data_from_text_fields,data_from_text_fields_wo_det)
        result.update(data_from_text_fields)
        return JSONResponse(content={"result": result})
    #except Exception as e:
        # raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    return JSONResponse(content={'result':"ok"})

@app.get("/temp")
async def temp(file: UploadFile = File(...)):
     return JSONResponse(content={"result": str(type(file))})

@app.get("/cuda_check")
async def cuda_check():
    import torch
    print(torch.cuda.is_available())
    print("Версия CUDA:", torch.version.cuda)
    return JSONResponse(content="ok")

@app.get("/paddle_check")
async def paddle_check():
    import paddle
    paddle.utils.run_check()
    return JSONResponse(content="ok")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8912)