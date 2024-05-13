import ultralytics
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.files import increment_path
from pathlib import Path
import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import shutil
import torch
import numpy as np
from PIL import Image
from zipfile import ZipFile
import bisect


torch.cuda.set_device(0)
app = FastAPI()


def extract_json_from_zip(source_file="./res_data.zip", target_file='data.json'):
    with ZipFile(source_file, 'r') as z:
        if target_file in z.namelist():
            with z.open(target_file) as json_file:
                json_data = json_file.read()
                data_dict = json.loads(json_data)
                return data_dict
            

def delete_artifacts():
    if os.path.exists("./res_data.zip"):
        os.remove("./res_data.zip")
    shutil.rmtree("./res", ignore_errors=True) 


def detect_GUI(image_path, file):
    weights = './YOLOv9_finetuned_30epochs.pt'
    model = ultralytics.YOLO(weights)
    results = model([image_path], conf=0.5, device=[2])
    # results = model([image_path], conf=0.05, iou=0.55, device=[2])
    return results


def identify_patches(res, num_squares_ax=3):
    height, width = res.orig_shape
    height_sq, width_sq = height // num_squares_ax, width // num_squares_ax
    xywh_bnd_box = res.boxes.xywh.cpu().numpy()
    x_center, y_center = xywh_bnd_box[:, 0], xywh_bnd_box[:, 1]
    sq_ind_x = x_center.astype(int) // width_sq
    sq_ind_y = y_center.astype(int) // height_sq
    return np.concatenate([sq_ind_x[:, np.newaxis], sq_ind_y[:, np.newaxis]], axis=1) 


def geom_sort(info):
    ind_sort = sorted(range(len(info)), key=lambda k: (info[k]["bnd_box"][1] + info[k]["bnd_box"][3]) / 2)
    boxes_ycenter_sorted = np.array([(obj["bnd_box"][1] + obj["bnd_box"][3]) / 2 for obj in info])[ind_sort]  
    l = 0
    while l < len(info):
        j = ind_sort[l]
        h = info[j]["bnd_box"][3]  
        r = bisect.bisect_left(boxes_ycenter_sorted, h)
        ind_sort[l:r] = sorted(ind_sort[l:r], key=lambda k: (info[k]["bnd_box"][0] + info[k]["bnd_box"][2]) / 2)
        l = r
    sorted_info = [info[ind] for ind in ind_sort]
    return sorted_info


def gen_output(results):
    results_data = []
    os.mkdir("./res")
    for res in results:
        # save data to json
        cls2name = res.names
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        num_squares_ax = 3
        patches = identify_patches(res, num_squares_ax=num_squares_ax)
        res_data = {}

        for it1 in range(num_squares_ax):
            for it2 in range(num_squares_ax):
                patch = str([it1, it2])
                res_data[patch] = []

        for obj_ind in range(len(classes)):
            obj_dict = dict()
            obj_dict["bnd_box"] = boxes[obj_ind, :].tolist()
            obj_dict["cls"] = cls2name[classes[obj_ind]]
            if obj_dict["cls"] == "Background Image":
                obj_dict["cls"] = "Image"

            # generate filename for bndbox image
            filename = str(increment_path(Path("./res/" + obj_dict["cls"] + "/im_.jpg")).with_suffix('.jpg'))

            # save bndbox image
            save_one_box(
                torch.tensor(boxes[obj_ind, :]),
                res.orig_img.copy(),
                file=Path(filename),
                BGR=True,
            )
            obj_dict["filename"] = filename

            res_data[str(patches[obj_ind, :].tolist())].append(obj_dict)

        results_data.append(res_data)
    
    # sort objects inside patches
    for it1 in range(num_squares_ax):
        for it2 in range(num_squares_ax):
            patch = str([it1, it2])
            results_data[0][patch] = geom_sort(results_data[0][patch])

    with open('./data.json', 'w') as f:
        json.dump(results_data, f)
    
    # move json to out directory
    os.rename("./data.json", "./res/data.json")

    # form zip archive
    shutil.make_archive("./res_data", "zip", "./res")

    
@app.post("/detect_GUI")
async def detect(file: UploadFile = File(...)):
    try:
        with open(f"./tmp.jpg", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # inference 
        gen_output(detect_GUI(f'./tmp.jpg', file))

        # form response
        with open("./res_data.zip", "rb") as file:
            file_content = file.read() 
        headers = {
            "Content-Disposition": f"attachment; filename=./res_data.zip"
        }
        return Response(content=file_content, media_type="application/zip", headers=headers)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        delete_artifacts()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8873)