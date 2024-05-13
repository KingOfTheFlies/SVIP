import argparse
# import logging
import os

import cv2
import torch
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from zipfile import ZipFile

# model params
output_folder = "output"
detector_weights = "models/yolov8x_person_face.pt"
checkpoint_path = "models/model_imdb_person_fin.pth.tar"
device = "cuda"
with_persons = True
disable_faces = False
draw = False
verbose = False
folder_with_imgs = "Image/"

GENDER_CONFIDENCE_THRESHOLD = 0.94


args_dict = {
    'output': output_folder,
    'detector_weights': detector_weights,
    'checkpoint': checkpoint_path,
    'with_persons': with_persons,
    'disable_faces': disable_faces,
    'draw': draw,
    'device': device,
    'verbose': verbose
}

# _logger = logging.getLogger("inference")

app = FastAPI()

def get_parser(input_file):
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--input", type=str, default=input_file, required=False, help="image file or folder with images")
    parser.add_argument("--output", type=str, default=args_dict["output"], required=False, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default=args_dict["detector_weights"], required=False, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default=args_dict["checkpoint"], type=str, required=False, help="path to mivolo checkpoint")
    parser.add_argument(
        "--with-persons", action="store_true", default=args_dict["with_persons"], help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=args_dict["disable_faces"], help="If set model will use only persons if available"
    )
    parser.add_argument("--draw", action="store_true", default=args_dict["draw"], help="If set, resulted images will be drawn")
    parser.add_argument("--device", default=args_dict["device"], type=str, help="Device (accelerator) to use.")
    return parser


def extract_images_only(zip_path, target_folder):
    with ZipFile(zip_path, 'r') as zip_file:
        all_files = zip_file.namelist()
        images = [f for f in all_files if f.startswith(folder_with_imgs) and not f.endswith('/')]
        for image in images:
            zip_file.extract(image, target_folder)

def format_json_response(data):
    response = {}
    for filename, detections in data.items():
        file_info = {
            'filename': filename,
            'detected_people': []
        }
        for person_id, details in detections.items():
            person_info = {
                'person_id': person_id,
                'face_index': details['face_ind'],
                'body_index': details['body_ind'],
                'age': details['age'],
                'gender': details['gender'],
                'gender_confidence': details['gender_score'],
                'face_boundaries': details['face_boundings'],
                'body_boundaries': details['body_boundings']
            }
            file_info['detected_people'].append(person_info)
        response[filename] = file_info.copy()

    return response

def process_age_model_to_output(data):
    output_res = {}

    def calculate_area(boundaries):
        if boundaries and isinstance(boundaries[0], int) and len(boundaries) == 4:
            width = boundaries[2] - boundaries[0]
            height = boundaries[3] - boundaries[1]
            return width * height
        return 0

    for key in data.keys():
        sorted_people = sorted(
            data[key]['detected_people'],
            key=lambda x: max(
                calculate_area(x['face_boundaries']) if x['face_index'] is not None else 0,
                calculate_area(x['body_boundaries']) if x['body_index'] is not None else 0
            ),
            reverse=True
        )

        filtered_people = []
        for person in sorted_people:
            if person["face_index"] is not None:
                filtered_people.append(
                    {
                        'age': person['age'],
                        'gender': person['gender'] if person['gender_confidence'] >= GENDER_CONFIDENCE_THRESHOLD else "undefined",
                        'bounding_box_face': person['face_boundaries'],
                        'bounding_box_body': person['body_boundaries'] if person['body_index'] is not None else []
                    })
        if filtered_people:
            output_res[key] = filtered_people
    response = {'age_gender_result': output_res}
    
    return JSONResponse(content=response)

@app.post("/age_estimate")
async def inference(file: UploadFile = File(...)):
    temp_dir = "./temp"
    standart_folder = "res/"
    os.makedirs(temp_dir, exist_ok=True)

    # Сохранение zip file локально
    temp_zip_path = f"{temp_dir}/{file.filename}"
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #images dir
    images_dir = f"{temp_dir}/input_images/"
    os.makedirs(images_dir, exist_ok=True)
    # extracting zip to images_dir
    extract_images_only(temp_zip_path, images_dir)

    parser = get_parser(images_dir)
    # setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    input_type = get_input_type(args.input)
    predictor = Predictor(args, verbose=args_dict['verbose'])
    files_results = {}
    if input_type == InputType.Image:
        image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input]
        for img_p in image_files:
            img = cv2.imread(img_p)
            detected_objects, out_im, cur_result = predictor.recognize(img)
            if args.draw:
                bname = os.path.splitext(os.path.basename(img_p))[0]
                filename = os.path.join(args.output, f"out_{bname}.jpg")
                cv2.imwrite(filename, out_im)
                # _logger.info(f"Saved result to {filename}")

            if cur_result:
                files_results[standart_folder + img_p[len(images_dir):]] = cur_result.copy()
    else:
        raise ValueError("Unsupported input file type")
        shutil.rmtree(temp_dir)

    shutil.rmtree(temp_dir)
    preprocessed_files_result = format_json_response(files_results)         # full info json
    return process_age_model_to_output(preprocessed_files_result)           # эвристика вывода

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6175)