import asyncio
import logging
import sys
import requests

from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile, FSInputFile

from gtts import gTTS
from pydub import AudioSegment
import httpx

from io import BytesIO
from zipfile import ZipFile
import json
import copy

import os

TOKEN = "7060923400:AAGvMcMVtyWgCza8GAFeanfmCIMpCYJYajw"

dp = Dispatcher()
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


def extract_json_from_zip(data_bytes, target_file='data.json'):
    with ZipFile(BytesIO(data_bytes), 'r') as z:
        if target_file in z.namelist():
            with z.open(target_file) as json_file:
                json_data = json_file.read()
                data_dict = json.loads(json_data)
                return data_dict

def form_final_json(meta_json):
    output_json = {}
    class_item_counter = {}
    for patch in meta_json["det_res_json"]:
        class_item_counter[patch] = {}
        for obj_per_patch in meta_json["det_res_json"][patch]:
            if obj_per_patch["filename"] in meta_json["ocr_res_json"]:
                obj_per_patch["ocr"] = meta_json["ocr_res_json"][obj_per_patch["filename"]]

            if obj_per_patch["filename"] in meta_json["age_res_json"]:
                obj_per_patch["people_estimation"] = meta_json["age_res_json"][obj_per_patch["filename"]]

            if obj_per_patch["filename"] in meta_json["cap_res_json"]:
                obj_per_patch["caption"] = meta_json["cap_res_json"][obj_per_patch["filename"]]

            if obj_per_patch["filename"] in meta_json["cartoon_res_json"]:
                obj_per_patch["is_cartoon"] = meta_json["cartoon_res_json"][obj_per_patch["filename"]]


            if obj_per_patch["cls"] in class_item_counter[patch]:
                class_item_counter[patch][obj_per_patch["cls"]] += 1
            else:
                class_item_counter[patch][obj_per_patch["cls"]] = 1

    return meta_json["det_res_json"], class_item_counter


def form_answer(res_json, class_item_counter):
    answer = ""
    def translate_position(pos):
        translations = {
            "[0, 0]": "In the upper left corner",
            "[0, 1]": "At the top center",
            "[0, 2]": "In the upper right corner",
            "[1, 0]": "On the left side in the middle",
            "[1, 1]": "In the center of the image",
            "[1, 2]": "On the right side in the middle",
            "[2, 0]": "In the bottom left corner",
            "[2, 1]": "At the bottom center",
            "[2, 2]": "In the bottom right corner"
        }
        return translations.get(pos)

    def classify_age(age_pred, gender_pred):
        output_str = f"{age_pred} years old "
        if age_pred < 18:
            if gender_pred == 'male':
                output_str += 'boy'
            elif gender_pred == 'female':
                output_str += 'girl'
            else:
                output_str += 'person'
        else:
            if gender_pred == 'male':
                output_str += 'man'
            elif gender_pred == 'female':
                output_str += 'woman'
            else:
                output_str += 'person'
        output_str += '; '
        return output_str


    for position, items in res_json.items():
        location = translate_position(position)
        count = len(items)
        if count > 0:
            answer += '    '
            if count == 1:
                answer += f"{location} there is {count} user interface element:\n"
            else:
                answer += f"{location} there are {count} user interface elements:\n"

            # TODO: количество элементов классов

            for cls_name, cls_cnt in class_item_counter[position].items():
                if cls_cnt == 1:
                    answer += f"{cls_cnt} {cls_name}; "
                elif cls_cnt > 1:
                    answer += f"{cls_cnt} {cls_name}s; "

            answer += '\n'
            is_first_obj = True
            for item in items:
                if "caption" in item or "ocr" in item: 
                    if is_first_obj:
                        answer += "    First "
                        is_first_obj = False   
                    else:
                        answer += "    Next " 

                if "caption" in item:
                    if "is_cartoon" in item:
                        answer += "cartoonish " if item["is_cartoon"] else "realistic "
                    answer += f"image displays: {item['caption']}. "

                    if "people_estimation" in item and not item["is_cartoon"]:
                        people_estimations = item.get("people_estimation")

                        if len(people_estimations) > 1:
                                answer += "There are "
                        else:
                            answer += "There is "

                        for person in people_estimations:   
                            res_age = round(person.get('age'))
                            res_gender = person.get('gender')
                            answer += classify_age(res_age, res_gender)
                    
                    if "ocr" in item:
                        answer += f"And there is text element in it: {item['ocr'].get('text')}.\n"
                    else:
                        answer += '\n'

                elif "ocr" in item:
                    answer += f"text element contains: \"{item['ocr'].get('text')}\"\n"
            answer += '\n'
    return answer

def detect_language(text):
    # Примитивное определение языка по наличию кириллицы
    if any('а' <= ch <= 'я' or 'А' <= ch <= 'Я' for ch in text):
        return 'ru'
    return 'en'

def create_multilingual_speech(text, filename):
    # Разделяем текст на слова
    words = text.split()
    
    # Создаём пустой аудиофайл
    full_audio = AudioSegment.silent(duration=0)
    
    current_lang = detect_language(words[0])
    current_text_segment = []
    
    for word in words:
        word_lang = detect_language(word)
        
        # Проверяем, соответствует ли язык слова текущему фрагменту
        if word_lang == current_lang:
            current_text_segment.append(word)
        else:
            # Создаём аудио для текущего фрагмента
            tts = gTTS(' '.join(current_text_segment), lang=current_lang)
            segment_filename = 'temp_segment.mp3'
            tts.save(segment_filename)
            segment_audio = AudioSegment.from_mp3(segment_filename)
            full_audio += segment_audio
            
            # Начинаем новый фрагмент
            current_lang = word_lang
            current_text_segment = [word]
    
    # Обработка последнего фрагмента
    tts = gTTS(' '.join(current_text_segment), lang=current_lang)
    segment_filename = 'temp_segment.mp3'
    tts.save(segment_filename)
    segment_audio = AudioSegment.from_mp3(segment_filename)
    full_audio += segment_audio
    
    # Экспортировать результат в один файл
    full_audio.export(filename, format="mp3")
    
    # Удаляем временный файл
    os.remove(segment_filename)



@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")

@dp.message(F.photo)
async def photo_handler(message: Message) -> None:
    # try:
    file_id = message.photo[-1].file_id
    
    #----------------------------------- Сохранение картинки ботом 
    file = await bot.get_file(file_id)                        # FIRST
    downloaded_file = await bot.download_file(file.file_path)
    input_image = {'file': ("boba_"+file.file_path.split('/')[-1], BytesIO(downloaded_file.read()), 'image/jpeg')}
    
    url_ocr = 'http://recognizer:8912/recognize'
    url_age = 'http://age_estimator:6175/age_estimate'
    url_cap = 'http://caption:3220/caption'
    url_det = 'http://GUI_detector:8873/detect_GUI'

    #-------------- cartoon
    url_cartoon = 'http://cartoon_or_not:3219/cartoon_or_not'
    #--------------

    #------------------------------------ Чтение zip из локалки
    # file_path = './detect_tg.zip'
    # with open(file_path, 'rb') as file:
    #     files = {'file': ('filename.zip', file.read(), 'application/zip')}

    meta_json = {"det_res_json": {},
                 "ocr_res_json": {},
                 "age_res_json": {},
                 "cap_res_json": {},
                 "cartoon_res_json": {}}
                 
    async with httpx.AsyncClient() as client:
        response_det = await client.post(url_det, files=copy.deepcopy(input_image), timeout=60)
        if response_det.status_code == 200:
            det_res_zip = response_det.content
            files = {'file': ('filename.zip', det_res_zip, 'application/zip')}
            meta_json["det_res_json"] = extract_json_from_zip(det_res_zip)[0]
            # await message.answer(f"Результат det: {extract_json_from_zip(det_res_zip)[0]}")
            # await message.answer(f"Результат det: OK")
        else:
            await message.answer("det " + response_det.text)

        #-------------- cartoon
        cartoon_corutine = client.post(url_cartoon, files=copy.deepcopy(files), timeout=300)    # TODO timeout->var
        # #--------------

        ocr_corutine =  client.post(url_ocr, files=copy.deepcopy(files), timeout=300)
        age_corutine =  client.post(url_age, files=copy.deepcopy(files), timeout=300)
        cap_corutine =  client.post(url_cap, files=copy.deepcopy(files), timeout=300)
        
        #-------------- cartoon
        response_ocr, response_age, respones_cap, respones_cartoon = await asyncio.gather(ocr_corutine, age_corutine, cap_corutine, cartoon_corutine)
        #
        #--------------
        if response_ocr.status_code == 200:
            ocr_res_json = response_ocr.json()
            
            meta_json["ocr_res_json"] = ocr_res_json['result']
            # await message.answer(f"Результат ocr: {ocr_res_json['result']}") 
            # await message.answer_voice(voice_file, caption="Вот ваше аудиосообщение из ocr!")    

        else:
            await message.answer("ocr " + response_ocr.text)
        
        if response_age.status_code == 200:
            age_res_json = response_age.json() 
            meta_json["age_res_json"] = age_res_json['age_gender_result']
            # await message.answer(f"Результат age: {age_res_json['age_gender_result']}")       
        else:
            await message.answer("age " + response_age.text)

        if respones_cap.status_code == 200:
            cap_res_json = respones_cap.json()
            meta_json["cap_res_json"] = cap_res_json['caption_result']
            # await message.answer(f"Результат cap: {cap_res_json['caption_result']}")          
        else:
            await message.answer("cap " + respones_cap.text)

        #-------------- cartoon
        if respones_cartoon.status_code == 200:
            cartoon_res_json = respones_cartoon.json()
            meta_json["cartoon_res_json"] = cartoon_res_json['cartoon_or_not_result']
            # await message.answer(f"Результат cartoon: {cartoon_res_json['cartoon_or_not_result']}")          
        else:
            await message.answer("cartoon " + respones_cartoon.text)

    # print("FIN JSON", form_final_json(meta_json), flush=True)

    final_json, class_item_counter = form_final_json(meta_json)                     # TODO prettify

    final_answer = form_answer(final_json, class_item_counter)
    # await message.answer(f"{final_answer}")
    await message.reply(f"{final_answer}")

    #---------------- simple voice 
    # voice = gTTS(final_answer, lang='ru') 
    # file_in_bytes = b''.join([byte for byte in voice.stream()])
    # voice_file = BufferedInputFile(file_in_bytes, filename='ocr voice')
    #-----------------
    create_multilingual_speech(final_answer, "multilingual_speech.mp3")
    voice_file = BufferedInputFile.from_file(path = "multilingual_speech.mp3", filename='voice')

    await message.reply_voice(voice_file, caption="Audio")    

    with open('data.json', 'w', encoding='utf8') as json_file:
        json.dump(final_json, json_file, indent=4, ensure_ascii=False)

    # with open('data.json', 'rb') as json_file:
    #     await message.reply_document(document=json_file, caption="Your JSON file:")

    await message.reply_document(document=FSInputFile('data.json'))

async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
