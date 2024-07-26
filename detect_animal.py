from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image
import numpy as np
import io
import json
import csv

class ClassificationURL:
    def __init__(self, key, endpoint, id, model):
        self.prediction_key = key
        self.prediction_endpoint = endpoint
        self.project_id = id
        self.model_name = model

        credentials = ApiKeyCredentials(in_headers={"Prediction-key": self.prediction_key})
        self.predictor = CustomVisionPredictionClient(endpoint=self.prediction_endpoint, credentials=credentials)

dic_classification = {
    'tiger' : ClassificationURL(
        'cc4e7910294b4d3bb0cd9e59c9a1fe33',
        'https://b021customvision-prediction.cognitiveservices.azure.com/',
        'ae6e8a0e-125e-458e-bc7b-415740df5c8f',
        'Iteration1'
    ),
    'panda' : ClassificationURL(
        'cc4e7910294b4d3bb0cd9e59c9a1fe33',
        'https://b021customvision-prediction.cognitiveservices.azure.com/',
        '8791e920-0312-40f3-83ba-e650eeedf908',
        'Iteration1'
    ),
    'zebra' : ClassificationURL(
        'cc4e7910294b4d3bb0cd9e59c9a1fe33',
        'https://b021customvision-prediction.cognitiveservices.azure.com/',
        'd072742e-fbfd-4178-b302-485b5503200e',
        'Iteration5'
    ) 
}

prediction_key = "cc4e7910294b4d3bb0cd9e59c9a1fe33"
prediction_endpoint = "https://b021customvision-prediction.cognitiveservices.azure.com/"
project_id = "46eb142e-665d-45dc-8942-41ab48c89225"
model_name = "Iteration2"

credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

pass_probability = 0.8
classfy_probability = 0.8

db_path = 'DB_animal.csv'

def ndarr_to_bytearr(nd_arr, format):
    byte_arr = io.BytesIO()
    nd_arr.save(byte_arr, format=format)
    byte_arr = byte_arr.getvalue()
    return byte_arr

def split_detect_image(image, predict_results):
    h, w, ch = np.array(image).shape

    detections = []
    detect_name = []
    for prediction in predict_results. predictions:
        if prediction.probability > pass_probability:
            #print(f'{float(prediction.probability*100.0):.2f}')
            left = prediction.bounding_box.left * w
            top = prediction.bounding_box.top * h
            right = left + prediction.bounding_box.width * w
            bottom = top + prediction.bounding_box.height * h

            cropped_img = image.crop((left, top, right, bottom))
            detections.append(cropped_img)
            detect_name.append(prediction.tag_name)
            break
    return detect_name, detections

def animal_classify(names, detect_imgs, img_format):
    if len(names) == 0:
        return names
    
    url_info = dic_classification.get(names[0])
    if url_info == None:
        return names

    detect_name = []
    image_data = ndarr_to_bytearr(detect_imgs[0], img_format)
    results = url_info.predictor.classify_image(url_info.project_id, url_info.model_name, image_data)
    for prediction in results.predictions:
        if prediction.probability > classfy_probability:
            detect_name.append(prediction.tag_name)
            break
    return detect_name

def animal_detect(image):
    image_data = ndarr_to_bytearr(image, image.format)
    results = predictor.detect_image(project_id, model_name, image_data)
    tags, detections = split_detect_image(image, results)
    names = animal_classify(tags, detections, image.format)
    return names

def get_animal_info(name):
    info = ''
    with open(db_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['ID'] == name:
                info = f"{row['ID']},{row['species']},{row['name']},{row['birthday']},{row['attribute']},{row['info']}"
                break
    return info

def c_to_python(dir):
    test_img = Image.open(dir)
    detect_img = animal_detect(test_img)
    if len(detect_img):
        print('fail to detect')

    infos = []
    for name in detect_img:
        print('name: ', name)
        infos.append(get_animal_info(name))

    return infos

# if __name__=='__main__':
#     c_to_python(sys.argv[1])