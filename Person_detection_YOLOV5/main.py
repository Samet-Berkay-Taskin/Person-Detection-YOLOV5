# -*- coding: utf-8 -*-
"""
Created on 19 August 2022

@author: Berkay
"""


# libraries
import numpy as np
import cv2
import tensorflow as tf
import keras
import io
import yaml
import torch
from fastapi import Depends, FastAPI, File, UploadFile
from PIL import Image
from fastapi.responses import JSONResponse, RedirectResponse
from human_detection import HumanDetection
from utilities import Object, IImage
from keras.preprocessing import image

# Read config.yaml.In config.yaml there is the model's path and thresh value
with open("config.yml", "r") as f:
    conf = yaml.safe_load(f)

# Working with CPU or GPU check
gpu = torch.cuda.is_available()
if gpu:
    print("GPU detected, you are working on GPU")
else:
    print("You are working on CPU")

Human = HumanDetection(conf["Data"]["path"], conf["Data"]["conf_thresh"])


"""If you want to use the trained model locally, the following codes are enough. """
# image = keras.utils.load_img('image_name.jpg')
# result = Human.show_predict(image)

app = FastAPI()


@app.get("/")
async def root():
    return {"Welcome to Person Detection"}


@app.post("/detection/")
async def show_detection_image(file: UploadFile = File(...)):
    try:
        imgs = file.file.read()
        image = Image.open(io.BytesIO(imgs))
        result = Human.show_detection(image)
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"detail": "Not uploaded image."})


@app.post("/detection_path")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image1 = np.frombuffer(contents, np.uint8)
        image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image_arr = np.array(image1, dtype=np.float32)
        if (image_arr.shape[0] < 200) or (image_arr.shape[1] < 200):
            return JSONResponse(status_code=400,
                                content={"detail": "The shapes of the image must be greater than (200, 200)."})
    except Exception as err:
        return JSONResponse(status_code=400,
                            content={"detail": "Not uploaded image."})
    result = Human.detection(image_arr)
    object_list = [i for i in result]
    if bool(result[0]):
        found_object = len(result)
        return object_list
    else:
        found_object = 0
    rresult = {"found_object": found_object,
               "object_list": object_list}
    return "There is no human", rresult


@app.post("/object/")
async def request_image_object(item: Object):
    return item


@app.post('/image/')
async def image(item: IImage):
    return item


