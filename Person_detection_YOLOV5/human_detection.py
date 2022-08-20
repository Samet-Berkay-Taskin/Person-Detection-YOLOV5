# -*- coding: utf-8 -*-
"""
Created on 19 August 2022

@author: Berkay
"""

# libraries
import torch
import torch.backends.cudnn as cudnn
import os
import yaml
import cv2
from PIL import Image


# A class for human detection


class HumanDetection:

    def __init__(self, model_path, conf):
        """
        model_path: Trainedmodel path
        conf: Modelpredictthresholdnumber.
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = conf
    
    def show_detection(self, imgs):
        """
        Display of detected humans in the uploaded image
        """
        self.img = imgs
        self.results = self.model(self.img, size=512)
        return self.results.show()

    def detection(self, path: str):
        """
        Returns the x and y coordinates of the humans detected in the received image.
        """
        self.results = self.model(path, size=512)
        results_df = self.results.pandas().xyxy[0]
        if len(results_df) != 0:
            results_df.iloc[:, :4] = results_df.iloc[:, :4].apply(lambda x: x.astype("int"))
            results_df.drop("name", axis=1, inplace=True)
            results_df.rename(columns={"class": "human_class"}, inplace=True)

            results_list = [dict(results_df.loc[i, :]) for i in range(len(results_df))]
            for i, _ in enumerate(results_list):
                results_list[i]["xmin"] = int(results_list[i]["xmin"])
                results_list[i]["ymin"] = int(results_list[i]["ymin"])
                results_list[i]["xmax"] = int(results_list[i]["xmax"])
                results_list[i]["ymax"] = int(results_list[i]["ymax"])
                results_list[i]["confidence"] = float(results_list[i]["confidence"])
                results_list[i]["human_class"] = float(results_list[i]["human_class"])
            return results_list
        else:

            return [{}]





