# -*- coding: utf-8 -*-
"""
Created on 19 August 2022

@author: Berkay
"""


# libraries
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

# this is the request body


class Object(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float
    human_class: float


class IImage(BaseModel):
    url: HttpUrl
    name: str
    type: str
