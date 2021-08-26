import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

path='C:/Project files'
img = cv.imread(path+"/s.jpg")
img_color = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_for_crop = cv.cvtColor(img, cv.COLOR_BGR2RGB)

def BGR2ExG(img):
    "Excess Green Index"
    img_bgr = img_color
    b = img_bgr[...,0].astype('float32')
    g = img_bgr[...,1].astype('float32')
    r = img_bgr[...,2].astype('float32')
    ExG = 2 * g - r - b  
    return ExG
img_ExG = BGR2ExG(img_color)
# pix = img_ExG[100,100]
# print(pix)










