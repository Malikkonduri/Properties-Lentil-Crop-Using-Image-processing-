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

def BGR2MExG(img):
    "Modified Excess Green Index"
    B = img[...,0].astype('float32')
    G = img[...,1].astype('float32')
    R = img[...,2].astype('float32')
    MExG = 1.262 * G - 0.884 * R - 0.311 * B
    return MExG
img_MExG = BGR2MExG(img_color)



def BGR2CIVE(img):
    "Colour Index of Vegetation Extraction"
    B = img[...,0].astype('float32')
    G = img[...,1].astype('float32')
    R = img[...,2].astype('float32')
    CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    return CIVE
img_CIVE = BGR2CIVE(img_color)

def MExGCIVE(i):
    osimg = BGR2MExG(i) - BGR2CIVE(i)
    return osimg

img_MExGCIVE = MExGCIVE(img)

def croped(im1,im2):
    x,y,z = im1.shape
    for i in range (x):
        for j in range (y):
            pix = im2[i,j]
            if pix<0:
                im1[i,j]=[0,0,0]
            else:
                im1[i,j]=im1[i,j]
    
    return im1

cropped = croped(img_for_crop,img_MExGCIVE)

ret, threshExG = cv.threshold(img_ExG,43,255, cv.THRESH_BINARY)
ret, threshCIVE = cv.threshold(img_CIVE,9,255, cv.THRESH_BINARY_INV)
ret, threshMExG = cv.threshold(img_MExG,40,255, cv.THRESH_BINARY)
ret, threshMExGCIVE = cv.threshold(img_MExGCIVE,0,255, cv.THRESH_BINARY)


thresh = threshMExGCIVE
i = img_MExGCIVE

#Normalization of Image

norm_img = np.zeros((0,800))
final_img = cv.normalize(cropped,  norm_img, 0, 255, cv.NORM_MINMAX)

titles = ['Original', 'cropped image', 'Norm image']
images = [img_color, cropped, final_img]
for i in range(len(images)):
    plt.figure(titles[i])
    plt.imshow(images[i], cmap='gray')
plt.show()

# Vegitative indix
def VI(img):
    "Colour Index of Vegetation Extraction"
    b = img[...,0].astype('float32')
    g = img[...,1].astype('float32')
    r = img[...,2].astype('float32')
    t=r+g+b

    NR = r / t
    NG=g/t
    NB=b/t
    ExR=((1.4*r)-g)/t
    ExB=((1.4*b)-g)/t
    ExGR =(3*g-2.4*r-b) /t
    GBD = g-b
    RBD = r-g
    RGD=r-g
    GRR=g/r
    GBR=g/b
    NGRD = (g-r)/(g+r)
    NGBD = (g-b)/(g+b)
    MNGRD =((g*g)-(r*r))/((g*g)+(r*r))
    VD =(2*g-b-r)/(2*g+b+r)
    RGBVI = ((g*g)-(b*r))/ ((g*g)+(b*r))
    CI = (2*b)/(r+b)
    CIVE= 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
    TGI =95*g - 35*r -60*b
    MExG =1.262*g - 0.884*r -0.311*b

    return [NR,NG,NB,ExR,ExB,ExGR,GBD,RBD,RGD,GRR,GBR,NGRD,NGBD,MNGRD,VD,RGBVI,CI,CIVE,TGI,MExG]

title =['Normalized Red','Normalized green','Normalized blue','Excess red','Excess blue', 'Excess green red', 'Green blue difference','Red blue difference','Red green difference','Green red ratio','Green blue ratio','Normalized green red difference','Normalized green blue difference','Modified Normalized green red difference','Visible band difference','Red green blue vegitation index','Crust index','Color index of vegitation index','Triangular greenness index','Modifed excess green']
imgs = VI(final_img)

for i in range(len(title)):
    plt.figure(title[i])
    plt.imshow(imgs[i])
plt.show()

#Normalized red
#


def VegitativeIndexValues(cropImage, ImagesArray):
    # array 1 contains title {normalized red,normalized green}
    # array 2 need to add every vi value
    a=[]
    for u in ImagesArray:
        img=u
        x, y = img.shape
        n=0
        sum=0
        for i in range(x):
            for j in range(y):
                pix = cropImage[i, j]
                if pix > 0:
                    n=n+1
                    sum=sum+img[i,j]
        average = sum/n
        a.append(average)
    return a

VI_values = VegitativeIndexValues(img_MExGCIVE, imgs)
for i in VI_values:
    print(i)








