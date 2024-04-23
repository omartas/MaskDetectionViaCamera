import tensorflow as tf
import keras
import cv2
from keras.models import Sequential
import keras
from keras.layers.convolutional import Convolution2D,MaxPooling2D
import numpy as np
model = tf.keras.models.load_model(
    'maskDetectionModel.h5',compile=True,
)


def guess(foto):
    sizedImage = cv2.resize(foto,(300,300),interpolation=cv2.INTER_AREA)
    image = np.expand_dims(sizedImage, axis=0)
    pred=model.predict(image)
    text=''
    pred[0][0]+=0.2
    if pred[0][0]<pred[0][1]:
        return 'Masked',pred
    else:
        return 'WithoutMask',pred



# VIde capture procces :
while True:
    camera=cv2.VideoCapture(0) # 0 numaralı kayıtlı kamerayı alma
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480) #­ 
    

    ret,image=camera.read() # kamera okumayı başlatma
    
    text,pred=guess(image)    
    
    cv2.putText(image,text,(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255,255),2,cv2.LINE_4);
    print(pred)
    cv2.imshow('Normal Goruntu',image)
    cv2.waitKey(40)

cv2.destroyAllWindows()
    
