from functions import  YOLO_loss ,nms , dataset 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import PIL.Image
import cv2
from keras import layers , Sequential
from keras.layers import Conv2D ,MaxPooling2D ,Flatten ,Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from YOLO_reshape import YOLO_reshape

S=7
B=2
C=20


model_path = './yolo_v1_fire.h5'
video_path='./Electric car catches fire, burns passenger.mp4'
iou_threshold = 0.3
cls_score_threshold=0.2
input_size=(224,224)



leaky_lelu=tf.keras.layers.LeakyReLU(alpha=0.1)
decay= l2(1e-4)
input_shape= (224,224,3)
initializer = tf.keras.initializers.HeNormal()

pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)


for layer in pre_trained_vgg.layers:
    layer.trainable=False

model= Sequential()
model.add(pre_trained_vgg)
model.add(Conv2D(filters = 1024 , kernel_size= (3,3), padding= 'same' , activation = leaky_lelu, kernel_regularizer=decay,kernel_initializer=initializer ))
model.add(Conv2D(filters = 1024 , kernel_size= (3,3), padding= 'same' , activation = leaky_lelu, kernel_regularizer=decay , strides=(2,2),kernel_initializer=initializer))

model.add(Conv2D(filters = 1024 , kernel_size= (3,3), padding= 'same' , activation = leaky_lelu, kernel_regularizer=decay ,kernel_initializer=initializer))
model.add(Conv2D(filters = 1024 , kernel_size= (3,3), padding= 'same' , activation = leaky_lelu, kernel_regularizer=decay ,kernel_initializer=initializer))

model.add(Flatten())
model.add(Dense(4096, activation = leaky_lelu, kernel_regularizer=decay,kernel_initializer=initializer))
model.add(Dropout(0.5))
model.add(Dense(S*S*(5*B+C), activation = leaky_lelu, kernel_regularizer=decay,kernel_initializer=initializer))
model.add(YOLO_reshape(target_shape= (7,7,30)))

model.summary()


model.load_weights(model_path)


cap= cv2.VideoCapture(video_path)
if cap.isOpened():

    print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))
	
while True:
    ret, img = cap.read()
    if not ret:# img로 한 프레임씩 읽고 만약 ret이 False 이면 읽기에 실패한 것이다 
        break
   


    img_input = tf.image.resize (img , input_size)
    img_input = tf.expand_dims(img_input,axis=0)

    y_pred = model.predict(img_input)


   
    
    bboxes_list,clss=nms(y_pred,iou_threshold=iou_threshold,cls_score_threshold=cls_score_threshold)

   
    for i in range(len(bboxes_list)):

        x_c,y_c,w,h = bboxes_list[i]
        if x_c==0. and y_c==0. and w==0. and h==0.:
            continue

        min_x= int((x_c-(w/2))*cap.get(3))
        min_y= int((y_c-(h/2))*cap.get(4))
        max_x = int((x_c+(w/2))*cap.get(3))
        max_y= int((y_c+(h/2))*cap.get(4))
        

        # print(i)

        cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,255,0),3)


        cls = {0:'fire',1:'unknown'}
        if clss[i]> 0:
            num=1
        else:
            num=clss[i]
        cls_name = cls[num]
        
        cv2.putText(img, cls_name, (min_x, min_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    

    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()	