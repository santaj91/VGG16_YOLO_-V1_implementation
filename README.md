# YOLO_V1_implementation

Python = 3.7.13 , Tensorflow = 2.8.2

This model is a fire detection model implementated by yolo v1 paper.
but it has very poor performance because of high loss values.
I tried so many times, but it didn't work well

![img_2022-07-12_16-01-39_AdobeExpress](https://user-images.githubusercontent.com/93965016/178448209-0aaf5eda-6f80-4f87-84af-4c5bd03e9e30.gif)
![fire (2)](https://user-images.githubusercontent.com/93965016/178639262-078d8b3f-8555-49c6-8b92-cc6a4d54e33d.gif)



# model

At first, i had built original YOLO_v1 layers in paper by a way of trial. as expected it was not trained well. <br> So i took VGG16 model at keras and then concatenated with yolo's inference layers.<br>
I trained this model for almost 24 hours, but loss values have not went down from 3 for some hours and also even i trained only one class, but sometimes it predict other class( i named it unknown) <br>
I'm still improving this model.

# dataset

I trained this with almost 5,900 fire pictures.<br>

https://www.kaggle.com/datasets/phylake1337/fire-dataset <br>
https://www.kaggle.com/datasets/ankan1998/fire-detection-in-yolo-format<br>
https://github.com/spacewalk01/Yolov5-Fire-Detection

# reference

https://arxiv.org/abs/1506.02640 <br>
https://www.maskaravivek.com/post/yolov1/ <br>
https://velog.io/@minkyu4506/YOLO-v1-%EB%A6%AC%EB%B7%B0-%EC%BD%94%EB%93%9C-%EA%B5%AC%ED%98%84tensorflow2
