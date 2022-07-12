import tensorflow as tf
import numpy as np

from keras.utils import img_to_array, load_img
#iou_list, IOU , YOLO_loss , nms ,dataset

def dataset(train_image_path_list):
    result = []

    for image_path in train_image_path_list:
        image = load_img(image_path,target_size = (448,448))
        input_arr = img_to_array(image).astype(np.float32)/255.0
        result.append(input_arr)
    result =np.reshape(result,(-1,448,448,3))
    return result


def iou_list (list1,list2):

  x_c1,y_c1,w1,h1 = list1
  x_c2,y_c2,w2,h2 = list2
  x1_min= x_c1-(w1/2)
  x1_max= x_c1+(w1/2)
  y1_min = y_c1-(h1/2)
  y1_max= y_c1+(h1/2)

  x2_min= x_c2-(w2/2)
  x2_max= x_c2+(w2/2)
  y2_min = y_c2-(h2/2)
  y2_max= y_c2+(h2/2)

  intersect_min_x = max(x2_min,x1_min)
  intersect_max_x = min(x2_max,x1_max)
  intersect_min_y = max(y2_min,y1_min)
  intersect_max_y = min(y2_max,y1_max)

  intersect_w = max(intersect_max_x-intersect_min_x,0)
  intersect_h = max(intersect_max_y-intersect_min_y,0)

  intersect_area= intersect_w*intersect_h

  area1 = w1*h1
  area2= w2*h2

  union_area= area1 + area2 - intersect_area

  iou = intersect_area / union_area

  return iou

def IOU(label_box,pred_box,size=224):
    # input1 = n 7 7 1 4
    # input2 = n 7 7 1 4
    
    label_xy = label_box[...,:2]* size # n 7 7 1 2
    label_wh = label_box[...,2:4]* size # n 7 7 1 2
    pred_xy = pred_box[...,:2]* size # n 7 7 1 2
    pred_wh =  pred_box[...,2:4]* size # n 7 7 1 2

    label_min_xy = tf.math.subtract(label_xy , label_wh/2) # n 7 7 1 2
    label_max_xy = tf.math.add(label_xy , label_wh/2)

    pred_min_xy = tf.math.subtract(pred_xy , pred_wh/2)
    pred_max_xy = tf.math.add(pred_xy , pred_wh/2)

    intersect_mins = tf.math.maximum (label_min_xy,pred_min_xy) # n 7 7 1 2
    intersect_maxs = tf.math.minimum (label_max_xy,pred_max_xy)

    intersect_wh =  tf.math.maximum (intersect_maxs-intersect_mins,0)# 음수는 intersect =0
    intersect_areas = intersect_wh[...,0]*intersect_wh[...,1] # n 7 7 1

    true_areas = label_wh[...,0]*label_wh[...,1]# n 7 7 1 
    pred_areas = pred_wh[...,0]*pred_wh[...,1]# n 7 7 1

    union_areas= true_areas + pred_areas - intersect_areas# n 7 7 1
    iou = intersect_areas / union_areas# n 7 7 1
 
    return iou # n 7 7 1


def YOLO_loss(y_true,y_pred):
    label_cls = y_true[...,10:] # n 7 7 20
    label_box = y_true[...,1:5] # n 7 7 4
    confidence = y_true[...,0] # n 7 7
    confidence_mask = tf.expand_dims(confidence,axis=3)# n 7 7 1

    predict_cls = y_pred[...,10:] # n 7 7 20
    predict_trust1= tf.expand_dims(y_pred[...,0], axis = 3)# n 7 7 1
    predict_trust2= tf.expand_dims(y_pred[...,5],axis = 3)# n 7 7 1
    predict_trust = tf.concat([predict_trust1,predict_trust2], axis= 3)# n 7 7 2

    bbox1= y_pred[...,1:5]# n 7 7 4
    bbox2= y_pred[...,5:9]# n 7 7 4

    _label_box = tf.reshape(label_box,[-1,7,7,1,4])
    _bbox1_pred = tf.reshape(bbox1,[-1,7,7,1,4])
    _bbox2_pred = tf.reshape(bbox2,[-1,7,7,1,4])

    iou_bbox1=IOU(_label_box,_bbox1_pred)+(1e-9)# n 7 7 1
    iou_bbox2=IOU(_label_box,_bbox2_pred)# n 7 7 1

    iou_bbox = tf.concat([iou_bbox1,iou_bbox2],axis=3 ) # n 7 7 2
    best_box=tf.math.reduce_max(iou_bbox,axis=3,keepdims=True) # n 7 7 1

    box_mask = tf.cast(iou_bbox==best_box,dtype= tf.float32) # n 7 7 2

    #confidence loss
    no_object_loss= 0.5 * (1-tf.concat([confidence_mask,confidence_mask],axis=3) ) * tf.math.square(0 - predict_trust) # box_mask * confidence_mask = exist 1 no 0 / n 7 7 2
    object_loss = box_mask*confidence_mask*tf.math.square(1-predict_trust)# n 7 7 2
    confidence_loss= tf.math.reduce_sum(no_object_loss+object_loss) # shape= () , tensor

    # class loss
    cls_loss= confidence_mask * tf.math.square(label_cls - predict_cls)# n 7 7 20
    cls_loss = tf.math.reduce_sum(cls_loss) # shape= () , tensor 

    #box loss
    selected_box1 =  tf.expand_dims(box_mask[...,0],axis=3) * bbox1 # n 7 7 4  
    selected_box2 =  tf.expand_dims(box_mask[...,1],axis=3) * bbox2 # n 7 7 4
    selected_box = selected_box1+selected_box2# n 7 7 4

    xy_label = label_box[...,0:2]
    xy_pred = selected_box[...,0:2]

    box_loss_xy = 5  * confidence_mask * tf.square(xy_label-xy_pred)# n 7 7 2
    
    wh_label = label_box[...,2:4]
    wh_pred = selected_box[...,2:4]

    zeros= tf.zeros(tf.shape(wh_label))

    box_loss_wh = 5  * confidence_mask * tf.square(tf.math.sqrt(tf.math.maximum(wh_label,zeros))-tf.math.sqrt(tf.math.maximum(wh_pred,zeros)))# n 7 7 2

    box_loss= tf.math.reduce_sum(box_loss_xy + box_loss_wh)

    loss= confidence_loss + cls_loss + box_loss
    loss=loss/len(y_true)

    return loss


def nms(y_pred,iou_threshold=0.5,cls_score_threshold=0.3):
    y_pred= np.reshape(y_pred,(7,7,30))
    for y in range(7):
        for x in range(7):
            real_x1 = (x + y_pred[y][x][1])/7
            real_y1= (y + y_pred[y][x][2])/7 

            real_x2 = (x + y_pred[y][x][6])/7
            real_y2 = (y + y_pred[y][x][7])/7 

            y_pred[y][x][1:3]=[real_x1,real_y1]
            y_pred[y][x][6:8]=[real_x2,real_y2]
    

    cls = np.array(y_pred[...,10:])


    cls_confidence_score1 = cls * y_pred[...,0:1]
    cls_confidence_score2= cls * y_pred[...,5:6]

    score_mask1 = (cls_confidence_score1 >= cls_score_threshold).astype(np.float32) # 0.2 이상만 1 
    score_mask2 = (cls_confidence_score2 >= cls_score_threshold).astype(np.float32) 

    masked_cls_confidence_score1= cls_confidence_score1 * score_mask1 # making zero if cls_confidence_score is too low / 7 7 20
    masked_cls_confidence_score2= cls_confidence_score2 * score_mask2

    bbox_mask1= (np.sum(masked_cls_confidence_score1,axis=2, keepdims = True)>0).astype(np.float32)
    bbox_mask2 = (np.sum(masked_cls_confidence_score2,axis=2, keepdims = True)>0).astype(np.float32)# 7 7 1
    
    bboxes1 = y_pred[...,1:5]  * bbox_mask1
    bboxes2 = y_pred[...,6:10]  * bbox_mask2 # 7 7 4

    bboxes1_con=np.concatenate([masked_cls_confidence_score1,bboxes1],axis= 2)
    bboxes2_con= np.concatenate([masked_cls_confidence_score2,bboxes2],axis=2)# 7 7 24
    
    bboxes1_con=np.reshape(bboxes1_con,(49,24))
    bboxes2_con=np.reshape(bboxes2_con,(49,24))
    bboxes_con = np.concatenate([bboxes1_con,bboxes2_con],axis=0)# 98 24
    

    sorted_list=bboxes_con
    for k in range(20):
        sorted_list= sorted(sorted_list,reverse=True,key = lambda x: x[k])# 98 24
        sorted_list=np.array(sorted_list)
  
        for i in range(len(sorted_list)):
            for j in range(i+1,len(sorted_list)):
                if sorted_list[j][k]==0:
                    continue
                iou=iou_list(sorted_list[i][20:24],sorted_list[j][20:24])
                if iou >= iou_threshold:
                    sorted_list[j][k] =0



   
    sorted_list_mask = (np.amax(sorted_list[:,0:20],axis=1,keepdims=True)>0).astype(dtype=np.float32)
    selected_bboxes= sorted_list_mask * sorted_list
    # for i in selected_bboxes:
    #     print(i)


    boxes=[x[20:24] for x in selected_bboxes]
    cls = [np.argmax(x[:20]) for x in selected_bboxes]
 

    return boxes, cls


    






