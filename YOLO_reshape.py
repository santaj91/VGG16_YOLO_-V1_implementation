import tensorflow as tf


class YOLO_reshape(tf.keras.layers.Layer):
    def __init__(self,target_shape=(7,7,30),**kwargs):
        super(YOLO_reshape,self).__init__()
        self.target_shape = target_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'target_shape': self.target_shape
            }
        )
        return config
    def call(self,input):# input= 1470
        S= self.target_shape[0]
        B = 2
        C= 20

        idx1= S*S*C
        idx2= idx1 + S*S*B

        input = tf.reshape(input,(-1,S*S*(B*5+C))) 
        
        # class probabilities
        class_probs= tf.reshape(
            input[:,:idx1],
            (tf.shape(input)[0],)+(S,S,C))# n 7 7 20
        class_probs= tf.nn.softmax(class_probs)
        # confidence
        confidence = tf.reshape(
            input[:,idx1:idx2],
            (tf.shape(input)[0],)+(S,S,B))
        confidence=tf.nn.sigmoid(confidence)
        c1,c2 = tf.split(confidence, num_or_size_splits=2,axis=3)# n 7 7 1 x 2
        # boxes
        boxes = tf.reshape(
            input[:,idx2:],
            (tf.shape(input)[0],)+(S,S,B*4)
        )
        boxes= tf.nn.sigmoid(boxes)
        b1,b2= tf.split(boxes,num_or_size_splits=2,axis=3)# n 7 7 4 x 2


        outputs= tf.concat([c1,b1,c2,b2,class_probs],axis=3)

        return outputs