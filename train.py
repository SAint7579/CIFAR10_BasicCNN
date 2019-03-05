#Importing files with helper class and functions
from dataset import *
from helper_func import *
import pdb
#Setting up the dataset
ch = CifarHelper()
ch.set_up_images()

#Importing tensorflow
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf

#Declaring constants
LR = 0.0001     #Learing rate for training
DNN_LEN = 1024  
FET_MAP = 64    
IMG_DIM = 32    # For both lenght and width
IMG_DEP = 3     # RGB layers
POOL_LAYERS = 2
#Creating the placeholders
x = tf.placeholder(dtype = tf.float32, shape=[None,IMG_DIM,IMG_DIM,3], name = "Input_data")
y_true = tf.placeholder(dtype=tf.float32,shape=[None,10], name = "Correct_op")
#For dropout layer
hold_proba = tf.placeholder(dtype= tf.float32)

#Creating the CNN network (2 convolution + 2 max_pool layers)
convo_1 = conv_layer(x,shape=[4,4,3,FET_MAP],name = "conv1")
pool_1 = max_pool_2by2(convo_1)
convo_2 = conv_layer(pool_1,shape=[4,4,FET_MAP,FET_MAP*2],name = "conv2")
pool_2 = max_pool_2by2(convo_2)
#convo_3 = conv_layer(pool_2,shape=[4,4,FET_MAP*2,FET_MAP*4],name = "conv3")
#pool_3 = max_pool_2by2(convo_3)
#Flattening the output layer
flattened = tf.reshape(pool_2,shape=(-1,(IMG_DIM//(POOL_LAYERS*2))*(IMG_DIM//(POOL_LAYERS*2))*(FET_MAP*2)),name = "flattened")
#Creating the DNN layers for predictions
dnn_1 = dnn_layer(flattened,size=DNN_LEN, name = 'dense1')
drop_out_1 = tf.nn.dropout(dnn_1,keep_prob=hold_proba)  #Dropout for regularization
#Creating the output layer
y_pred = dnn_layer(drop_out_1 ,size=10,name = 'output')
#Creating the loss function
cross_entropy =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
#Creating the optimizer and trainer
optim = tf.train.AdamOptimizer(learning_rate=LR)
train = optim.minimize(cross_entropy)

#Variable initializer and saver
init = tf.global_variables_initializer()
sav = tf.train.Saver()

#Training the model
with tf.Session() as sess:
    sess.run(init)
    steps = 20000
    for i in range(steps):
        batch = ch.next_batch(100)
        sess.run(train,feed_dict={x:batch[0],y_true:batch[1],hold_proba:0.5})
        if i%100 == 0:
            try:
                #generating training batches
                train_batch = ch.next_batch(100)
                correct = tf.equal(tf.argmax(y_pred,axis=1),tf.argmax(y_true,axis=1))
                #making accuracy tensor
                acc  = tf.reduce_mean(tf.cast(correct,dtype=tf.float32))
                #displaying accuracy
                print("ON STEP: ",i,"\t Accuracy= ",sess.run(acc,feed_dict={x:train_batch[0],y_true:train_batch[1],hold_proba:1}))
            except:
                print("Couldn't predict the accuracy")
                continue
    sav.save(sess,"CIFAR10_CNN/model.ckpt")

