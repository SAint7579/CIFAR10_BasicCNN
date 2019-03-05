import os
#Setting environment variable for GPU (Set to -1 for CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
#Importing CIFAR dataset
from dataset import *
import numpy as np
import matplotlib.pyplot as plt
ch = CifarHelper()
ch.set_up_images()

with tf.Session() as sess:  
    #Loading model and the graphs
    saver = tf.train.import_meta_graph('CIFAR10_CNN/model.ckpt.meta')
    saver.restore(sess,'CIFAR10_CNN/model.ckpt')
    #Creating a tf graph object
    graph = tf.get_default_graph()
    #Restoring placeholders
    x = graph.get_tensor_by_name("Input_data:0")
    y_true = graph.get_tensor_by_name("Correct_op:0")
    hold_proba = graph.get_tensor_by_name("Placeholder:0")
    #Restoring operation y_pred
    y_pred = graph.get_tensor_by_name("output:0")
    softmax = tf.nn.softmax(y_pred,name= 'softmax')
    #Creating graph
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    #Label dictionary
    labels ={
              0:'airplane',
              1:'car',
              2:'bird',
              3:'cat',
              4:'deer',
              5:'dog',
              6:'frog',
              7:'horse',
              8:'ship',
              9:'truck'
            }

    batch = ch.next_batch(100)
    for i in range(1, columns*rows +1):
        #Adding subplot
        fig.add_subplot(rows, columns, i)
        #Adding the figure to graph
        plt.imshow(batch[0][i].reshape(32,32,3))
        pred = sess.run( tf.argmax(softmax[0]) , feed_dict = {x: batch[0][i].reshape(1, 32, 32, 3), y_true: batch[1][i].reshape(1,10), hold_proba:1.0} )
        #Displaying predicted and actual values
        print("Predicted Value: "+ labels[pred] + "\t\tActual Value: "+labels[list(batch[1][i]).index(1.0)])
    #Showing the graph
    plt.show()

        
        
