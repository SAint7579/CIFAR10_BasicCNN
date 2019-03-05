import os
#Setting environment variable for GPU (Set to -1 for CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
#Importing CIFAR dataset
from dataset import *
import numpy as np
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
    acc_list = []
    for i in range(0,1000,100):
        img_batch = ch.test_images[i:i+100]
        lab_batch = ch.test_labels[i:i+100]
        correct = tf.equal(tf.argmax(y_pred,axis=1),tf.argmax(y_true,axis=1))
        #making accuracy tensor
        acc  = tf.reduce_mean(tf.cast(correct,dtype=tf.float32))
        #displaying accuracy
        print("For test images ",i," to ", i+100)
        accuracy = sess.run(acc,feed_dict={x:img_batch,y_true:lab_batch,hold_proba:1})
        print("ON STEP: ",i,"\t Accuracy= ",accuracy)
        acc_list.append(accuracy)
    print("\nMAXIMUM ACCURACY:", max(acc_list))
    print("MINIMUM ACCURACY:", min(acc_list))
        
        
