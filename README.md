# Image recognition model trained on CIFAR10 
CIFAR is a set of pictures categorized into 10 different classses <br>
link to dataset and documentaion: https://www.cs.toronto.edu/~kriz/cifar.html

## Model description 
* The model is Build using 2 Conv2D layers with 2x2 Max pooling layers
* Final number of feature maps = 128
* DNN layers contain 1024 neurons with a drop_out layer with holding porbability of 0.5

## Accuracy
On the test set, the accuracy is ~75%
<br>
### Accuracy test
![alt text](https://github.com/SAint7579/CIFAR10_BasicCNN/blob/master/accuracy_op.png)
<br>
### Image recognition demo
![alt text](https://github.com/SAint7579/CIFAR10_BasicCNN/blob/master/output.png)

