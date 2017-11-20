import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data
mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
img_size=28
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)
num_classes=10
images=mnist.test.images[1]

for i in range(20):

    plt.plot(images[1].reshape(20,20))
plt.show()
