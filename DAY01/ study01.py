import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data
mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
img_size=28
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)
num_classes=10
images=mnist.test.images[0:9]
mnist.test.cls = np.array([label.argmax() for label in mnist.test.labels])
cls_true=mnist.test.cls[0:9]
# print(cls_true) # the list of the number [7,2,1,0,1,4,9,5]

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    # fig, axes = plt.subplot(3, 3)
    fig, axes = plt.subplots(3,3)
    # fig=plt.sub
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
            # pass
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
            pass
        ax.set_xlabel(xlabel) # show the true number under the x

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

plot_images(images=images,cls_true=cls_true)