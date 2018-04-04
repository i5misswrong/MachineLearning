import  matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/path/to/MNIST_data/",one_hot=True)


def TRAIN_SIZE(num):
    x_train = mnist.train.images[:num, :]
    y_train = mnist.train.labels[:num, :]
    return x_train, y_train

def display_digit(num):
    x_train, y_train = TRAIN_SIZE(55000)
    print(y_train[num])

    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28, 28])

    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))

    plt.show()

if __name__ == '__main__':
    display_digit(100)