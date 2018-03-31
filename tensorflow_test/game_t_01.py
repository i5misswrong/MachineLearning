import tensorflow as tf
import numpy as np

bandits=[0.2,0,-0.2,-5]
num_bandits=len(bandits)

def pullBandits(bandits):
    result=np.random.randn(1)
    print(result)
    if result>bandits:
        return 1
    else:
        return -1


def agent_s():
    tf.reset_default_graph()
    weights=tf.Variable(tf.ones([num_bandits]))
    chosen_action=tf.argmax(weights,0)

    reward_holder=tf.placeholder(shape=[1],dtype=tf.float32)
    action_holder=tf.placeholder(shape=[1],dtype=tf.int32)
    responsible_weight=tf.slice(weights,action_holder)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
    update=optimizer.minimize(loss=0.1)

if __name__=='__main__':
    print(pullBandits(0.2))

