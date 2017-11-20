import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

g=tf.Graph()

with g.as_default():
    x=tf.constant([[2.0,1.0]],dtype=tf.float32)
    y=tf.constant([3.0],dtype=tf.float32)

with g.as_default():
    w=tf.Variable(tf.constant([[1.0],[2.0]],dtype=tf.float32),name="w",trainable=True)
    b=tf.Variable(tf.constant([0.0],dtype=tf.float32),name="b",trainable=True)
    y_=tf.matmul(x,w)+b

with g.as_default():
    loss=tf.reduce_mean((y-y_)**2)

lr=0.1
with g.as_default():
    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(learning_rate=lr,
                                             global_step=global_step,
                                             decay_rate=0.9,
                                             decay_steps=100)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

with g.as_default():
    var_list=tf.trainable_variables()
    grad_var_list=optimizer.compute_gradients(loss=loss,var_list=var_list)
    grad_list=[grad for (grad,var) in grad_var_list]
    grad_SSEs=[tf.reduce_sum(var**2) for var in grad_list]
    train=optimizer.apply_gradients(grad_var_list,global_step=global_step)
with g.as_default():
    var_list=tf.trainable_variables()
    grad_list=tf.gradients(ys=loss,xs=var_list)
    grad_SSEs=[tf.reduce_sum(var**2) for var in grad_list]
    train=optimizer.apply_gradients(zip(grad_list,var_list),global_step=global_step)
with tf.Session(graph=g) as sess:
    sess.run((w.initializer))
    sess.run(b.initializer)
    sess.run(global_step.initializer)
    grad_SSEs_list=[]
    steps=10
    for i in range(steps+1):
        grad_SSEs_list.append(sess.run(grad_SSEs))
        print(grad_SSEs_list[i])
        print(sess.run(y_))
        print("***")
        sess.run(train)


grad_SSEs_list=np.array(grad_SSEs_list)
fig=plt.figure(1)
ax=fig.add_subplot(1,1,1)
ax.plot(grad_SSEs_list[:,0],'r')
ax.plot(grad_SSEs_list[:,1],'b')
ax.legend()
ax.set_title('123')
plt.show()
