#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def weight_variable(shape):
    #truncated_normal 从截断的正态分布输出随机值 stddev-截断前正态分布的标准偏差
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    #constant 定义常数张量 0.01-value-常量值 shape-张量尺寸
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    '''
    计算给定4-D input和filter张量的2-D卷积。
    给定形状的输入张量和形状[batch, in_height, in_width, in_channels] 的滤波/内核张量 [filter_height, filter_width, in_channels, out_channels]，该操作执行以下操作：
    将滤镜平整为具有形状的2-D矩阵 [filter_height * filter_width * in_channels, output_channels]。
    从输入张量中提取图像块以形成虚拟 的形状张量[batch, out_height, out_width, filter_height * filter_width * in_channels]。
    对于每个修补程序，右乘过滤器矩阵和图像修补程序向量。
    :param x:输入 4维张量
    :param W:过滤器 4为张量
    :param stride:幅度 1维张量
    :return:#nn.conv2d-计算2d卷积
    padding：string来自："SAME", "VALID"。要使用的填充算法的类型。
    '''
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    '''
    value：Tensor由指定的格式的4-D data_format。
    ksize：4元素的1-D int张量。输入张量的每个维度的窗口大小。
    strides：4元素的1-D int张量。输入张量的每个维度的滑动窗口的步幅。
    padding：一个字符串，'VALID'或者'SAME'。填充算法。在这里看到评论
    data_format：一个字符串。'NHWC'，'NCHW'和'NCHW_VECT_C'被支持。
    name：操作的可选名称。
    :param x:Tensor由指定的格式的4-D data_format
    :return:#在输入上执行最大池化
    '''
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    #返回输入  输出 ？？
    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    # reduce_sum 计算一个张量维数的总和
    # multiply 以元素方式返回x * y。
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    # reduce_mean 计算张量维度上元素的平均值
    # square 平方
    cost = tf.reduce_mean(tf.square(y - readout_action)) # 成本函数=(y-输出值)^2
    #AdamOptimizer 使用Adam算法
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    # 打开游戏
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque() #list

    # printing  打开文件  输出
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    # 获取第一次状态 -无动作和对图片预处理
    #无动作 矩阵 action=2 生成一个1*2矩阵
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1# do_nothing=[1,0]
    #x_t288*512*3 r_0=0.1 terminal=false
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    # x_t 为当前游戏画面 的像素数据
    # 此处为对游戏画面简化  去除背景 增加对比度……
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    # saver-保存并恢复变量
    saver = tf.train.Saver()
    #初始化变量
    sess.run(tf.initialize_all_variables())
    # get_checkpoint_state-从“检查点”文件返回CheckpointState原型。
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    # 检查 '检查点' 载入旧的训练数据
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON #0.00001
    t = 0
    # 开始训练
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily # 选择一个贪心 动作 参数
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS]) #创建动作表格[0,0]
        action_index = 0# 动作索引=0
        if t % FRAME_PER_ACTION == 0:#如果时间对帧数取余=0 即在帧数末尾 发出动作的那一帧
            if random.random() <= epsilon:#如果随机数小于0.00001
                print("----------Random Action----------")#
                action_index = random.randrange(ACTIONS)#随机产生动作索引[0,1]
                a_t[random.randrange(ACTIONS)] = 1#-----将动作表格随机生成1---即随机产生一个动作
            else:
                action_index = np.argmax(readout_t)# 获取输出的最大值 赋值给动作索引
                a_t[action_index] = 1#???
        else:
            a_t[0] = 1 # do nothing 其他帧不做动作

        # scale down epsilon  epsilon下降尺度
        if epsilon > FINAL_EPSILON and t > OBSERVE:#如果epsilon大于0.0001 并且 t>200000
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE #epsilon 作何变化?

        # run the selected action and observe next state and reward
        #运行 动作列表 +观察下一步状态和期望
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)#运行动作列表
        #图形处理
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D  将……添加到D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY: #如果D的长度大于 20000  总长度
            D.popleft()#移除左边的

        # only train if done observing
        if t > OBSERVE: #如果t大于10000  上一步的记忆数量?
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
