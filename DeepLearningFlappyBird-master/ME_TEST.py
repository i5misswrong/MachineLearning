# import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

ACTIONS = 2
do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
fly=np.zeros(ACTIONS)
fly[1]=1
print(do_nothing)
game_state = game.GameState()
i=0
while "flappy bird" != "angry bird":

    # if i%10 ==0:
    #     x_t, r_0, terminal = game_state.frame_step(fly)
    #     print('noting')
    # else:
    #     x_t, r_0, terminal = game_state.frame_step(do_nothing)
    #     print('fly')
    # i=i+1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    plt.imshow(s_t)
    plt.show()
    # x_t, r_0, terminal = game_state.frame_step(do_nothing)
    time.sleep(100)