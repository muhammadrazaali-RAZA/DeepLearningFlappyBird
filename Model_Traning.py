from collections import deque

import sys

import cv2 as ipic
#############
import tensorflow.compat.v1 as tensflw
#############
sys.path.append("game/")
import wrap_fb as game
import random


obs = 100000.
expl = 2000000.
finalEpsilon = 0.0001
initialEpsilon = 0.0001

tensflw.disable_v2_behavior()
import numpy as nmpy
def Network_Train(shp, rd, h_FC_1, ss):
    a_var = tensflw.placeholder("float", [None, 2])
    y_var = tensflw.placeholder("float", [None])
    readAction = tensflw.reduce_sum(tensflw.multiply(rd, a_var), reduction_indices=1)
    stepTrain = tensflw.train.AdamOptimizer(1e-6).minimize(tensflw.reduce_mean(tensflw.square(y_var - readAction)))
    game_FB_State = game.FB_Run_game()
    Deq = deque()
    stay_still = nmpy.zeros(2)
    stay_still[0] = 1
    x_axis_term, rev_0, win_term = game_FB_State.steps_of_frame(stay_still)
    x_axis_term = ipic.cvtColor(ipic.resize(x_axis_term, (80, 80)), ipic.COLOR_BGR2GRAY)
    retrn, x_axis_term = ipic.threshold(x_axis_term,1,255,ipic.THRESH_BINARY)
    shp_term = nmpy.stack((x_axis_term, x_axis_term, x_axis_term, x_axis_term), axis=2)
    model_saver = tensflw.train.Saver()
    ss.run(tensflw.initialize_all_variables())
    check_Point = tensflw.train.get_checkpoint_state("model")
    if check_Point and check_Point.model_checkpoint_path:
        model_saver.restore(ss, check_Point.model_checkpoint_path)
        print("Successfully loaded:", check_Point.model_checkpoint_path)
    else:
        print("Could not find old network weights")
################ Traing Model  #############
    epsl = initialEpsilon
    trn = 0
    while "flappy bird" != "angry bird":
        read_term = rd.eval(feed_dict={shp : [shp_term]})[0]
        app_term = nmpy.zeros([2])
        act_Idx = 0
        if trn % 1 == 0:
            if random.random() <= epsl:
                print("----------Random Action----------")
                act_Idx = random.randrange(2)
                app_term[random.randrange(2)] = 1
            else:
                act_Idx = nmpy.argmax(read_term)
                app_term[act_Idx] = 1
        else:
            app_term[0] = 1
################### scale down of epsl
        if epsl > finalEpsilon and trn > obs:
            epsl -= (initialEpsilon - finalEpsilon) / expl
        x_axis_term1_colored, rtrt, win_term = game_FB_State.steps_of_frame(app_term)
        x_axis_term1 = ipic.cvtColor(ipic.resize(x_axis_term1_colored, (80, 80)), ipic.COLOR_BGR2GRAY)
        retrn, x_axis_term1 = ipic.threshold(x_axis_term1, 1, 255, ipic.THRESH_BINARY)
        x_axis_term1 = nmpy.reshape(x_axis_term1, (80, 80, 1))
        shp_term1 = nmpy.append(x_axis_term1, shp_term[:, :, :3], axis=2)
        Deq.append((shp_term, app_term, rtrt, shp_term1, win_term))
        if len(Deq) > 50000:
            Deq.popleft()
        if trn > obs:
            small_bats = random.sample(Deq, 32)
            sj_bats = [d[0] for d in small_bats]
            app_bats = [d[1] for d in small_bats]
            retn_bats = [d[2] for d in small_bats]
            sj1_bats = [d[3] for d in small_bats]
            y_axis_bats = []
            read_j1_bats = rd.eval(feed_dict = {shp : sj1_bats})
            for k in range(0, len(small_bats)):
                win_term = small_bats[k][4]
                if win_term:
                    y_axis_bats.append(retn_bats[k])
                else:
                    y_axis_bats.append(retn_bats[k] + 0.99 * nmpy.max(read_j1_bats[k]))
            stepTrain.run(feed_dict = {
                y_var : y_axis_bats,
                a_var : app_bats,
                shp : sj_bats}
            )
        shp_term = shp_term1
        trn += 1
        if trn % 10000 == 0:
            model_saver.save(ss, 'model/' + 'bird' + '-dqn', global_step = trn)
        curnt_stt = ""
        if trn <= obs:
            curnt_stt = "observe"
        elif trn > obs and trn <= obs + expl:
            curnt_stt = "explore"
        else:
            curnt_stt = "train"
        print("TIMESTEP", trn, "/ STATE", curnt_stt, \
            "/ EPSILON", epsl, "/ ACTION", act_Idx, "/ REWARD", rtrt, \
            "/ Q_MAX %e" % nmpy.max(read_term))