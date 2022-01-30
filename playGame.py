from Network_Model import Model_build
from Model_Traning import Network_Train

import tensorflow.compat.v1 as tensflw
tensflw.disable_v2_behavior()


def play_fb_Game():
    shp, rd, h_FC_1 = Model_build()
    Network_Train(shp, rd, h_FC_1 , tensflw.InteractiveSession())