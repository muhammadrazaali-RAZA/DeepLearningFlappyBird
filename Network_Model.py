from layes_model import wght_var
from layes_model import var_bias
from layes_model import D2cov
from layes_model import max_2x2_pool

import tensorflow.compat.v1 as tensflw

def Model_build():
    
    shp = tensflw.placeholder("float", [None, 80, 80, 4])

    h_con_V1 = tensflw.nn.relu(D2cov(shp, wght_var([8, 8, 4, 32]), 4) + var_bias([32]))

    h_con_V2 = tensflw.nn.relu(D2cov(max_2x2_pool(h_con_V1), wght_var([4, 4, 32, 64]), 2) + var_bias([64]))

    h_con_V3 = tensflw.nn.relu(D2cov(h_con_V2, wght_var([3, 3, 64, 64]), 1) + var_bias([64]))

    h_FC_1 = tensflw.nn.relu(tensflw.matmul(tensflw.reshape(h_con_V3, [-1, 1600]), wght_var([1600, 512])) + var_bias([512]))

    return shp, tensflw.matmul(h_FC_1, wght_var([512, 2])) + var_bias([2]), h_FC_1

