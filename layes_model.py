import tensorflow.compat.v1 as tensflw
tensflw.disable_v2_behavior()
def max_2x2_pool(h):
    return tensflw.nn.max_pool(h,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding = "SAME")
def D2cov(A, Z, strd):
    return tensflw.nn.conv2d(A, Z, strides = [1, strd, strd, 1], padding = "SAME")
def var_bias(shp):
    AS = tensflw.constant(0.01,shape=shp)
    return tensflw.Variable(AS)
def wght_var(shp):
    AS = tensflw.truncated_normal(shp,stddev=0.01)
    return tensflw.Variable(AS)