import tensorflow as tf
import math


def __Xavir(shape):
    return math.sqrt(1.0/sum(shape))

def Weight(shape,name = None,trainable = True,ifAddToRegul = False,lambdaR = 1e-7):
    varia =  tf.Variable(initial_value=tf.truncated_normal(shape=shape,
                                                         stddev=__Xavir(shape)),
                       dtype=tf.float32,
                       trainable=trainable,
                       name=name)
    if ifAddToRegul is True:
        l2_loss = tf.nn.l2_loss(varia)
        tf.add_to_collection("Loss",tf.multiply(lambdaR,l2_loss))
    return varia

#data format is (b,c,h,w)
def Conv2d(inputTensor,fil,strides,name=None):
    return tf.nn.conv2d(inputTensor,
                        filter=fil,
                        strides=strides,
                        padding="SAME",
                        data_format="NCHW",
                        name=name)

def BatchNormalize(inputTensor,training,name = None,axis = 1):
    return tf.layers.batch_normalization(inputs=inputTensor,
                                         epsilon=1e-5,
                                         axis=axis,
                                         fused=True,
                                         center=True,
                                         scale=False,
                                         training = training,
                                         name=name)

def Pool(inputTensor,windowShape,strides,ptype,name=None,padding = "SAME"):
    return tf.nn.pool(inputTensor,
                      window_shape=windowShape,
                      pooling_type=ptype,
                      padding=padding,
                      strides=strides,
                      data_format="NCHW",
                      name=name)

def Dropout(inputTensor,keepPro,name = None):
    return tf.nn.dropout(inputTensor,
                         keep_prob=keepPro,
                         name=name)

