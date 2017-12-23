import sonnet as snt
import tensorflow as tf

class LTC(snt.AbstractModule):
    """ To implement latent topic clustering"""

    def __init__(self, dim_topic, num_topic_clusters, name='LTC'):

        super(LTC, self).__init__(name=name)
        self._dim_topic = dim_topic
        self._num_topic_clusters = num_topic_clusters


    def _build(self, input_x, param_m):

        p = tf.nn.softmax(tf.matmul(tf.transpose(input_x), param_m))
        y = tf.matmul(param_m, tf.transpose(p))

        # xk = tf.transpose(tf.reshape(tf.tile(p, [1,self._dim_topic]), [self._dim_topic, self._num_topic_clusters]))
        # y = tf.matmul(xk, param_m)

        return y



