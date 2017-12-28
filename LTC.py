import sonnet as snt
import tensorflow as tf

class LTC(snt.AbstractModule):
    """ To implement latent topic clustering"""

    def __init__(self, dim_topic, num_topic_clusters, seq_length, name='LTC'):
        super(LTC, self).__init__(name=name)
        self._dim_topic = dim_topic
        self._num_topic_clusters = num_topic_clusters
        self._seq_length = seq_length
        # self.param_m = tf.get_variable('param_m', [dim_topic, num_topic_clusters], tf.float32)

    def _build(self, input_x, param_m):

        # Deal with a 3D tensor
        input_x = tf.reshape(input_x, [-1, self._dim_topic])

        p = tf.nn.softmax(tf.matmul(input_x, param_m))
        y = tf.matmul(param_m, tf.transpose(p))

        # Deal with 3D tensor still
        y = tf.reshape(y, [-1, self._seq_length, self._dim_topic])

        # xk = tf.transpose(tf.reshape(tf.tile(p, [1,self._dim_topic]), [self._dim_topic, self._num_topic_clusters]))
        # y = tf.matmul(xk, param_m)

        return y



