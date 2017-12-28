from LTC import LTC
import tensorflow as tf

batch_size = 5
dim = 4
clusters = 3
numwords = 2

x = tf.random_normal([batch_size, numwords, dim], mean=0, stddev=1)
m = tf.random_normal([dim,clusters], mean=0, stddev=1)

ltc = LTC(dim, clusters)

a = ltc(x)
