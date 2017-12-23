from LTC import LTC
import tensorflow as tf


dim = 6
clusters = 3

x = tf.random_normal([dim, 1], mean=0, stddev=1)
m = tf.random_normal([dim,clusters], mean=0, stddev=1)

ltc = LTC(dim, clusters)

a = ltc(x,m)

a