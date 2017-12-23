import tensorflow as tf
import json
import time
import re
import numpy as np

def process_encoding_input(target_data, word2int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])

    dec_input = tf.concat([tf.fill([batch_size, 1],
                                   word2int['<GO>']), ending], 1)
    return dec_input

def seq2seq_model(input_data, target_data, keep_prob, batch_size, seq_length,
                  a_vocab_size, q_vocab_size, encoding_embedding_size, decoding_embedding_size,
                  num_units, num_layers, q_word2int):

    # inputs into the model
    # source input words (encoder inputs): [max_encoder_time, batch_size]
    # target input words (decoder inputs): [max_encoder_time, batch_size]

    # Embedding layer
    # Need to map words into a smaller dimension of fixed length
    # This mapping will be learned in the model
    # embedding_encoder: [source_vocab_size, embedding_size]
    # encoder_embedded_input = embedding_lookup(embedding_encoder, encoder_inputs)

    encoded_embed_input = tf.contrib.layers.embed_sequence(
        input_data,
        a_vocab_size + 1,
        encoding_embedding_size,
        initializer=tf.random_uniform_initializer(-1, 1))

    # Embed targets
    decoded_embeddings = tf.Variable(tf.random_uniform([q_vocab_size + 1, decoding_embedding_size], -1, 1))

    # target output words (Decoder outputs): [max_decoder_time, batch_size]
    #       (decoder inputs shifted left by one time step, <EOS> appended on the right
    decoded_output = process_encoding_input(target_data, q_word2int, batch_size)

    decoded_embed_input = tf.nn.embedding_lookup(decoded_embeddings, decoded_output)













    return train_logit











# Load data
q_train = json.load(open('/Users/luchen/Documents/TrueAI/train_q_final.json'))
a_train = json.load(open('/Users/luchen/Documents/TrueAI/train_a_final.json'))
q_word2int = json.load(open('/Users/luchen/Documents/TrueAI/train_q_word2int.json'))
a_word2int = json.load(open('/Users/luchen/Documents/TrueAI/train_a_word2int.json'))



# Encoder
# Word embeddings are fed as input to the encoder
# each cell is an LSTM
# encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
# outputs, state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp)

# Decoder
# decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
# helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True)
# projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)
# decoder = tf.contrib.seq2seq.BasicDecoder(
#  decoder_cell, helper, encoder_state,
#    output_layer=projection_layer)

# Dynamic decoding
# outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
# logits = outputs.rnn_output

# Loss function
# target_weights is a masking matrix that indicates where the padding is
# crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
#    labels=decoder_outputs, logits=logits)
# train_loss = (tf.reduce_sum(crossent * target_weights) /
#    batch_size)

# Gradient computation and optimization
# params = tf.trainable_variables()
# gradients = tf.gradients(train_loss, params)
# clipped_gradients, _ = tf.clip_by_global_norm(
#    gradients, max_gradient_norm)

# Optimization
# optimizer = tf.train.AdamOptimizer(learning_rate)
# update_step = optimizer.apply_gradients(
#    zip(clipped_gradients, params))