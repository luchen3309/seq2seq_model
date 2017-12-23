import json
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.layers import core as layers_core
import os

def get_seq_length(input):
    ls = []
    for i in input:
        ls.append(len(i))
    return ls

def append_EOS(targets, target_dict):
    t = targets
    for i in t:
        i.append(target_dict['<EOS>'])

    return t

def append_SOS(targets, target_dict):
    t = targets
    for i in t:
        i.insert(0, target_dict['<GO>'])
    return t

def test_batch_data(questions, answers_input, answers_output, q_dict, a_dict, batch_size):
    q_batch=questions[0:batch_size]
    a_inp_batch = answers_input[0:batch_size]
    a_out_batch = answers_output[0:batch_size]

    q_length = np.array(get_seq_length(q_batch)).astype(np.int32)
    a_length = np.array(get_seq_length(a_inp_batch)).astype(np.int32)

    pad_q_batch = np.array(pad_sentence_batch(questions_batch, q_dict)).astype(np.int32)
    pad_a_in_batch = np.array(pad_sentence_batch(a_inp_batch, a_dict)).astype(np.int32)
    pad_a_out_batch = np.array(pad_sentence_batch(a_out_batch, a_dict)).astype(np.int32)

    return pad_q_batch, pad_a_in_batch, pad_a_out_batch, q_length, a_length


def batch_data(questions, answers_input, answers_output, q_dict, a_dict, batch_size):
    # for batch_i in range(0, len(questions) // batch_size):
    #     start_i = batch_i * batch_size
    #     questions_batch = questions[start_i:start_i + batch_size]
    #     answers_batch = answers[start_i:start_i + batch_size]
    #     q_length = np.array(get_seq_length(questions_batch))
    #     a_length = np.array(get_seq_length(answers_batch))
    #     pad_questions_batch = np.array(pad_sentence_batch(questions_batch, q_dict))
    #     pad_answers_batch = np.array(pad_sentence_batch(answers_batch, a_dict))
    #     yield pad_questions_batch, pad_answers_batch, q_length, a_length

    # get batched questions
    for batch_i in range(0, len(questions) // batch_size):
        start_i = batch_i*batch_size

        questions_batch = questions[start_i:start_i+batch_size]
        answers_inp_batch = answers_input[start_i:start_i+batch_size]
        answers_out_batch = answers_output[start_i:start_i+batch_size]

        # get sequence length of batches
        q_length = np.array(get_seq_length(questions_batch)).astype(np.int32)
        a_length = np.array(get_seq_length(answers_inp_batch)).astype(np.int32)

        # get padded batches
        pad_q_batch = np.array(pad_sentence_batch(questions_batch, q_dict)).astype(np.int32)
        pad_a_in_batch = np.array(pad_sentence_batch(answers_inp_batch, a_dict)).astype(np.int32)
        pad_a_out_batch = np.array(pad_sentence_batch(answers_out_batch, a_dict)).astype(np.int32)

        yield pad_q_batch, pad_a_in_batch, pad_a_out_batch, q_length, a_length

def pad_sentence_batch(sentence_batch, word2int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [word2int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def create_embeddings(src_vocab_size, tgt_vocab_size, source_emb_size, target_emb_size):
    enc_embeddings = tf.get_variable('embedding_encoder', [src_vocab_size, source_emb_size], tf.float32)
    dec_embeddings = tf.get_variable('embedding_decoder', [tgt_vocab_size, target_emb_size], tf.float32)

    return enc_embeddings, dec_embeddings

def get_embedded_inputs(enc_inputs, dec_inputs, enc_embeddings, dec_embeddings):
    enc_emb_inp = tf.nn.embedding_lookup(enc_embeddings, enc_inputs)
    dec_emb_inp = tf.nn.embedding_lookup(dec_embeddings, dec_inputs)

    return enc_emb_inp, dec_emb_inp

def build_encoder(enc_emb_inp, seq_length, num_units):
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    enc_outputs, enc_states = tf.nn.dynamic_rnn(encoder_cell, enc_emb_inp, sequence_length=seq_length, dtype=tf.float32)

    return enc_outputs, enc_states

def build_decoder_cell(enc_outputs, seq_length, num_units):

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, enc_outputs, memory_sequence_length = seq_length)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = num_units)

    return decoder_cell

def decode_decoder_cell(enc_state, decoder_cell, type_train, input, seq_length, proj_layer, batch_size,
                        max_iters=None, sos_id=None, eos_id=None):

    if type_train:
        helper = tf.contrib.seq2seq.TrainingHelper(input, seq_length)
    else:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(input, tf.fill([batch_size], sos_id), eos_id)

    enc_state = decoder_cell.zero_state(batch_size, dtype=tf.float32).clone(cell_state=enc_state)

    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, enc_state, output_layer=proj_layer)
    outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_iters)

    logits = outputs.rnn_output

    return logits

# Putting it all together:

# inputs into this model are:
# input sentence
# target sentence
# input dictionary
# output dictionary
# num units
# type
# input embedding size
# target embedding size

def build_full_model(source_inputs, target_inputs,
                     source_vocab_size, target_vocab_size,
                     sos_id, eos_id,
                     enc_emb_size, dec_emb_size,
                     source_seq_length, target_seq_length,
                     num_units,
                     istrain,
                     batch_size):
    """
    TODO: fill this in.
    :param source_inputs:
    :param target_inputs:
    :param source_dict:
    :param target_dict:
    :param enc_emb_size:
    :param dec_emb_size:
    :param source_seq_length:
    :param target_seq_length:
    :param num_units:
    :param type_train:
    :param batch_size:
    :return:
    """

    # 2. Define embeddings
    enc_emb, dec_emb = create_embeddings(source_vocab_size, target_vocab_size, enc_emb_size, dec_emb_size)
    enc_emb_inp, dec_emb_inp = get_embedded_inputs(source_inputs, target_inputs, enc_emb, dec_emb)

    # 3. Build encoder
    enc_outputs, enc_states = build_encoder(enc_emb_inp, source_seq_length, num_units)

    # 4. Build decoder
    decoder_cell = build_decoder_cell(enc_outputs, source_seq_length, num_units)

    # 5. Decode decoder
    proj_layer = layers_core.Dense(target_vocab_size, use_bias=False)
    if istrain:
        logits = decode_decoder_cell(enc_states, decoder_cell, istrain, dec_emb_inp, target_seq_length, proj_layer, batch_size)
    else:
        max_iters = tf.round(tf.reduce_max(source_seq_length)*2)
        logits = decode_decoder_cell(enc_states, decoder_cell, istrain, dec_emb, None, proj_layer, batch_size, max_iters=max_iters,
                                     sos_id=sos_id, eos_id=eos_id)
    return logits

def init_placeholders():
    ## Input specific parameters
    # inputs into graph
    source_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='source_inputs')
    target_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_inputs')
    target_outputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_outputs')

    qlength = tf.placeholder(dtype=tf.int32, shape=[None], name='qlength')
    alength = tf.placeholder(dtype=tf.int32, shape=[None], name='alength')

    lr = tf.placeholder(dtype=tf.float32, name='learn_rate')
    kp = tf.placeholder(dtype=tf.float32, name='keep_prob')

    # # vocab size inputs
    # src_vocab_size = tf.placeholder(dtype=tf.int32, name='src_vocab_size')
    # tgt_vocab_size = tf.placeholder(dtype=tf.int32, name='tgt_vocab_size')
    #
    # # special code ids
    # tgt_sos_id = tf.placeholder(dtype=tf.int32, name='tgt_sos_id')
    # tgt_eos_id = tf.placeholder(dtype=tf.int32, name='tgt_eos_id')
    # tgt_pad_id = tf.placeholder(dtype=tf.int32, name='tgt_pad_id')
    # src_pad_id = tf.placeholder(dtype=tf.int32, name='src_pad_id')
    #
    # ## Model parameters
    # enc_emb_size = tf.placeholder(dtype=tf.int32, name='enc_emb_size')
    # dec_emb_size = tf.placeholder(dtype=tf.int32, name='dec_emb_size')

    # num_units = tf.placeholder(dtype=tf.int32, name='num_units')
    # batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')

    return source_inputs, target_inputs, target_outputs, qlength, alength, lr, kp

questions = json.load(open('/Users/luchen/Documents/TrueAI/train_q_final_2.json'))[0:10]
answers = json.load(open('/Users/luchen/Documents/TrueAI/train_a_final_2.json'))[0:10]
q_dict = json.load(open('/Users/luchen/Documents/TrueAI/train_q_word2int_2.json'))
a_dict = json.load(open('/Users/luchen/Documents/TrueAI/train_a_word2int_2.json'))

answers_input = append_SOS(answers, a_dict)
answers_output = append_EOS(answers, a_dict)

source_vocab_size = len(q_dict)
target_vocab_size = len(a_dict)

sos_id = a_dict['<GO>']
eos_id = a_dict['<EOS>']

# vals = list(a_dict.values())
# vals.sort()
# prev_val = None
# for val in vals:
#     if prev_val is None:
#         prev_val = val
#         continue
#     if prev_val == val:
#         print(val, "is duplicated")
#     if prev_val + 1 != val:
#         print(prev_val, val, "error")
#     prev_val = val

batch_size = 10
enc_emb_size = 512
dec_emb_size = 512
num_units = 512
max_gradient_norm = 1
learning_rate = 0.0001
keep_prob = 0.8

##################################################
# Build graph
tf.reset_default_graph()
sess = tf.Session()

source_inputs, target_inputs, target_outputs, source_seq_length, target_seq_length, \
lr, kp = init_placeholders()


logits = build_full_model(source_inputs, target_inputs,
                     source_vocab_size, target_vocab_size,
                     sos_id, eos_id,
                     enc_emb_size, dec_emb_size,
                     source_seq_length, target_seq_length,
                     num_units,
                     True,
                     batch_size)

with tf.name_scope("optimization"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_outputs, logits=logits)
    # target_weights = np.ones(target_outputs.shape)
    # target_weights[target_outputs == a_dict['<PAD>']] = 0
    #
    # train_loss = (tf.reduce_sum(cross_entropy * target_weights) / batch_size)

    train_loss = (tf.reduce_sum(cross_entropy) / batch_size)
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

#############################################

# Run Tensorflow

total_train_loss=0

path = os.path.dirname(os.path.realpath(__file__))
# checkpoint = path + "best_model.ckpt"
checkpoint = 'tmp/best_model.ckpt'
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for batch_i, (questions_batch, answers_inp_batch, answers_out_batch, q_length, a_length
              ) in enumerate(batch_data(questions, answers_input, answers_output,
                                                                    q_dict, a_dict, batch_size)):

    start_time = time.time()

    _,loss = sess.run([train_op, train_loss],
                      feed_dict={source_inputs: questions_batch,
                                 target_inputs: answers_inp_batch,
                                 target_outputs: answers_out_batch,
                                 source_seq_length: q_length,
                                 target_seq_length: a_length,
                                 lr: learning_rate,
                                 kp: keep_prob})

    total_train_loss += loss
    end_time = time.time()
    batch_time = end_time - start_time
    avg_train_loss = total_train_loss/ (len(questions)/batch_size)
    print(loss)


path = saver.save(sess, checkpoint)
print('Model saved in file :%s' % path)
