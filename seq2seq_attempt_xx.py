import json
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.layers import core as layers_core
import copy
from LTC import LTC

debug_dict = {}

#########################
# HELPER FUNCTIONS
#########################

def debug_dictionary(d):

    vals = list(d.values())
    vals.sort()
    prev_val = None
    for val in vals:
        if prev_val is None:
            prev_val = val
            continue
        if prev_val == val:
            print(val, "is duplicated")
        if prev_val + 1 != val:
            print(prev_val, val, "error")
        prev_val = val

# works
def get_seq_length(input):
    ls = []
    for i in input:
        ls.append(len(i))
    return ls

# works
def append_EOS(targets, target_dict):
    t = copy.deepcopy(targets)
    for i in t:
        i.append(target_dict['<EOS>'])

    return t

# works
def append_SOS(targets, target_dict):
    t = copy.deepcopy(targets)
    for i in t:
        i.insert(0, target_dict['<GO>'])
    return t

def test_batch_data(questions, answers_input, answers_output, q_dict, a_dict, batch_size):
    q_batch=questions[0:batch_size]
    a_inp_batch = answers_input[0:batch_size]
    a_out_batch = answers_output[0:batch_size]

    q_length = np.array(get_seq_length(q_batch)).astype(np.int32)
    a_length = np.array(get_seq_length(a_inp_batch)).astype(np.int32)

    pad_q_batch = np.array(pad_sentence_batch(q_batch, q_dict)).astype(np.int32)
    pad_a_in_batch = np.array(pad_sentence_batch(a_inp_batch, a_dict)).astype(np.int32)
    pad_a_out_batch = np.array(pad_sentence_batch(a_out_batch, a_dict)).astype(np.int32)

    return pad_q_batch, pad_a_in_batch, pad_a_out_batch, q_length, a_length

# works
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

# works
def pad_sentence_batch(sentence_batch, word2int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [word2int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

##########################
# Model building functions
##########################

# works
def create_embedding(vocab_size, emb_size, name):
    embeddings = tf.get_variable(name, [vocab_size, emb_size], tf.float32)
    return embeddings

# works
def get_embedded_input(inputs, embeddings):
    emb_inp = tf.nn.embedding_lookup(embeddings, inputs)
    return emb_inp

def build_encoder(enc_emb_inp, seq_length, num_units, keep_prob):
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    encoder_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, input_keep_prob=keep_prob)
    enc_outputs, enc_states = tf.nn.dynamic_rnn(encoder_cell, enc_emb_inp, sequence_length=seq_length, dtype=tf.float32)

    return enc_outputs, enc_states

def build_decoder_cell(enc_outputs, seq_length, num_units, keep_prob):

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, input_keep_prob=keep_prob)
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
    preds = outputs.sample_id

    return logits, preds

def initialize_representation(num_dim, num_topics, reg_scale):
    param_m = tf.get_variable('param_m', [num_dim, num_topics], tf.float32, regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale))
    return param_m

def useLTC(input, representation, num_dim, num_topics, seq_length):

    ltc = LTC(num_dim, num_topics, tf.reduce_max(seq_length))
    inp_k = ltc(input, representation)
    inp = tf.concat([input, inp_k], 2)

    return inp

def build_full_model(source_inputs,
                     source_seq_length,
                     source_vocab_size,
                     target_vocab_size,
                     batch_size,
                     enc_emb_size,
                     dec_emb_size,
                     num_units,
                     istrain,
                     sos_id, eos_id,
                     keep_prob,
                     target_inputs=None,
                     target_seq_length=None,
                     num_topics=None):

    reg_scale = 0.5

    # 1. Initialize embeddings
    enc_emb = create_embedding(source_vocab_size, enc_emb_size, 'embedding_encoder')
    dec_emb = create_embedding(target_vocab_size, dec_emb_size, 'embedding_decoder')
    param_m = initialize_representation(dec_emb_size, num_topics, reg_scale)

    # 2. Apply embeddings
    enc_emb_inp = get_embedded_input(source_inputs, enc_emb)

    # Apply LTC model - now decoded embedded inputs are batch_size * max_target_seq_length * 2 * dec_emb_size
    if istrain:
        dec_emb_inp = get_embedded_input(target_inputs, dec_emb)
        dec_emb_inp = useLTC(dec_emb_inp, param_m, dec_emb_size, num_topics, tf.reduce_max(target_seq_length))
    # else:
    #     dec_emb_inp = get_embedded_input(source_inputs, enc_emb)
    #     dec_emb_inp = tf.concat([dec_emb_inp, dec_emb_inp],2)

    # 3. Build encoder
    enc_outputs, enc_states = build_encoder(enc_emb_inp, source_seq_length, num_units, keep_prob)

    # Encoded embedded inputs are batch_size * max_source_seq_length * 2 enc_emb_size
    if ~istrain:
        enc_outputs = useLTC(enc_outputs, param_m, enc_emb_size, num_topics, tf.reduce_max(source_seq_length))
    # else:
    #     enc_outputs = tf.concat([enc_outputs, enc_outputs], 2)

    # 4. Build decoder
    decoder_cell = build_decoder_cell(enc_outputs, source_seq_length, num_units, keep_prob)

    # 5. Decode decoder
    proj_layer = layers_core.Dense(target_vocab_size, use_bias=False)
    if istrain:
        logits, preds = decode_decoder_cell(enc_states, decoder_cell, istrain, dec_emb_inp, target_seq_length, proj_layer, batch_size)
    else:
        max_iters = tf.round(tf.reduce_max(source_seq_length)*2)
        # Tried to see if this solves the shape problem - THIS IS THE PROBLEM
        logits, preds = decode_decoder_cell(enc_states, decoder_cell, istrain, tf.concat([dec_emb, dec_emb],1), None, proj_layer, batch_size, max_iters=max_iters,
                                     sos_id=sos_id, eos_id=eos_id)

    # 6. Get regularization loss from representation
    total_reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    return logits, total_reg_loss, preds

# works
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

questions = json.load(open('/Users/luchen/Documents/TrueAI/train_q_final.json'))[0:100]
answers = json.load(open('/Users/luchen/Documents/TrueAI/train_a_final.json'))[0:100]
q_dict = json.load(open('/Users/luchen/Documents/TrueAI/q_dict.json'))
a_dict = json.load(open('/Users/luchen/Documents/TrueAI/a_dict.json'))

answers_input = append_SOS(answers, a_dict)
answers_output = append_EOS(answers, a_dict)

source_vocab_size = len(q_dict)
target_vocab_size = len(a_dict)

sos_id = a_dict['<GO>']
eos_id = a_dict['<EOS>']


batch_size = 10
enc_emb_size = 512
dec_emb_size = 512
num_units = 512
max_gradient_norm = 1
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_prob = 1
num_topics = 50
epochs = 50

##################################################
# Build graph

train_graph = tf.Graph()
test_graph = tf.Graph()

with train_graph.as_default():
    source_inputs, target_inputs, target_outputs, source_seq_length, target_seq_length, \
    lr, kp = init_placeholders()

    logits, reg_loss, _ = build_full_model(source_inputs,
                         source_seq_length,
                         source_vocab_size,
                         target_vocab_size,
                         batch_size,
                         enc_emb_size,
                         dec_emb_size,
                         num_units,
                         True,
                         sos_id, eos_id,
                         keep_prob,
                         target_inputs,
                         target_seq_length,
                         num_topics)

    with tf.name_scope("optimization"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_outputs, logits=logits)
        pad_mat = tf.ones_like(target_outputs)*a_dict['<PAD>']
        inv_mask = tf.equal(target_outputs, pad_mat)
        mask = tf.cast(tf.logical_not(inv_mask), dtype=tf.float32)

        train_loss = (tf.reduce_sum(cross_entropy*mask) / batch_size) + reg_loss
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    saver = tf.train.Saver()
    initializer = tf.global_variables_initializer()

train_sess = tf.Session(graph=train_graph)

#############################################
# Run Tensorflow
#############################################
total_train_loss=0

checkpoint = 'tmp/best_model.ckpt'
train_sess.run(initializer)

# [pad_q_batch, pad_a_in_batch, pad_a_out_batch, q_length, a_length] = test_batch_data(questions, answers_input,
#                                                                                      answers_output, q_dict,
#                                                                                      a_dict, batch_size)
#
# source_inputs_result, enc_outputs, enc_states = sess.run([source_inputs, debug_dict['enc_outputs'], debug_dict['enc_states']],
#                       feed_dict={source_inputs: pad_q_batch,
#                                  target_inputs: pad_a_in_batch,
#                                  target_outputs: pad_a_out_batch,
#                                  source_seq_length: q_length,
#                                  target_seq_length: a_length,
#                                  kp: 1.0})

for epoch_i in range(1, epochs + 1):
    for batch_i, (questions_batch, answers_inp_batch, answers_out_batch, q_length, a_length
                  ) in enumerate(batch_data(questions, answers_input, answers_output,
                                                q_dict, a_dict, batch_size)):

        start_time = time.time()

        _, loss = train_sess.run([train_op, train_loss],
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

    learning_rate *= learning_rate_decay
    if learning_rate < min_learning_rate:
        learning_rate = min_learning_rate

path = saver.save(train_sess, checkpoint)
print('Model saved in file :%s' % path)



# Validation!!!!!

# Running an inference graph
q_valid = json.load(open('/Users/luchen/Documents/TrueAI/valid_q_final.json'))
a_valid = json.load(open('/Users/luchen/Documents/TrueAI/valid_a_final.json'))

a_valid_input = append_SOS(a_valid, a_dict)
a_valid_output = append_EOS(a_valid, a_dict)

[pad_q_batch, pad_a_in_batch, pad_a_out_batch, q_length, a_length] = test_batch_data(q_valid, a_valid_input, a_valid_output, q_dict, a_dict, batch_size)

with test_graph.as_default():
    test_source_inputs, test_target_inputs, test_target_outputs, test_source_seq_length, test_target_seq_length, \
    test_lr, test_kp = init_placeholders()

    _, _, translations = build_full_model(test_source_inputs,
                                    test_source_seq_length,
                                    source_vocab_size,
                                    target_vocab_size,
                                    batch_size,
                                    enc_emb_size,
                                    dec_emb_size,
                                    num_units,
                                    False,
                                    sos_id, eos_id,
                                    keep_prob,
                                    target_inputs=None,
                                    target_seq_length=None,
                                    num_topics=num_topics)


    # with tf.name_scope("optimization"):
    #     t_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=test_target_outputs, logits=test_logits)
    #     test_loss = (tf.reduce_sum(t_cross_entropy) / batch_size)
    #     t_params = tf.trainable_variables()
    #     t_gradients = tf.gradients(test_loss, t_params)
    #     t_clipped_gradients, _ = tf.clip_by_global_norm(t_gradients, max_gradient_norm)
    #
    #     t_optimizer = tf.train.AdamOptimizer(test_lr)
    #     test_op = t_optimizer.apply_gradients(zip(t_clipped_gradients, t_params))

    test_saver = tf.train.Saver()
    test_initializer = tf.global_variables_initializer()

test_session = tf.Session(graph=test_graph)

test_session.run(test_initializer)

# try and see what shapes these are and why
test_saver.restore(test_session, checkpoint)

print(pad_a_out_batch.shape)


res = test_session.run(translations,
                 feed_dict={test_source_inputs: pad_q_batch,
                            test_target_inputs: pad_a_in_batch,
                            test_target_outputs: pad_a_out_batch,
                            test_source_seq_length: q_length,
                            test_target_seq_length: a_length,
                            test_lr: learning_rate,
                            test_kp: keep_prob})


a_rev_dict = {v_i: v for v, v_i in a_dict.items()}

print(res)

for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        print(a_rev_dict[res[i][j]])
        if j == res.shape[1]:
            print('end')

for i in range(pad_a_out_batch.shape[0]):
    for j in range(pad_a_out_batch.shape[1]):
        print(a_rev_dict[pad_a_out_batch[i][j]])
        if j == pad_a_out_batch.shape[1]:
            print('end')
