import tensorflow as tf
import json
import data_cleaning as dc
import numpy as np
import time

def model_inputs():
    """Create placeholders for inputs to the model"""
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob

def process_encoding_input(target_data, vocab_to_int, batch_size):
    """Remove the last word id from each batch and concate <GO> to the
    beginning of each batch"""
    ending = tf.strided_slice(target_data, [0,0], [batch_size, -1], [1,1])

    dec_input = tf.concat([tf.fill([batch_size, 1],
                                   vocab_to_int['<GO>']), ending],1)

    return dec_input


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   sequence_length):
    """ Create the encoding layer """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    drop = tf.contrib.rnn.DropoutWrapper(lstm,
        input_keep_prob=keep_prob)

    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

    _, enc_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=enc_cell,
        cell_bw=enc_cell,
        sequence_length=sequence_length,
        inputs=rnn_inputs,
        dtype=tf.float32)

    return enc_state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         sequence_length, decoding_scope,
                         output_fn, keep_prob, batch_size):
    """ Decode the training data """
    attention_states = tf.zeros([batch_size,
                                 1,
                                 dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = \
        tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option="bahdanau",
            num_units=dec_cell.output_size)

    train_decoder_fn = \
        tf.contrib.seq2seq.attention_decoder_fn_train(
            encoder_state[0],
            att_keys,
            att_vals,
            att_score_fn,
            att_construct_fn,
            name="attn_dec_train")

    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell,
        train_decoder_fn,
        dec_embed_input,
        sequence_length,
        scope=decoding_scope)

    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)

    return output_fn(train_pred_drop)


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings,
                         start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope,
                         output_fn, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,
                                 1,
                                 dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = \
        tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option="bahdanau",
            num_units=dec_cell.output_size)

    infer_decoder_fn = \
        tf.contrib.seq2seq.attention_decoder_fn_inference(
            output_fn,
            encoder_state[0],
            att_keys,
            att_vals,
            att_score_fn,
            att_construct_fn,
            dec_embeddings,
            start_of_sequence_id,
            end_of_sequence_id,
            maximum_length,
            vocab_size,
            name="attn_dec_inf")

    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell,
        infer_decoder_fn,
        scope=decoding_scope)
    return infer_logits


def decoding_layer(dec_embed_input, dec_embeddings, encoder_state,
                   vocab_size, sequence_length, rnn_size,
                   num_layers, vocab_to_int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(
            lstm,
            input_keep_prob=keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(
            x,
            vocab_size,
            None,
            scope=decoding_scope,
            weights_initializer=weights,
            biases_initializer=biases)
        train_logits = decoding_layer_train(encoder_state,
                                            dec_cell,
                                            dec_embed_input,
                                            sequence_length,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)
        decoding_scope.reuse_variables()

        infer_logits = decoding_layer_infer(encoder_state,
                                            dec_cell,
                                            dec_embeddings,
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'],
                                            sequence_length - 1,
                                            vocab_size,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)
    return train_logits, infer_logits


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  sequence_length, answers_vocab_size,
                  questions_vocab_size, enc_embedding_size,
                  dec_embedding_size, rnn_size, num_layers,
                  questions_vocab_to_int):
    enc_embed_input = tf.contrib.layers.embed_sequence(
        input_data,
        answers_vocab_size + 1,
        enc_embedding_size,
        initializer=tf.random_uniform_initializer(-1, 1))

    enc_state = encoding_layer(enc_embed_input,
                               rnn_size,
                               num_layers,
                               keep_prob,
                               sequence_length)
    dec_input = process_encoding_input(target_data,
                                       questions_vocab_to_int,
                                       batch_size)
    dec_embeddings = tf.Variable(
        tf.random_uniform([questions_vocab_size + 1,
                           dec_embedding_size],
                          -1, 1))

    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings,
                                             dec_input)

    train_logits, infer_logits = decoding_layer(
        dec_embed_input,
        dec_embeddings,
        enc_state,
        questions_vocab_size,
        sequence_length,
        rnn_size,
        num_layers,
        questions_vocab_to_int,
        keep_prob,
        batch_size)
    return train_logits, infer_logits

def pad_sentence_batch(sentence_batch, word2int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [word2int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def batch_data(questions, answers, batch_size):
    for batch_i in range(0, len(questions) // batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, q_word2int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, a_word2int))
        yield pad_questions_batch, pad_answers_batch



# -------------------------------------
# Initialize Parameters
# -------------------------------------

# parameters for tensorflow
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75

# parameters for pre-processing
min_length = 2
max_length = 40
freq_threshold = 5

# -------------------------------------
# Clean data
# -------------------------------------
# read json file and see what's there
train_data = json.load(open('/Users/luchen/Downloads/sample_dataset/train/dialogues_task.json'))
validate_data = json.load(open('/Users/luchen/Downloads/sample_dataset/valid/dialogues_task.json'))
test_data = json.load(open('/Users/luchen/Downloads/sample_dataset/test/dialogues_task.json'))

# Try this with a smaller set of training data first, taking too long
train_data = train_data[0:2000]

# <editor-fold desc= "Set up training data">
# split into questions and answers
questions, answers = dc.split_q_a(train_data)

# Clean questions and answers
clean_qs = dc.apply_cleaning(questions)
clean_as = dc.apply_cleaning(answers)

# Remove questions and answers that are too long/short
[final_qs, final_as] = dc.filter_data(clean_qs, clean_as, min_length, max_length)

# Determine which words should be designated as 'Unknown'
vocab = {}
vocab = dc.create_word_freq(final_qs, vocab)
vocab = dc.create_word_freq(final_as, vocab)

q_word2int = dc.word_2_int_dict(vocab, freq_threshold)
a_word2int = dc.word_2_int_dict(vocab, freq_threshold)

# add unique codes
codes = ['<PAD>','<EOS>','<UNK>','<GO>']
for code in codes:
    q_word2int[code] = len(q_word2int)+1
    a_word2int[code] = len(a_word2int)+1

# integers to words
q_int2word = {v_i: v for v, v_i in q_word2int.items()}
a_int2word = {v_i: v for v, v_i in a_word2int.items()}

# add <EOS> to final_as
for i in range(len(final_as)):
    final_as[i] += ' <EOS>'

# convert text to numbers
q_vec = dc.convert_words2int(final_qs, q_word2int)
a_vec = dc.convert_words2int(final_as, a_word2int)

q_train, a_train = dc.sort_by_length(q_vec, a_vec, max_length)

#</editor-fold>

# Set up validation data
q_valid, a_valid = dc.split_q_a(validate_data)
q_valid = dc.apply_cleaning(q_valid)
a_valid = dc.apply_cleaning(a_valid)
[q_valid, a_valid] = dc.filter_data(q_valid, a_valid, min_length, max_length)

for i in range(len(a_valid)):
    a_valid[i] += ' <EOS>'

q_valid = dc.convert_words2int(q_valid, q_word2int)
a_valid = dc.convert_words2int(a_valid, a_word2int)

# -----------------------------------------------
# Set up Tensorflow
# -----------------------------------------------

# Load session
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Set up inputs
input_data, targets, lr, keep_prob = model_inputs()

# Sequence length is max sentence length for each batch
sequence_length = tf.placeholder_with_default(
    max_length,
    None,
    name='sequence_length')
# For sequence loss
input_shape = tf.shape(input_data)

# Create training and inference logits
train_logits, inference_logits = seq2seq_model(
    tf.reverse(input_data, [-1]),
    targets,
    keep_prob,
    batch_size,
    sequence_length,
    len(a_word2int),
    len(q_word2int),
    encoding_embedding_size,
    decoding_embedding_size,
    rnn_size,
    num_layers,
    q_word2int)

with tf.name_scope("optimization"):
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for
                        grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

# Run Tensorflow

display_step = 100
stop_early = 0
stop = 5
validation_check = ((len(q_train))//batch_size//2)-1
total_train_loss = 0
summary_valid_loss = []

checkpoint = "best_model.ckpt"

sess.run(tf.global_variables_initializer())

for epoch_i in range(1, epochs + 1):
    for batch_i, (questions_batch, answers_batch) in enumerate(
            batch_data(q_train, a_train, batch_size)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: learning_rate,
             sequence_length: answers_batch.shape[1],
             keep_prob: keep_probability})

        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % display_step == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs,
                          batch_i,
                          len(q_train) // batch_size,
                          total_train_loss / display_step,
                          batch_time * display_step))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in \
                    enumerate(batch_data(q_valid, a_valid, batch_size)):
                valid_loss = sess.run(
                    cost, {input_data: questions_batch,
                           targets: answers_batch,
                           lr: learning_rate,
                           sequence_length: answers_batch.shape[1],
                           keep_prob: 1})
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(q_valid) / batch_size)
            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!')
                stop_early = 0
                saver = tf.train.Saver()
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.")
                stop_early += 1
                if stop_early == stop:
                    break

    if stop_early == stop:
        print("Stopping Training.")
        break
