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

    test_logits, _ = build_full_model(test_source_inputs,
                                    test_source_seq_length,
                                    source_vocab_size,
                                    target_vocab_size,
                                    batch_size,
                                    enc_emb_size,
                                    dec_emb_size,
                                    num_units,
                                    False,
                                    sos_id, eos_id,
                                    1,
                                    target_inputs=None,
                                    target_seq_length=None,
                                    num_topics=num_topics)


    with tf.name_scope("optimization"):
        t_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=test_target_outputs, logits=test_logits)
        test_loss = (tf.reduce_sum(t_cross_entropy) / batch_size)
        t_params = tf.trainable_variables()
        t_gradients = tf.gradients(test_loss, t_params)
        t_clipped_gradients, _ = tf.clip_by_global_norm(t_gradients, max_gradient_norm)

        t_optimizer = tf.train.AdamOptimizer(test_lr)
        test_op = t_optimizer.apply_gradients(zip(t_clipped_gradients, t_params))

    test_saver = tf.train.Saver()
    test_initializer = tf.global_variables_initializer()

test_session = tf.Session(graph=test_graph)

test_session.run(test_initializer)
test_saver.restore(test_session, checkpoint)

_,t_loss = test_session.run([test_op, test_loss],
                 feed_dict={test_source_inputs: pad_q_batch,
                            test_target_inputs: pad_a_in_batch,
                            test_target_outputs: pad_a_out_batch,
                            test_source_seq_length: q_length,
                            test_target_seq_length: a_length,
                            test_lr: learning_rate,
                            test_kp: keep_prob})

print(t_loss)


# Debugging purposes: add everything to debugging dict
debug_dict['enc_emb'] = enc_emb
debug_dict['dec_emb'] = dec_emb
debug_dict['param_m'] = param_m
debug_dict['enc_emb_inp'] = enc_emb_inp
debug_dict['dec_emb_inp'] = dec_emb_inp
debug_dict['enc_outputs'] = enc_outputs
debug_dict['enc_states'] = enc_states
debug_dict['decoder_cell'] = decoder_cell
debug_dict['logits'] = logits
debug_dict['preds'] = preds


# Debugging checks
[pad_q_batch, pad_a_in_batch, pad_a_out_batch, q_length, a_length] = test_batch_data(questions, answers_input,
                                                                                     answers_output, q_dict,
                                                                                     a_dict, batch_size)
source_inputs,\
target_inputs,\
target_outputs,\
enc_emb,\
dec_emb,\
param_m,\
enc_emb_inp,\
dec_emb_inp,\
enc_outputs,\
enc_states,\
logits,\
preds = train_sess.run([source_inputs, target_inputs, target_outputs, debug_dict['enc_emb'],
debug_dict['dec_emb'],
debug_dict['param_m'],
debug_dict['enc_emb_inp'],
debug_dict['dec_emb_inp'],
debug_dict['enc_outputs'],
debug_dict['enc_states'],
debug_dict['logits'],
debug_dict['preds'] ],
                      feed_dict={source_inputs: pad_q_batch,
                                 target_inputs: pad_a_in_batch,
                                 target_outputs: pad_a_out_batch,
                                 source_seq_length: q_length,
                                 target_seq_length: a_length,
                                 kp: 1.0})
print(source_inputs.shape)
print(target_inputs.shape)
print(target_outputs.shape)
print(enc_emb.shape)
print(dec_emb.shape)
print(param_m.shape)
print(enc_emb_inp.shape)
print(dec_emb_inp.shape)
print(enc_outputs.shape)
# print(enc_states.shape)
print(logits.shape)
print(preds.shape)


def clean_counts(questions, answers, min_length, max_length):
    q=[]
    a=[]

    for i in range(len(questions)):
        if questions[i] >=min_length and questions[i] <= max_length \
            and answers[i] >= min_length and answers[i] <= max_length:

            q.append(questions[i])
            a.append(answers[i])

    return q,a






















