import json

def print_words(ls, dicts):
    for i in ls:
        print(dicts[i])

q_train = json.load(open('/Users/luchen/Documents/TrueAI/train_q_final.json'))
a_train = json.load(open('/Users/luchen/Documents/TrueAI/train_a_final.json'))
q_valid = json.load(open('/Users/luchen/Documents/TrueAI/valid_q_final.json'))
a_valid = json.load(open('/Users/luchen/Documents/TrueAI/valid_a_final.json'))

input_dict = json.load(open('/Users/luchen/Documents/TrueAI/train_q_word2int.json'))
target_dict = json.load(open('/Users/luchen/Documents/TrueAI/train_a_word2int.json'))

reverse_dict_input = {v: k for k, v in input_dict.items()}
reverse_dict_target = {v: k for k, v in target_dict.items()}

for i in a_train[0]:
    print(reverse_dict_target[i])
