import numpy as np
import pandas as pd
import json
from scipy.stats import ks_2samp
from random import randint

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.transpose(x)
    e_x = np.exp(x - np.max(x))
    s = e_x / e_x.sum(axis=0)
    return np.transpose(s)

def convert_to_embedding(input_x, embedding):
    """Return embedded representation of the input"""
    return embedding[input_x,:]

def topic_distribution(embedded_input, param_m):
    """Return logits from softmax"""
    p = softmax(embedded_input.dot(param_m))
    return p

def dispersion_index(dist):
    """Calculates the uniformity of the distribution of clusters.
    The closer the number is to the number of clusters, the more uniform it is"""
    plnp = dist * np.log(dist)
    entropy = -1.0 * sum(plnp)
    return np.exp(entropy)

def sketchy_conversion_2D(dist):
    dist = np.sum(dist, axis=0)
    dist = softmax(dist)

    return dist

def calc_dispersion(input_x, embedding, param_m):
    """Combines helper functions to calculate a raw sentence vector's topioc representation"""

    embedded_input = convert_to_embedding(input_x, embedding)
    dist = topic_distribution(embedded_input, param_m)

    # sketchy thing to convert to a single dimension
    dist = sketchy_conversion_2D(dist)
    disp_idx = dispersion_index(dist)

    return disp_idx

def order_by_disp_ind(target_data, embedding, param_m):
    """Loops through all inputs to create a list of dispersion metrics"""
    disp_idx = []

    for x in target_data:
        disp_idx.append(calc_dispersion(x, embedding, param_m))

    return disp_idx

def convert_to_words(text, mapping):
    """Converts integer representations to actual words"""
    t_vec = []
    for t in text:
        w_vec = []
        for w in t:
            w_vec.append(mapping[w])
        t_vec.append(w_vec)

    return t_vec

def get_dist_from_target(target_data, embedding, param_m):
    """Calculates topic distribution from a list of raw inputs"""
    ls_dist = []
    for x in target_data:
        emb_x = convert_to_embedding(x, embedding)
        dist = topic_distribution(emb_x, param_m)
        dist = sketchy_conversion_2D(dist)

        ls_dist.append(dist)

    return ls_dist

def compare_distns(ls_dist, num_samples):
    """Randomly selects two distributions to compare similarity. It will test num_samples pairs of distributions"""
    pval_ls=[]

    for i in range(num_samples):
        idx = randint(0, len(ls_dist))
        idx2 = randint(0, len(ls_dist))

        while idx == idx2:
            idx2 = randint(0, len(ls_dist))

        _,pval = ks_2samp(ls_dist[idx], ls_dist[idx2])
        pval_ls.append(pval)

    return pval_ls


# parameters needed to test things
num_dim = 512
num_topics = 10
vocab_size = 18200
uncertainty_threshold = 9

data_path = r'/Users/luchen/Documents/TrueAI/document_data/'

param_m = np.random.random_sample((num_dim, num_topics))
embedding = np.random.random_sample((vocab_size, num_dim))
target_data = json.load(open(data_path + 'valid_a_final.json'))
a_dict = json.load(open(data_path + 'a_dict.json'))
rev_dict = {v_i: v for v, v_i in a_dict.items()}

# Test uncertainty
disp_idx = order_by_disp_ind(target_data, embedding, param_m)

target_data = np.array(target_data)
disp_idx = np.array(disp_idx)

uncertain_phrases = target_data[disp_idx > 9]
certain_phrases = target_data[disp_idx < 2]

# read through certain phrases and see if they're more similar than uncertain phrases

# Test one-to-one topic
num_samples=6

connectivity = [0,1,2,5,7,10,11,12,14,15,16,18,20,25,33,35,36,42,57,59,82]
fourg = [6,43,138]
sim = [4,76,102,127,144]
customer_service = [8,9,32,109,126,143,146,151,152,168]
account_queries = [38, 39,105]
abroad = [37,58,61,86,88]
threelovesxmas = [21,23,40,45,46,68,79,106,113,130,163]
data_limits = [27,47,48,49,50,51,52,53,54,55,56]
mean_complaints = [75,93]

connectivity_targets = target_data[connectivity]
xmas_targets = target_data[threelovesxmas]
customer_targets = target_data[customer_service]

connectivity_dist = get_dist_from_target(connectivity_targets, embedding, param_m)
xmas_dist = get_dist_from_target(xmas_targets, embedding, param_m)
customer_dist = get_dist_from_target(customer_targets, embedding, param_m)

connectivity_pvals = compare_distns(connectivity_dist, num_samples)
xmas_pvals = compare_distns(xmas_dist, num_samples)
customer_pvals = compare_distns(customer_dist, num_samples)


