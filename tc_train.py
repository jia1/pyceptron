import os
import re
import sys
from string import punctuation
from porter import PorterStemmer
from random import seed, randrange

k = 3
max_compromise = 0
num_both, num_train = 0, 0
train_ratio = 1
# train_ratio = 0.8
# test_ratio = 1 - train_ratio
feat_prune_ratio = 0.5

num_folds_list = [5] # [5, 10]
alpha_list = [0.05] # [0.02, 0.03, 0.05, 0.07, 0.1]
max_iterations_list = [500] # [500, 1000, 2000]

args = sys.argv
if len(args) != 4:
    sys.exit('Usage: python3 tc_train.py stopword-list train-class-list model')

stopword_list_file, train_class_list_file, model_file = args[1:]
p = PorterStemmer()

class_list = []
class_to_text, text_to_freq = {}, {}

nxxs = ['n10', 'n11']
nxxs_map = { 'n00': 'n10', 'n01': 'n11' }
nxx_dict = { n: {} for n in nxxs }
chi_dict = {}

class_to_feat_chi_tup = {}
class_to_feat_set = {}
class_to_feat_list_sort_by_lex = {}
class_to_feat_mat = {}

seed(4248)

def strip_and_filter_line(ln):
    if all(x in ln for x in [':', '@']):
        return []
    tokens = map(lambda t: t.strip().strip(punctuation).lower(), ln.split(' '))
    return list(filter(lambda t: t and len(t) > 2 and t.isalpha() and t not in stop_list, tokens))

def is_in(a, b):
    return 1 if a in b else 0

def count_nxx(nxx, w, c):
    global class_list, class_to_text, text_to_freq
    answer = 0
    if nxx == 'n10':
        for class_name in filter(lambda x: x != c, class_list):
            for text in class_to_text[class_name]:
                answer += is_in(w, text_to_freq[text])
    elif nxx == 'n11':
        for text in class_to_text[c]:
            answer += is_in(w, text_to_freq[text])
    return answer

def chi_square(w, c):
    global num_train, nxxs, nxxs_map, nxx_dict
    ns_dict = {}
    for n in nxxs:
        if w not in nxx_dict[n]:
            nxx_dict[n][w] = {}
        if c not in nxx_dict[n][w]:
            nxx_dict[n][w][c] = count_nxx(n, w, c)
        ns_dict[n] = nxx_dict[n][w][c]
    for n, nn in nxxs_map.items():
        ns_dict[n] = num_train - nxx_dict[nn][w][c]
    n00, n01, n10, n11 = ns_dict['n00'], ns_dict['n01'], ns_dict['n10'], ns_dict['n11']
    return ((n11+n10+n01+n00)*(n11*n00-n10*n01)**2)/((n11+n01)*(n11+n10)*(n10+n00)*(n01+n00))

def put_chi_dict(c, w, chi_square_value):
    global chi_dict
    if w not in chi_dict[c]:
        chi_dict[c][w] = chi_square_value
    else:
        chi_dict[c][w] = max(chi_dict[c][w], chi_square_value)

def gen_feat():
    global class_list, chi_dict, class_to_feat_chi_tup
    max_feat_len = sys.maxsize
    feat_queue_dict = { c: [] for c in class_list }
    for c in chi_dict:
        feat_queue_dict[c] = sorted(chi_dict[c].items(), key = lambda x: x[1], reverse = True)
        max_feat_len = min(max_feat_len, len(feat_queue_dict[c]))
    max_feat_len *= feat_prune_ratio 
    class_to_feat_chi_tup = { c: feat_queue_dict[c][:int(max_feat_len)] for c in feat_queue_dict }

def feat_select():
    global class_list, class_to_text, text_to_freq, class_to_feat_chi_tup
    for c in class_list:
        for text in class_to_text[c]:
            for w in text_to_freq[text]:
                put_chi_dict(c, w, chi_square(w, c))
                gen_feat()

with open(stopword_list_file, 'r') as s:
    stop_list = list(map(lambda ln: ln.strip(), s.readlines()))

with open(train_class_list_file, 'r') as t:
    lines = map(lambda ln: ln.strip().split(' '), t.readlines())
    for ln in lines:
        file, curr_class = ln
        text = file.split('/')[-1]
        num_both += 1
        num_train += 1
        flat_text, freq_dict = [], {}
        with open(file, 'r') as f:
            processed_lines = map(lambda ln: strip_and_filter_line(ln), f.readlines())
            for line in processed_lines:
                flat_text.extend(list(map(lambda word: p.stem(word, 0, len(word) - 1), line)))
            num_words = len(flat_text)
            last_index = num_words - 1
            for i in range(num_words):
                word = flat_text[i]
                if word not in freq_dict:
                    freq_dict[word] = 1
                else:
                    freq_dict[word] += 1
                if i < last_index:
                    bigram = '{} {}'.format(word, flat_text[i + 1])
                    if bigram not in freq_dict:
                        freq_dict[bigram] = 1
                    else:
                        freq_dict[bigram] += 1
            # TODO: Abstract to a function
            fin_freq_dict = { word: freq for word, freq in freq_dict.items() if freq >= k }
            compromise = 1
            while not fin_freq_dict and compromise <= max_compromise:
                fin_freq_dict = { word: freq for word, freq in freq_dict.items() if freq >= k - compromise }
                compromise += 1
            len_gram = 2
            while not fin_freq_dict and len_gram:
                fin_freq_dict = { word: freq for word, freq in freq_dict.items() if len(word.split(' ')) >= len_gram }
                len_gram -= 1
            # END
            sum_freq = sum(fin_freq_dict.values())
            if curr_class not in class_list:
                class_list.append(curr_class)
            if curr_class not in class_to_text:
                class_to_text[curr_class] = set()
            else:
                class_to_text[curr_class].add(text)
            if text not in text_to_freq:
                text_to_freq[text] = {}
            else:
                text_to_freq[text] = { word: freq / sum_freq for word, freq in fin_freq_dict.items() }

chi_dict = { c: {} for c in class_list }
class_to_feat_chi_tup = { c: set() for c in class_list }
feat_select()

class_to_feat_set = { c: set() for c in class_list }
for c in class_to_feat_chi_tup:
    for p in class_to_feat_chi_tup[c]:
        w = p[0]
        class_to_feat_set[c].add(w)
    curr_num_feat = len(class_to_feat_set[c])
    num_feat_per_neg_class = curr_num_feat // (num_class - 1)
    for nc in class_to_feat_chi_tup:
        if nc != c:
            num_added = 0
            for t in class_to_feat_chi_tup:
                class_to_feat_set[c].add(t[0])
                if num_added >= num_feat_per_neg_class:
                    break

class_to_feat_list_sort_by_lex = { c: sorted(list(class_to_feat_set[c])) for c in class_list }
class_to_feat_to_index = { c: {} for c in class_list }

for c in class_to_feat_list_sort_by_lex:
    for i in range(len(class_to_feat_list_sort_by_lex[c])):
        class_to_feat_to_index[c][class_to_feat_list_sort_by_lex[c][i]] = i

def get_folds(data_mat, num_folds):
    folds = []
    data_clone = list(data_mat)
    fold_size = int(len(data_mat) / num_folds)
    for i in range(num_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(data_clone))
            fold.append(data_clone.pop(index))
        folds.append(fold)
    return folds

def get_accuracy(predicted, actual):
    num_correct = 0
    for i in range(len(actual)):
        if predicted[i] == actual[i]:
            num_correct += 1
    return num_correct / len(actual) * 100

def get_cross_validation_scores(data_mat, algorithm, num_folds, *args):
    folds = get_folds(data_mat, num_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = [row for fold in train_set for row in fold]
        test_set = []
        for row in fold:
            row_clone = list(row)
            test_set.append(row_clone)
            row_clone[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = get_accuracy(predicted, actual)
        scores.append(accuracy)
    return scores

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1 if activation >= 0 else 0

def train_weights(train, alpha, max_iterations = 1000):
    weights = [0 for i in range(len(train[0]))]
    for _ in range(max_iterations):
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + alpha * error
            for i in range(len(row) - 1):
                weights[i + 1] += alpha * error * row[i]
    return weights

def perceptron(train, test, alpha, max_iterations):
    predictions = list()
    weights = train_weights(train, alpha, max_iterations)
    print(weights) # TODO: Record the weights somewhere
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions

class_to_feat_mat = { c: [] for c in class_list }
for c in class_list:
    for d in class_list:
        texts = class_to_text[d]
        num_texts = len(texts)
        texts = iter(texts)
        if c != d:
            num_texts_to_train = int((1 - train_ratio) * num_texts)
        else:
            num_texts_to_train = num_texts
        for i in range(num_texts_to_train):
            text = next(texts)
            feat_vec = [0 for i in range(len(class_to_feat_to_index[d]) + 1)]
            for word in text_to_freq[text]:
                if word in class_to_feat_to_index[d]:
                    index = class_to_feat_to_index[d][word]
                    feat_vec[index] = text_to_freq[text][word]
            feat_vec[-1] = 1 if c == d else 0
            class_to_feat_mat[c].append(feat_vec)

data = class_to_feat_mat
for num_folds in num_folds_list:
    for alpha in alpha_list:
        for max_iterations in max_iterations_list:
            print('{}-fold cross validation'.format(num_folds))
            print('Learning rate: {}, maximum number of iterations: {}'.format(alpha, max_iterations))
            for c in class_list:
                scores = get_cross_validation_scores(data[c], perceptron, num_folds, alpha, max_iterations)
                print('Class: {}'.format(c))
                print('Cross validation scores: {}'.format(scores))
                print('Mean accuracy: {:.2f}%'.format(sum(scores) / num_folds))
                print()

# Write model to file
# 1. Class list
# 2. Class to feature to index on feature vector
# 3. Class to weights
