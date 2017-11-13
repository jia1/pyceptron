import math
import os
import re
import sys
from string import punctuation
from porter import PorterStemmer
from random import seed, randrange

args = sys.argv
if len(args) != 4:
    sys.exit('Usage: python3 tc_train.py stopword-list train-class-list model')

print('Command line arguments accepted.')

stopword_list_file, train_class_list_file, model_file = args[1:]

k = 3
max_compromise = 1
num_both, num_train = 0, 0
train_ratio = 1
# train_ratio = 0.8
# test_ratio = 1 - train_ratio

num_folds_list = [10] # [5, 10]
alpha_list = [0.05] # [0.02, 0.03, 0.05, 0.07, 0.1]
max_iterations_list = [1000] # [500, 1000, 2000]

num_class = 0
class_list = []
class_to_text = {}

feat_prune_ratio = 0.5

text_to_count = {}
nxx_list = ['n10', 'n11']
nxx_map = {
    'n00': 'n10',
    'n01': 'n11'
}
nxx_to_word_to_class_to_chi = { n: {} for n in nxx_list }
class_to_word_to_chi = {}
class_to_feat_chi_tup = {}

text_to_word_list = {}
class_to_vocab_to_tfidf = {}
class_to_feat_tfidf_tup = {}

class_to_feat_set = {}
class_to_feat_list_sort_by_lex = {}
class_to_feat_to_index = {}
class_to_feat_mat = {}

class_to_weights = {}

p = PorterStemmer()
seed(4248)

print('Data structures loaded.')

def strip_and_filter_line(ln):
    if all(x in ln for x in [':', '@']):
        return []
    tokens = map(lambda t: t.strip().strip(punctuation).lower(), ln.split(' '))
    return list(filter(lambda t: t and len(t) > 2 and t.isalpha() and t not in stop_list, tokens))

def get_word_to_count(word_list):
    word_to_count = {}
    num_words = len(word_list)
    prev_unigram = word_list[0]
    for i in range(1, num_words):
        curr_unigram = word_list[i]
        ngrams = [curr_unigram, '{} {}'.format(prev_unigram, curr_unigram)]
        for ngram in ngrams:
            if ngram not in word_to_count:
                word_to_count[ngram] = 1
            else:
                word_to_count[ngram] += 1
        prev_unigram = curr_unigram
    return word_to_count

def get_weaker_word_to_count(word_to_count):
    fin_word_to_count = {}
    for compromise in range(1, max_compromise - 1):
        if fin_word_to_count:
            break
        fin_word_to_count = { word: count for word, count in word_to_count.items() \
                             if count >= k - compromise }
        for len_gram in range(2, 0, -1):
            fin_word_to_count = { word: count for word, count in fin_word_to_count.items() \
                                 if len(word.split(' ')) >= len_gram }
            if fin_word_to_count:
                break
    return fin_word_to_count

def is_in(a, b):
    return 1 if a in b else 0

def count_nxx(nxx, w, c):
    nxx_value = 0
    if nxx == 'n10':
        for class_name in filter(lambda x: x != c, class_list):
            for text in class_to_text[class_name]:
                nxx_value += is_in(w, text_to_count[text])
    elif nxx == 'n11':
        for text in class_to_text[c]:
            nxx_value += is_in(w, text_to_count[text])
    return nxx_value

def calc_chi_square(w, c):
    nxx_to_count = {}
    for n in nxx_list:
        if w not in nxx_to_word_to_class_to_chi[n]:
            nxx_to_word_to_class_to_chi[n][w] = {}
        if c not in nxx_to_word_to_class_to_chi[n][w]:
            nxx_to_word_to_class_to_chi[n][w][c] = count_nxx(n, w, c)
        nxx_to_count[n] = nxx_to_word_to_class_to_chi[n][w][c]
    for n, nn in nxx_map.items():
        nxx_to_count[n] = num_train - nxx_to_word_to_class_to_chi[nn][w][c]
    n00, n01, n10, n11 = nxx_to_count['n00'], nxx_to_count['n01'], nxx_to_count['n10'], nxx_to_count['n11']
    return ((n11+n10+n01+n00)*(n11*n00-n10*n01)**2)/((n11+n01)*(n11+n10)*(n10+n00)*(n01+n00))

def put_chi(c, w, chi_value):
    global class_to_word_to_chi
    if w not in class_to_word_to_chi[c]:
        class_to_word_to_chi[c][w] = chi_value
    else:
        class_to_word_to_chi[c][w] = max(class_to_word_to_chi[c][w], chi_value)

def gen_feat_by_chi():
    global class_to_feat_chi_tup
    max_feat_vec_len = sys.maxsize
    class_to_feat_sorted = { c: [] for c in class_list }
    for c in class_to_word_to_chi:
        class_to_feat_sorted[c] = sorted(class_to_word_to_chi[c].items(), key = lambda x: x[1], reverse = True)
        max_feat_vec_len = min(max_feat_vec_len, len(class_to_feat_sorted[c]))
    max_feat_vec_len *= feat_prune_ratio 
    class_to_feat_chi_tup = { c: class_to_feat_sorted[c][:int(max_feat_vec_len)] for c in class_to_feat_sorted }

def gen_feat_by_tfidf():
    global class_to_vocab_to_tfidf
    
    for c in class_list:
        for text in class_to_text[c]:
            word_list = text_to_word_list[text]
            prev_unigram = word_list[0]
            class_to_vocab_to_tfidf[c][prev_unigram] = 0
            for i in range(1, len(word_list)):
                curr_unigram = word_list[i]
                bigram = '{} {}'.format(prev_unigram, curr_unigram)
                class_to_vocab_to_tfidf[c][curr_unigram] = 0
                class_to_vocab_to_tfidf[c][bigram] = 0
                prev_unigram = curr_unigram
    for c in class_list:
        for text in class_to_text[c]:
            word_list = text_to_word_list[text]
            prev_unigram = word_list[0]
            class_to_vocab_to_tfidf[c][prev_unigram] = 0
            for i in range(1, len(word_list)):
                curr_unigram = word_list[i]
                bigram = '{} {}'.format(prev_unigram, curr_unigram)
                class_to_vocab_to_tfidf[c][curr_unigram] += 1
                class_to_vocab_to_tfidf[c][bigram] += 1
                prev_unigram = curr_unigram

    for c in class_list:
        num_texts = len(class_to_text[c])
        for v in class_to_vocab_to_tfidf[c]:
            class_to_vocab_to_tfidf[c][v] = math.log(num_texts / (1 + class_to_vocab_to_tfidf[c][v]))
                
    max_feat_vec_len = sys.maxsize
    class_to_feat_sorted = { c: [] for c in class_list }
    for c in class_to_word_to_chi:
        class_to_feat_sorted[c] = sorted(class_to_vocab_to_tfidf[c].items(), key = lambda x: x[1], reverse = True)
        max_feat_vec_len = min(max_feat_vec_len, len(class_to_feat_sorted[c]))
    max_feat_vec_len *= feat_prune_ratio 
    class_to_vocab_to_tfidf = { c: class_to_feat_sorted[c][:int(max_feat_vec_len)] for c in class_to_feat_sorted }

def feat_select():
    # gen_feat_by_tfidf()
    for c in class_list:
        for text in class_to_text[c]:
            for w in text_to_count[text]:
                put_chi(c, w, calc_chi_square(w, c))
    gen_feat_by_chi()

print('Helper functions defined.')

with open(stopword_list_file, 'r') as s:
    stop_list = list(map(lambda ln: ln.strip(), s.readlines()))

print('Stop words loaded into memory.')

with open(train_class_list_file, 'r') as t:
    lines = map(lambda ln: ln.strip().split(' '), t.readlines())
    for ln in lines:
        file, curr_class = ln
        text = file.split('/')[-1]
        num_both += 1
        num_train += 1
        flat_text = []
        with open(file, 'r') as f:
            for line in map(lambda ln: strip_and_filter_line(ln), f.readlines()):
                flat_text.extend(list(map(lambda word: p.stem(word, 0, len(word) - 1), line)))
            word_to_count = get_word_to_count(flat_text)
            fin_word_to_count = { word: count for word, count in word_to_count.items() if count >= k }
            if not fin_word_to_count:
                fin_word_to_count = get_weaker_word_to_count(word_to_count)
            sum_count = sum(fin_word_to_count.values())

            if curr_class not in class_list:
                class_list.append(curr_class)
                num_class += 1
                class_to_text[curr_class] = set()
                class_to_word_to_chi[curr_class] = {}
                class_to_feat_chi_tup[curr_class] = set()
                class_to_vocab_to_tfidf[curr_class] = {}
                # class_to_word_to_num_text[curr_class] = {}
                class_to_feat_tfidf_tup[curr_class] = set()
                class_to_feat_set[curr_class] = set()
                class_to_feat_list_sort_by_lex[curr_class] = []
                class_to_feat_to_index[curr_class] = {}
                class_to_weights[curr_class] = []

            class_to_text[curr_class].add(text)
            text_to_word_list[text] = flat_text
            text_to_count[text] = { word: count / sum_count for word, count in fin_word_to_count.items() }

print('Frequency of unigrams and bigrams counted.')

class_to_word_to_chi = { c: {} for c in class_list }
class_to_feat_chi_tup = { c: set() for c in class_list }
class_to_word_to_num_text = { c: {} for c in class_list }
class_to_feat_tfidf_tup = { c: set() for c in class_list }

print('Feature selection beginning...')

feat_select()

print('Feature selection completed.')

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

print('Features from negative classes added to each positive class.')

class_to_feat_list_sort_by_lex = { c: sorted(list(class_to_feat_set[c])) for c in class_list }
class_to_feat_to_index = { c: {} for c in class_list }

for c in class_to_feat_list_sort_by_lex:
    for i in range(len(class_to_feat_list_sort_by_lex[c])):
        class_to_feat_to_index[c][class_to_feat_list_sort_by_lex[c][i]] = i

print('Features mapped to vector indices.')

# https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
# Split data_mat into num_folds number of folds
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

# Calculate accuracy percentage
def get_accuracy(predicted, actual):
    num_correct = 0
    for i in range(len(actual)):
        if predicted[i] == actual[i]:
            num_correct += 1
    return num_correct / len(actual) * 100

# Evaluate an algorithm using a cross validation split
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

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1 if activation >= 0 else 0

# Estimate Perceptron weights using stochastic gradient descent
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

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, alpha, max_iterations):
    predictions = list()
    weights = train_weights(train, alpha, max_iterations)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions

print('Perceptron helper functions defined.')

# load and prepare data
class_to_feat_mat = { c: [] for c in class_list }
for c in class_list:
    for d in class_list:
        texts = class_to_text[d]
        num_texts = len(texts)
        texts = iter(texts)
        if c != d:
            num_texts_to_train = int(num_texts * train_ratio / (num_class - 1))
        else:
            num_texts_to_train = num_texts
        for i in range(num_texts_to_train):
            text = next(texts)
            feat_vec = [0 for i in range(len(class_to_feat_to_index[d]) + 1)]
            for word in text_to_count[text]:
                if word in class_to_feat_to_index[d]:
                    index = class_to_feat_to_index[d][word]
                    feat_vec[index] = text_to_count[text][word]
            feat_vec[-1] = 1 if c == d else 0
            class_to_feat_mat[c].append(feat_vec)

print('Training data converted to vectors.')

data = class_to_feat_mat

print('Cross validation beginning...')

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

print('Cross validation completed.')
print('Full training beginning...')

for c in class_list:
    class_to_weights[c] = train_weights(data[c], alpha_list[0], max_iterations_list[0])

print('Weights being written to model file...')

with open(model_file, 'w') as m:
    lines_to_write = []
    lines_to_write.append(str(class_list))
    lines_to_write.append(str(class_to_feat_to_index))
    lines_to_write.append(str(class_to_weights))
    m.write('\n'.join(lines_to_write))

print('Done.')

# Write model to file
# 1. Class list
# 2. Class to feature to index on feature vector
# 3. Class to weights
