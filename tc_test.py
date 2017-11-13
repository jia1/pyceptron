import ast
import re
import sys

from string import punctuation
from porter import PorterStemmer

args = sys.argv
if len(args) != 5:
    sys.exit('Usage: python3 tc_test.py stopword-list model test-list test-class-list')

k = 3
max_compromise = 1
text_to_class = {}
lines_to_write = []
score, total = 0, 0

stopword_list_file, model_file, test_list_file, test_class_list_file = args[1:]
p = PorterStemmer()

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

with open(stopword_list_file, 'r') as s:
    stop_list = list(map(lambda ln: ln.strip(), s.readlines()))

with open(model_file, 'r') as m:
    lines = list(map(lambda w: ast.literal_eval(w), m.readlines()))
    class_list, class_to_feat_to_index, class_to_weights = lines

with open(test_class_list_file, 'r') as a:
    lines = map(lambda ln: ln.strip().split(' '), a.readlines())
    for ln in lines:
        file, curr_class = ln
        # text = file.split('/')[-1]
        text = re.split('[(\\\\)(\\)(\/)]', file)[-1]
        text_to_class[text] = curr_class

with open(test_list_file, 'r') as t:
    # lines = map(lambda ln: ln.strip(), t.readlines())
    lines = map(lambda ln: ln.strip().split(' ')[0], t.readlines())
    for ln in lines:
        file = ln
        # text = file.split('/')[-1]
        text = re.split('[(\\\\)(\\)(\/)]', file)[-1]
        flat_text = []
        with open(file, 'r') as f:
            for line in map(lambda ln: strip_and_filter_line(ln), f.readlines()):
                flat_text.extend(list(map(lambda word: p.stem(word, 0, len(word) - 1), line)))
            word_to_count = get_word_to_count(flat_text)
            fin_word_to_count = { word: count for word, count in word_to_count.items() if count >= k }
            if not fin_word_to_count:
                fin_word_to_count = get_weaker_word_to_count(word_to_count)
            sum_count = sum(fin_word_to_count.values())
            normalized_word_to_count = { word: count / sum_count for word, count in fin_word_to_count.items() }
            instance_class_to_output = { c: 0 for c in class_list }
            for c in class_list:
                for w in class_to_feat_to_index[c]:
                    if w in normalized_word_to_count:
                        index = class_to_feat_to_index[c][w]
                        instance_class_to_output[c] += class_to_weights[c][index] * normalized_word_to_count[w]
            instance_class_to_output = sorted(instance_class_to_output.items(), key = lambda x: x[1], reverse = True)
            lines_to_write.append('{} {}\n'.format(file, instance_class_to_output))
            total += 1
            if text_to_class[text] == instance_class_to_output[0][0]:
                score += 1

print(score)
print(total)

with open('answer', 'w') as f:
    f.writelines(lines_to_write)
