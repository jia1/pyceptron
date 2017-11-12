import ast
import re
import sys

from string import punctuation
from porter import PorterStemmer

args = sys.argv
if len(args) != 5:
    sys.exit('Usage: python3 tc_test.py stopword-list model test-list test-class-list')

stopword_list_file, model_file, test_list_file, test_class_list_file = args[1:]
p = PorterStemmer()

# stopword_list_file = 'stopword-list'
# test_list_file, test_class_list_file = 'test-list', 'test-class-list'
# test_list_file, test_class_list_file = 'train-class-list-copy', 'train-class-list-copy'
# model_file = 'model'

def strip_and_filter_line(ln):
    if all(x in ln for x in [':', '@']):
        return []
    tokens = map(lambda t: t.strip().strip(punctuation).lower(), ln.split(' '))
    return list(filter(lambda t: t and len(t) > 2 and t.isalpha() and t not in stop_list, tokens))

with open(model_file, 'r') as m:
    lines = list(map(lambda w: ast.literal_eval(w), m.readlines()))
    class_list, class_to_feat_to_index, class_to_weights = lines

text_to_class = {}
with open(test_class_list_file, 'r') as a:
    lines = map(lambda ln: ln.strip().split(' '), a.readlines())
    for ln in lines:
        file, curr_class = ln
        # text = file.split('/')[-1]
        text = re.split('[(\\\\)(\\)(\/)]', file)[-1]
        text_to_class[text] = curr_class

k = 3
answer_lines = []
score, total = 0, 0
with open(test_list_file, 'r') as t:
    # lines = map(lambda ln: ln.strip(), t.readlines())
    lines = map(lambda ln: ln.strip().split(' ')[0], t.readlines())
    for ln in lines:
        file = ln
        # text = file.split('/')[-1]
        text = re.split('[(\\\\)(\\)(\/)]', file)[-1]
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
            normalized_freq_dict = { word: freq / sum_freq for word, freq in fin_freq_dict.items() }
            answer = { c: 0 for c in class_list }
            for c in class_list:
                for w in class_to_feat_to_index[c]:
                    if w in normalized_freq_dict:
                        index = class_to_feat_to_index[c][w]
                        answer[c] += class_to_weights[c][index] * normalized_freq_dict[w]
            answer = sorted(answer.items(), key = lambda x: x[1], reverse = True)[0][0]
            answer_lines.append('{} {}\n'.format(file, answer))
            total += 1
            if text_to_class[text] == answer:
                score += 1

with open('answer', 'w') as f:
    f.writelines(answer_lines)
