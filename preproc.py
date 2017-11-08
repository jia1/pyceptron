from porter import PorterStemmer
from string import punctuation
import os

src_dir = os.path.abspath('tc')
dst_dir = os.path.abspath('tc_proc')

# 1. Load all stop words into a list
with open('stopword-list', 'r') as s:
    stop_list = list(map(lambda ln: ln.strip(), s.readlines()))

def strip_and_filter_line(ln):
    # A. Split each line by space into tokens
    # B. Strip all default white space characters from each token
    # C. Remove punctuation from each token
    # D. Return a list of tokens which are not stop words
    tokens = map(lambda t: t.strip().strip(punctuation).lower(), ln.split(' '))
    return list(filter(lambda t: t and len(t) > 2 and t.isalpha() and t not in stop_list, tokens))

def chi_square(w, c):
  n00 = 0
  n01 = 0
  n10 = 0
  n11 = 0
  return ((n11+n10+n01+n00)*(n11*n00-n10*n01)**2)/((n11+n01)*(n11+n10)*(n10+n00)*(n01+n00))

def feature_select(d):
  pass

p = PorterStemmer()
k = 2

for curr_dir, sub_dir, files in os.walk(src_dir):
    curr_class = curr_dir.split('/')[-1]
    for file in files:
        flat_text = []
        freq_dict = {}
        with open(os.path.join(curr_dir, file), 'r') as f:
            processed_lines = map(lambda ln: strip_and_filter_line(ln), f.readlines())
            for line in processed_lines:
              flat_text.extend(list(map(lambda word: p.stem(word, 0, len(word) - 1), line)))
            for word in flat_text:
              if word not in freq_dict:
                freq_dict[word] = 1
              else:
                freq_dict[word] += 1
            fin_freq_dict = { word: freq for word, freq in freq_dict.items() if freq >= k }
            if not fin_freq_dict:
              fin_freq_dict = freq_dict
            # Normalize the frequencies based on fin_freq_dict
            sum_freq = sum(fin_freq_dict.values())
            normalized_freq_dict = { word: freq / sum_freq for word, freq in fin_freq_dict.items() }
            # TODO:
            # Figure out the sequence of steps
            # Normalize then feature select, or
            # Feature select then normalize amongst the selected ones?
            feature_select(normalized_freq_dict)

# translator = str.maketrans('', '', string.punctuation)
# some_string.translate(translator)