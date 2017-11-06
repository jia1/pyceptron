import os

src_dir = os.path.abspath('tc')
dst_dir = os.path.abspath('tc_proc')

with open('stopword-list', 'r') as s:
    stop_list = list(map(lambda ln: ln.strip(), s.readlines()))

def rm_stop(ln):
    tokens = ln.split(' ')
    return list(filter(lambda t: t not in stop_list, tokens))

for curr_dir, sub_dir, files in os.walk(src_dir):
    curr_class = curr_dir.split('/')[-1]
    for file in files:
        with open(os.path.join(curr_dir, file), 'r') as f:
            filtered_lines = list(map(lambda ln: rm_stop(ln.strip()), f.readlines()))
