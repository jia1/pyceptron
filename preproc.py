import os

src_dir = os.path.abspath('tc')
dst_dir = os.path.abspath('tc_proc')

for curr_dir, sub_dir, files in os.walk(src_dir):
    for file in files:
        pass
