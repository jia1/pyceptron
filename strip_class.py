with open('test-class-list', 'r') as s:
    with open('test-list', 'w') as d:
        d.writelines(list(map(lambda ln: '{}\n'.format(ln.split(' ')[0]), s.readlines())))
