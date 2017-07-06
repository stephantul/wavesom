

def read_blp_format(filename, words):

    words = set(words)

    for line in open(filename):

        word, _, rt, *rest = line.strip().split("\t")

        if word not in words:
            continue
        yield((word, float(rt)))
