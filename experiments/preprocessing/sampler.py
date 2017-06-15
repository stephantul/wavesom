import numpy as np


def random_sample(words, counts, desired_size, min_count=1):

    assert(len(words) == len(counts))

    output = words * min_count

    print(len(set(output)))

    counts = counts / np.sum(counts)
    start = 0.0
    probas = np.zeros_like(counts)
    for idx, x in enumerate(counts):
        start += x
        probas[idx] = start

    probas = np.array(probas)

    for x in np.random.rand(desired_size - (min_count * len(words))):
        z = probas - x
        z = z > 0
        if not np.any(z):
            selected = words[-1]
        else:
            selected = words[np.flatnonzero(z)[0]]

        output.append(selected)

    return output