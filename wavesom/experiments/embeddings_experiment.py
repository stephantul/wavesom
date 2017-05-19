import logging
import numpy as np

from reach import Reach
from recursive import Recursive
from som import Som

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    r = Reach.load("/Users/stephantulkens/Documents/corpora/medline_vecs_small.txt")
    r.vectors = r.vectors[:, :32]
    r.norm_vectors = np.nan_to_num(r.vectors[2:] / np.sqrt(np.sum(np.square(r.vectors[2:]), axis=1))[:, np.newaxis])
    r.norm_vectors = np.vstack([r.vectors[:2], r.norm_vectors])

    recsom = Som((10, 10), 32, learning_rate=0.3)
    recsom.train(r.vectors, 1, show_progressbar=True)