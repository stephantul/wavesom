# wavesom

This repository contains the code for the poster "A self-organizing model of the bilingual reading system", as presented at [AMLAP 2017](http://wp.lancs.ac.uk/amlap2017/).

You can find the poster and a blogpost about the model [here](https://stephantul.github.io/dynamics/2017/09/03/amlap/)

# Experiments

If you would like access to the trained models, please contact me, I'd be happy to supply them together with the code I used to train them.

# Running the code

```python3
from wavesom import Wavesom

# Assumes some saved SOM you'd like to examine.
# The format is the format saved by SOMBER
s = Wavesom.load("path_to_saved_model.json")

# Assumes you have some data, X

representations = []

for x in X:
    # The final representation before converging is the
    # distributed representation for a given vector.
    representations.append(s.converge(x)[-1])

# compare only to a part of your vector.
s._predict_base_part(X, 10)

# Get the current state of the system.
s.state

# Get the item expressed by the current state of the system.
new_repr = s.statify()
```
