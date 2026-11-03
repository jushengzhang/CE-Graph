import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
_hv = HashingVectorizer(n_features=64, alternate_sign=False, norm="l2")
def local_embed_fn(texts):
    # deterministic local hashing features for offline clustering smoke tests
    return _hv.transform(list(texts)).toarray().astype("float64")
