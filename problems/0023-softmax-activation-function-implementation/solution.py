import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    m = np.max(scores)
    s = np.sum(np.exp(scores-m))
    probs = np.exp(scores-m)/s

    return probs