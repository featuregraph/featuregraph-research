import numpy as np
from scipy.stats import beta

np.random.seed(20)

n = 5
p = [np.random.rand() for i in range(n)]

wins = [0 for i in range(n)]
losses = [0 for i in range(n)]

trials = 1000
for trial in range(trials):
    prior = [beta(win+1, loss+1) for win, loss in zip(wins, losses)]
    sample = [s.rvs(1) for s in prior]
    choice = sample.index(max(sample))
    result = np.random.rand() < p[choice]
    wins[choice] += result
    losses[choice] += 1-result

pulls = [w + l for w, l in zip(wins, losses)]
observed_probabilities = [w / p for w, p in zip(wins, pulls)]
