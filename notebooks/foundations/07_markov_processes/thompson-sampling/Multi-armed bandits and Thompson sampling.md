```python
import numpy as np
import scipy
import scipy.stats as stats
```

We set the number of arms at 5 and choose a true probabilty of winning for each arm by generating a random number between 0 and 1. 

We choose a seed because we want to get the same results every time we run this simulation. To change the results, change or remove the constant value passed in as the seed.


```python
np.random.seed(100)

nBandits = 5
pBandits = [np.random.rand() for i in range(nBandits)]
```


```python
print(pBandits)
```

    [0.5434049417909654, 0.27836938509379616, 0.4245175907491331, 0.8447761323199037, 0.004718856190972565]


The fourth arm has the highest probability of winning at 84%. This is the arm we expect to choose most of the time in an exploration/exploitation scenario. (explain what exploration/exploitation is)


```python
nWins = [0 for i in range(nBandits)]
nPulls = [0 for i in range(nBandits)]
```


```python
def chooseArm(pBandit):
    if np.random.rand() < pBandits[pBandit]:
        return 1
    return 0
```

This is a Bernoulli bandit. Each action taken on it has a binary outcome (each of its arms pays out a reward of either 0 or 1). A conjugate prior for the Bernoulli distribution is the beta distribution. We are going to choose the beta distribution as a prior here because its posterior distribution will be another beta distribution. (clarify all of this)

We are going to run this simulation 1000 times.


```python
n = 1000
```


```python
for trial in range(n):
    beta = [stats.beta(a=1+win, b=1+pull-win) for pull, win in zip(nPulls, nWins)]
    theta = [sample.rvs(1) for sample in beta]
    choice = np.argmax(theta)
    arm = chooseArm(choice)
    nPulls[choice] += 1
    nWins[choice] += arm
```


```python
nPulls
```




    [8, 5, 5, 978, 4]




```python
nWins
```




    [3, 1, 1, 810, 0]




```python
[win / pull for win, pull in zip(nWins, nPulls)]
```




    [0.375, 0.2, 0.2, 0.8282208588957055, 0.0]



The fourth arm gets pulled more than any other arm. The ratio of wins to pulls for this arm is based on the largest sample size. Some of the arms are pulled very few times, so their ratio of wins to pulls is not as close to their true probability of winning.

When we focus on exploitation more than exploration, we end up taking high-valued actions more because we know they are high-valued. We spend less time trying to determine the value of lower-valued actions.


```python

```
