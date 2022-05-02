## Generative Flow Network demo in TensorFlow2

A Generative Flow Network (GFlowNet) implementation using TensorFlow 2.

#### About GFlowNets

GFlowNets were introduced by Bengio et. al. [here](https://arxiv.org/abs/2106.04399):

Bengio E, Jain M, Korablyov M, Precup D, and Bengio Y. (2021)
Flow Network based Generative Models for Non-Iterative Diverse
Candidate Generation.
NeurIPS.

The Trajectory Balance loss function was introduced [here](https://arxiv.org/abs/2201.13259):

Malkin N, Jain M, Bengio E, Sun C, and Bengio Y. (2022)
Trajectory Balance: Improved Credit Assignment in GFlowNets.
arXiv:2201.13259v1

Offline training is discussed [here](https://arxiv.org/abs/2203.04115):

Moksh J, Bengio E, Garcia A-H, et al. (2022)
Biological Sequence Design with GFlowNets.
arXiv:2203.04115v1


### Getting started

To get started you'll want to install pipenv then run:
```
pipenv install
```

which will build an environment with tensorflow and all the dependencies you'll be needing to complete this tutorial.

To run the demonstration jupyter notebook:

```
pipenv shell
```

and then, once in the pipenv environment:

```
jupyter lab
```

Open the notebook `gflownet_demo.ipynb` and run all cells!

The notebook imports `gfn.py` which is the tensorflow2 implementation of a 
generative flow network.
`gfn.py` in turn, imports `env.py`, an implemention of the simple cube reward 
environment used in the original GFlowNet papers.
