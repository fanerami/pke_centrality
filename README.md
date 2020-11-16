# pke_centrality
This repository contains code for keyphrases extraction using centrality measure and word embedding (word2vec and Glove).

## Installation
* First download pke(https://github.com/boudinfl/pke)
* Copy file centrality.py to the folder pke/unsupervised/graph_based/
* add the following line in the file pke/unsupervised/__init__.py :
```
from pke.unsupervised.graph_based.centrality import Centrality
```
