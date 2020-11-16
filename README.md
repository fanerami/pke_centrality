# pke_centrality
This repository contains code for keyphrases extraction using centrality measure and word embedding (word2vec and Glove).

## Installation
* First download pke(https://github.com/boudinfl/pke)
* Copy file centrality.py to the folder pke/unsupervised/graph_based/
* add the following line in the file pke/unsupervised/__init__.py :
```
from pke.unsupervised.graph_based.centrality import Centrality
```

## How to use (pipeline et arguments)

In the file test.py is an example on how to use the centrality keyphrases extraction

```python
#Create the method

extractor = Centrality()

#Read file

extractor.load_document(input="path/to/file", #path to the file
                        language='en', #language
                        normalization='stemming') #type of preprocessing : 'stemming' or 'lemmatization' or 'None'
                        
#Candidates selection : 
"""
Only one of the two variables pos or grammar should be affected. It depends on the type of candidates. There are three types :

- longest sequence of adjectives and nouns : pos={'NOUN', 'PROPN', 'ADJ'}, grammar=None

- grammatical pattern  : grammar="NP: {<ADJ>*<NOUN|PROPN>+}", pos=None

- grammatical pattern with word limit : grammar="NP: {<ADJ>*<NOUN|PROPN>+}", pos=None and  maximum_word_number=3
"""

extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'}, 
                              grammar=None, 
                              maximum_word_number=0, #maximum number of words composing the keyword. 0 means no limit.
                              stoplist=None)
                              
                              
#Load word2vec model (here the pre-trained by google)

import gensim.models as g

model = g.KeyedVectors.load_word2vec_format('paht/to/word2vecmodel/GoogleNews-vectors-negative300.bin', binary=True)

#keyphrases ranking

"""
Value of embedding: “word2vec”, “glove”, and “doc2vec”

Value of variable typ : 1, 2, and 3

1 – weighting with co-occurrence only. When typ = 1, then the variables d2v, glove and w2v are all equal to None
2 – weighting with the combination of co-occurrence and word/sentence embedding
3 – weighting with word/sentence embedding only
"""

extractor.candidate_weighting(window=2, #co-occurrence window
                              pos=None, #optional
                              embedding=None, #type of embedding model
                              model=model, #instance of embedding model.
                              typ = 2, #type of weighting. Default 1. The other possible values are 2 and 3,
                              alpha = 0.3, #alpha parameter. alpha = 1 means that the candidate's score is the average of the scores of the words that compose it
                              beta = 0.3, #beta parameter
                              centrality='degree_centrality', #Centrality measure to use
                              kwargs=None) #specific variable of the centrality measure. See networkx.

#get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)
```
## Example of loading glove (here pre-trained given by stanford)

```python
def load_glove_model(glove_file, size):
    print("Loading Glove Model")
    f = open(glove_file, 'r')
    model = {}
    vector_size = size
    for line in f:
        split_line = line.split()
        word = " ".join(split_line[0:len(split_line) - vector_size])
        embedding = np.array([float(val) for val in split_line[-vector_size:]])
        model[word] = embedding
    print("Done.\n" + str(len(model)) + " words loaded!")
    return model

model = load_glove_model(glove_model_path+"glove.6B.50d.txt",50)

```

## Example of loading doc2vec (here pre-trained on wikipedia) :
```python
model = g.Doc2Vec.load("path/to/doc2vec.bin")
```

# PARAMETERS ON SOME BENCHMARK DATASET

*INSPEC :

- normalization : "stemming"
-cadidate selection : pos={'NOUN', 'PROPN', 'ADJ'}, grammar=None
- window :  6
- embedding : "word2vec"
- typ : 3
- alpha : 0.5
- beta : 0.1
- centrality : closeness_centrality
- kwargs : {'distance':'inv_weight', 'wf_improved':False}

*TALN-Archives, KDD and WWW :

- normalization : "stemming"
-cadidate selection : pos={'NOUN', 'PROPN', 'ADJ'}, grammar=None
- window :  9
- embedding : None
- typ : 1
- alpha : 0.1
- beta : 0.1
- centrality : eigenvector_centrality_numpy
- kwargs : {'weight':'weight'}

*Semeval :

- normalization : "stemming"
-cadidate selection : pos={'NOUN', 'PROPN', 'ADJ'}, grammar=None
- window :  7
- embedding : "word2vec"
- typ : 3
- alpha : 0.4
- beta : 0.3
- centrality : closeness_centrality
- kwargs : {'distance':'inv_weight', 'wf_improved':False}

*NUS, PubMed and ACM :

- normalization : "stemming"
-cadidate selection : grammar="NP: {<ADJ>*<NOUN|PROPN>+}", pos=None and  maximum_word_number=3
- window :  9
- embedding : None
- typ : 1
- alpha : 0.1
- beta : 0.8
- centrality : degree_centrality
- kwargs : None

*Citeulike-180 :

- normalization : "stemming"
-cadidate selection : grammar="NP: {<ADJ>*<NOUN|PROPN>+}", pos=None and  maximum_word_number=3
- window :  2
- embedding : None
- typ : 1
- alpha : 0.6
- beta : 0.3
- centrality : betweenness_centrality
- kwargs : {'weight':'inv_weight', 'normalized':False}

*500N-KPCrowd :

- normalization : "stemming"
-cadidate selection : grammar="NP: {<ADJ>*<NOUN|PROPN>+}", pos=None and  maximum_word_number=3
- window :  10
- embedding : “word2vec”
- typ : 2
- alpha : 0.9
- beta : 0.9
- centrality : eigenvector_centrality_numpy
- kwargs : {'weight':'weight'}


*DUC-2001 :

- normalization : "stemming"
-cadidate selection : grammar="NP: {<ADJ>*<NOUN|PROPN>+}", pos=None and  maximum_word_number=3
- window :  8
- embedding : None
- typ : 1
- alpha : 0.9
- beta : 0.9
- centrality : hits_numpy
- kwargs : None
