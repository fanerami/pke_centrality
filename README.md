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
```
