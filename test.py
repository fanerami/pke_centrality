import numpy as np
import os
from string import punctuation
from pke.unsupervised import Centrality
import gensim.models as g

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

model = load_glove_model("glove.6B.50d.txt",50)


# define the set of valid Part-of-Speeches
pos = {'NOUN', 'PROPN', 'ADJ'}

# define the grammar for selecting the keyphrase candidates
grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
stoplist = list(punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

normal = "stemming"


extractor = Centrality()

#The contents of kwargs are the parameters for the centrality measure
kwargs = {'distance':'inv_weight', 'wf_improved':False}

extractor.load_document(input=os.path.join('data', '1939.txt'),
                        language='en',
                        normalization=normal)

extractor.candidate_selection(pos=pos,stoplist=stoplist)

extractor.candidate_weighting(window=6,
                              embedding = "glove",
                              model = model,
                              typ = 3,
                              alpha = 0.3,
                              beta = 0.3,
                              centrality='closeness_centrality',
                              kwargs=kwargs)
print(extractor.get_n_best(n=10))

