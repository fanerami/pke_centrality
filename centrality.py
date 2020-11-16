# -*- coding: utf-8 -*-
# Authors: Ygor Gallina, Florian Boudin
# Date: 10-18-2018

"""TextRank keyphrase extraction model.

Implementation of the TextRank model for keyword extraction described in:

* Rada Mihalcea and Paul Tarau.
  TextRank: Bringing Order into Texts
  *In Proceedings of EMNLP*, 2004.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging
from collections import defaultdict
import string
import bisect
import sys

import networkx as nx
import math
from scipy import spatial

from pke.base import LoadFile
from pke.data_structures import Candidate


class Centrality(LoadFile):
    """Centrality for keyword extraction.

    Parameterized example::

        import pke

        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}
        
        # load wor2vec model
        model_w2v = gensim.KeyedVectors.load_word2vec_format('paht/to/word2vecmodel/GoogleNews-vectors-negative300.bin', binary=True)

        # 1. create a Centrality extractor.
        extractor = pke.unsupervised.Centrality()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization="stemming")

        # 3. build the graph representation of the document and rank the keyphrases.
        #    Degree centrality is used to rang words in the graph
        #    the edge of the graph is a result of the combination of co-occurrence and word/sentence embedding
        
        extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'}, 
                                      grammar=None, 
                                      maximum_word_number=0,
                                      stoplist=None)
                                      
        extractor.candidate_weighting(window=2,
                                      pos=pos,
                                      embedding="word2vec",
                                      model=model_w2v,
                                      typ = 2,
                                      alpha = 0.3,
                                      beta = 0.3,
                                      centrality='degree_centrality',
                                      kwargs=None)

        # 4. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
    """

    def __init__(self):
        """Redefining initializer for TextRank."""

        super(Centrality, self).__init__()

        self.graph = nx.Graph()
        """The word graph."""

        self.d2v = None
        """ The doc embedding model. """

        self.Glove = None
        """ The glove embedding model. """

        self.word2vec = None
        """ The glove embedding model. """

        self.typ = 1
        """ type of the edge's weight: 1-coocurrence; 2-coocurrence + embedding; 3-embedding. """

    def candidate_selection(self, pos=None, grammar=None, maximum_word_number=0, stoplist=None):
        """Candidate selection using longest sequences of PoS.

        Args:
            pos (set): set of valid POS tags, defaults to ('NOUN', 'PROPN',
                'ADJ').
        """
        

        self.candidates = defaultdict(Candidate)
        """Keyphrase candidates container (dict of Candidate objects) Initialization."""

        if pos is None:        

            # select sequence of adjectives and nouns
            self.grammar_selection(grammar=grammar)

            if maximum_word_number !=0:
                # filter candidates greater than 3 words
                for k in list(self.candidates):
                    v = self.candidates[k]
                    if len(v.lexical_form) > maximum_word_number:
                        del self.candidates[k]

        else:
            # select sequence of adjectives and nouns
            self.longest_pos_sequence_selection(valid_pos=pos)

        #for c in self.candidates:



        # initialize stoplist list if not provided
        if stoplist is None:
            stoplist = self.stoplist

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=list(string.punctuation) +
                                          ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] +
                                          stoplist)
        #print(self.candidates.keys())

    def build_word_graph(self, window=10, pos=None):
        """Build a graph representation of the document in which nodes/vertices
        are words and edges represent co-occurrence relation. Syntactic filters
        can be applied to select only words of certain Part-of-Speech.
        Co-occurrence relations can be controlled using the distance (window)
        between word occurrences in the document.

        The number of times two words co-occur in a window is encoded as *edge
        weights*. Sentence boundaries **are not** taken into account in the
        window.

        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        """


        self.graph = nx.Graph()
        """The word graph initialization."""

        start_alpha=0.01
        infer_epoch=1000

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # flatten document as a sequence of (word, pass_syntactic_filter) tuples
        text = [(word, sentence.pos[i] in pos) for sentence in self.sentences
                for i, word in enumerate(sentence.stems)]
        true_text = [(word, sentence.pos[i] in pos) for sentence in self.sentences
                for i, word in enumerate(sentence.words)]

        # add nodes to the graph
        self.graph.add_nodes_from([word for word, valid in text if valid])

        # add edges to the graph
        for i, (node1, is_in_graph1) in enumerate(text):

            # speed up things
            if not is_in_graph1:
                continue

            for j in range(i + 1, min(i + window, len(text))):
                node2, is_in_graph2 = text[j]
                if is_in_graph2 and node1 != node2:
                    sim_weight = 0.0
                    if self.embedding =="doc2vec":
                        text1, temp = true_text[i]
                        text2, temp = true_text[j]
                        vect_node1 = self.model.infer_vector(text1, alpha=start_alpha, steps=infer_epoch)
                        vect_node2 = self.model.infer_vector(text2, alpha=start_alpha, steps=infer_epoch)
                        sim_weight = abs(1 - spatial.distance.cosine(vect_node1, vect_node2))
                    elif self.embedding =="glove":
                        text1, temp = true_text[i]
                        text2, temp = true_text[j]   
                        if text1 in self.model and text2 in self.model:
                            vect_node1 = self.model[text1]
                            vect_node2 = self.model[text2]
                            sim_weight = abs(1 - spatial.distance.cosine(vect_node1, vect_node2))
                    elif self.embedding =="word2vec":
                        text1, temp = true_text[i]
                        text2, temp = true_text[j]   
                        if text1 in self.model.wv.vocab and text2 in self.model.wv.vocab:
                            sim_weight = abs(self.model.similarity(text1,text1))
                    if not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(node1, node2, weight=0.0)
                        if self.typ == 3:
                            self.graph[node1][node2]['weight'] = sim_weight
                   
                    if self.typ == 1:
                        self.graph[node1][node2]['weight'] += 1.0
                    elif self.typ == 2:
                        self.graph[node1][node2]['weight'] += sim_weight

        for node1, node2 in self.graph.edges():
            self.graph[node1][node2]["inv_weight"] = 0.0
            if self.graph[node1][node2]['weight'] != 0:
                self.graph[node1][node2]["inv_weight"] = 1.0 / self.graph[node1][node2]['weight']
    
    def length_candidate(self, list_candidate, alpha=0.1):
        """
        Compute candidate score ccording to it length (number of words).
        """
        if len(list_candidate) > 1:
            return math.log2(len(list_candidate))
        else:
            return alpha

    def score_candidates(self, scores, keyphrase_candidates, alpha = 0.1):
        """
        Compute candidate scores according to the word scores given in 
        parameter and return an ordered list of (score, keyphrase) tuples. The 
        score of a candidate keyphrase is computed by summing the scores of the 
        words it contains normalized by its length + 1 to favor longer n-grams.
        """
        #alpha = 0.1 
        scored_candidates = []
        if alpha != 1:
            # Compute the score of each candidate according to its words
            for keyphrase in keyphrase_candidates:
                cands = keyphrase.split(' ')
                if len(cands) == 1 and scores[cands[0]] == 0:
                    score = 0
                else:
                    sum_inverse_score = 0
                    for word in cands:
                        if scores[word] !=0:
                            sum_inverse_score += (1/scores[word])

                    if sum_inverse_score !=0:
                        score = (len(cands)*self.length_candidate(cands, alpha))/sum_inverse_score
                    else:
                        score = 0

                #if self.use_tags:
                #    keyphrase = self.remove_pos(keyphrase)

                bisect.insort(scored_candidates, (score, keyphrase))
        else:
            for keyphrase in keyphrase_candidates:
                cands = keyphrase.split(' ')
                score = 0
                for word in cands:
                    score = score + scores[word]

                score = score/(len(cands)+1)

                bisect.insort(scored_candidates, (score, keyphrase))

        scored_candidates.reverse()

       # print scored_candidates

        return scored_candidates



    def final_score(self, candidates,scores,candidate_count,beta = 0.1, P = True):
        """
        Compute candidate score according to others candidates scores. This avoids the overlap of candidates.
        """        
        scored_candidates = []
        for candidate in candidates:
            cands = candidate.split()
            Ta =[]
            for cand in candidates:
                if cand == candidate:
                    continue
                if candidate in cand:
                    Ta.append(cand)
            score = self.length_candidate(cands,beta)*scores[candidate]*candidate_count[candidate]
            if len(Ta) > 0:
                score_Ta = 0
                for t in Ta:
                    score_Ta += scores[t]

                score = self.length_candidate(cands,beta)*((scores[candidate]*candidate_count[candidate]) - ((1/len(Ta))*score_Ta))

            bisect.insort(scored_candidates, (score, candidate))
        scored_candidates.reverse()
        return scored_candidates 
    
    def candidate_weighting(self,
                            window=2,
                            pos=None,
                            embedding=None,
                            model=None,
                            typ = 1,
                            alpha = 0.1,
                            beta = 0.1,
                            centrality='degree_centrality',
                            kwargs=None):
        """Tailored candidate ranking method for TextRank. Keyphrase candidates
        are either composed from the T-percent highest-ranked words as in the
        original paper or extracted using the `candidate_selection()` method.
        Candidates are ranked using the sum of their (normalized?) words.

        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 2.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
            top_percent (float): percentage of top vertices to keep for phrase
                generation.
            normalized (False): normalize keyphrase score by their length,
                defaults to False.
        """
        self.embedding = embedding
        self.model = model
        self.typ = typ
                 



        self.weights = {}
        """Weight container (can be either word or candidate weights) initialization."""

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # build the word graph
        self.build_word_graph(window=window, pos=pos)

        # compute the word scores using centrality measure
        w = None
        if kwargs is not None:
            if centrality == 'hits' or centrality == 'hits_numpy':
                w = getattr(nx,centrality)(self.graph,**kwargs)[1]
            else:
                w = getattr(nx,centrality)(self.graph,**kwargs)
        else:
            if centrality == 'hits' or centrality == 'hits_numpy':
                w = getattr(nx,centrality)(self.graph)[1]
            else:
                w = getattr(nx,centrality)(self.graph)
        #w = nx.pagerank_scipy(self.graph, alpha=0.85, tol=0.0001, weight=None)
        # generate the phrases from the T-percent top ranked words
        
        #self.weights = w
        
        #original text
        text = " ".join([word for sentence in self.sentences
                for i, word in enumerate(sentence.stems)])
        
        #print(text)
        
        #print(self.candidates.keys())
        #print(w)
        
        #count the number of candidates in the document
        candidates_count = {}
        
        for candidate in self.candidates.keys():
            candidates_count[candidate] = text.count(" "+candidate+" ") 
        
        #print(candidates_count)
        
        candidates_scores_temp = {}
        for candidate in self.score_candidates(w, self.candidates.keys(), alpha):
            candidates_scores_temp[candidate[1]] = candidate[0]
            
        #print(candidates_scores_temp)
        
        f_scores = self.final_score(self.candidates.keys(), candidates_scores_temp,candidates_count, beta)
        
        #print(f_scores)
        
        for f_sc in f_scores:
            self.weights[f_sc[1]] = f_sc[0]
            
        #print(self.weights)
