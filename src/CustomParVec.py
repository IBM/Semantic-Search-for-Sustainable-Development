"""
   Copyright 2018 IBM Corporation

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



class CustomParVec():
    '''
    Custom Paragraph Vector. Each paragraph(or sentence) is the sum of each of its word's Word2Vec vector 
    representation scaled by that word's tf-idf.
    '''
    def __init__(self, word_sentence_list, workers = 2, dimensions = 100, min_word_count = 2, context = 5, downsampling = 0, tfidf = True, pre_trained = None, supervised_docs = None):
        '''          
        Args:
            word_sentence_list (list[list[str]]) : List of lists of words that make up a paragraph(or sentence).
            workers (int)                        : Number of threads to run in parallel (Default = 2).
            dimensions (int)                     : Vector dimensionality (Default = 100).
            min_word_count (int)                 : Minimum word count (Default = 2).
            context (int)                        : Context window size (Default = 5). 
            downsampling (int)                   : Downsample setting for frequent words (Default = 0).
            tfidf (bool)                         : Specify whether or not to use idf in scaling (Default = True).
            pre_trained (Word2Vec)               : Use a pre-trained Word2Vec model (Default = None).
            supervised_docs (list[str])          : List of sentences from some "ground truth" (Default = None).
        '''
        
        self.dimensions = dimensions # Set the number of dimension
        if not pre_trained:
            self.word2vec_model = Word2Vec(word_sentence_list, workers=workers, \
                                           size=self.dimensions, min_count = min_word_count, \
                                           window = context, sample = downsampling)
            self.word2vec_model.init_sims(replace=True) # used for memory efficiency
        else:
            self.word2vec_model = pre_trained
        
        self.sentences = [' '.join(words) for words in word_sentence_list] # Keep a list of the full sentences themselves.
        
        self.tf_idf_obj = TfidfVectorizer(use_idf = tfidf) # Create TfidfVectorizer object
        self.tf_idf_obj.fit(self.sentences)  # Transform and fit tf-idf to all sentences(could be paragraphs)
        self.tf_idf = self.tf_idf_obj.transform(self.sentences) 
        self.word_index = self.tf_idf_obj.get_feature_names() # Keep track of words by index for lookups
        
        
        if supervised_docs:
            self.extra_tf_idf_obj = TfidfVectorizer(use_idf = tfidf) # Create TfidfVectorizer object
            self.extra_tf_idf_obj.fit(supervised_docs)  # Transform and fit tf-idf to all sentences(could be paragraphs)
            self.extra_tf_idf = self.extra_tf_idf_obj.transform(supervised_docs) 
            self.extra_word_index = self.extra_tf_idf_obj.get_feature_names() # Keep track of words by index for lookups
        else:
            self.extra_tf_idf_obj = None

    def learnVectors(self):
        '''
        Create a vector representation of every paragraph(or sentence) in the initial data provided.

        Yields:
            numpy.ndarray: Next numpy array representing the paragraph (or sentence). 
        '''
        rows, cols = self.tf_idf.nonzero() # Get the rows and column indices of non zero tf-idf values 
        curr_line = 0
        curr_vec = np.zeros(self.dimensions)
        for row, col in zip(rows, cols):
            if curr_line == row: # Check that the current word belongs to the same paragraph (or sentence).
                try:
                    # Infer the vector of the current word by scaling the word's word2vec vector by its tf-idf value.
                    # Add that inferred vector to the current vector representing the current paragraph.
                    curr_vec += (self.word2vec_model[(self.word_index[col])] * self.tf_idf[row, col])
                except:
                    continue
            else:
                # If we are on the next paragraph, yield the current vector and reset it.
                yield(curr_vec)
                curr_line = row
                curr_vec = np.zeros(self.dimensions)
                try:
                    curr_vec = self.word2vec_model[(self.word_index[col])] * self.tf_idf[row, col]
                except:
                    continue
    
    def train(self):
        self.vectors = list(self.learnVectors())
        
    def getMostSimilar(self, sentence, top_n = 10, threshold = 0.5, sentences = None, vectors = None):
        '''
        Given a new sentence, find the closest top_n elements
 
        Args: 
            sentence(string)              : Text we want to find most similar to.
            top_n (int)                   : Total number of most similar tuples we want returned (Default value is 5).
            threshold (float)             : Minimum Cosine Distance to be returned
            sentences (list[string])      : List of sentences to be compared to
            vectors[list[numpy nd array]] : Vector embedding of sentences

        Returns: 
            list[(float, string)]: A list of (cosine similarity, sentence) tuples of size top_n closest to the 
                                     input sentence.
        '''
        inferred_vector = self.inferVector(sentence)
        if sentences and vectors:
            corpus = sentences
            vecs = vectors
        else:
            corpus = self.sentences
            vecs = self.vectors
            
        cos_similarities = np.ravel(cosine_similarity(inferred_vector.reshape(1,-1), vecs))
        most_similar = np.argpartition(-cos_similarities, top_n)[:top_n]
        return [(cos_similarities[sentence_index], corpus[sentence_index]) for sentence_index in most_similar if cos_similarities[sentence_index] >= threshold]
                           
    def inferVector(self, line):
        if self.extra_tf_idf_obj:
            return self.inferVector2(line)
        return self.inferVector1(line)
    
    def inferVector1(self, line):
        '''
        Given a new line, infer a custom vector representation using the corpus tfidf.
 
        Args: 
            line : new sentence to be inferred

        Returns: 
            numpy.ndarray : vector representation of the line
        '''
        line = ' '.join(simple_preprocess(line)) # pre-process the line
        line_tf_idf = self.tf_idf_obj.transform([line]) # infer the tf-idf values for the words in the line
        rows, cols = line_tf_idf.nonzero()
        
        new_vec = np.zeros(self.dimensions)
        # Apply the same sentence to vector conversion as above. 
        for col in cols:
            try:    
                new_vec += (self.word2vec_model[(self.word_index[col])] * line_tf_idf[0, col])
            except:
                continue
        return np.asarray(new_vec)
    
    def inferVector2(self, line):
        '''
        Given a new line, infer a custom vector representation using the ground truth tfidf.
 
        Args: 
            line : new sentence to be inferred

        Returns: 
            numpy.ndarray : vector representation of the line
        '''
        line = ' '.join(simple_preprocess(line)) # pre-process the line
        
        replacement_words = []
        for word in line.split():
            if word not in self.extra_tf_idf_obj.vocabulary_:
                try:
                    similar_words = self.word2vec_model.similar_by_word(word, topn=10, restrict_vocab=None)
                    for sim in similar_words:
                        if sim[0] in self.extra_tf_idf_obj.vocabulary_:
                            replacement_words.append((word, sim[0]))
                            break
                except:
                    continue
                    
        for old, new in replacement_words:
            line = line.replace(old, new)
            
        line_tf_idf = self.extra_tf_idf_obj.transform([line]) # infer the tf-idf values for the words in the line
        rows, cols = line_tf_idf.nonzero()
        
        new_vec = np.zeros(self.dimensions)
        # Apply the same sentence to vector conversion as above. 
        for col in cols:
            try:    
                new_vec += (self.word2vec_model[(self.extra_word_index[col])] * line_tf_idf[0, col])
            except:
                continue
                            
        return np.asarray(new_vec)