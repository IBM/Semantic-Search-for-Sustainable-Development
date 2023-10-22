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
#%%
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from logging import Logger
from custom_logger import CustomLogger

#%%

class CustomParVec():
    '''
    Custom Paragraph Vector. Each paragraph(or sentence) is the sum of each of its word's Word2Vec vector 
    representation scaled by that word's tf-idf.
    '''
    def __init__(
            self,
            word_sentence_list, 
            workers=2, 
            dimensions=100, 
            min_word_count=2, 
            context=5, 
            downsampling=0, 
            tfidf=True, 
            pre_trained=None, 
            supervised_docs=None,
            logger: Logger = None
        ):
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
        
        self.logger = logger or CustomLogger(self.__class__.__name__, write_local=True)

        self.dimensions = dimensions  # Set the number of dimension
        if not pre_trained:
            self.logger.info("init a Word2Vec model...")
            self.word2vec_model = Word2Vec(word_sentence_list, workers=workers, \
                                           vector_size=self.dimensions, min_count = min_word_count, \
                                           window = context, sample = downsampling)
            self.word2vec_model.init_sims(replace=True) # used for memory efficiency
        else:
            self.logger.info("Use a pre-trained model.")
            self.word2vec_model = pre_trained
        
        self.sentences = [' '.join(words) for words in word_sentence_list] # Keep a list of the full sentences themselves.
        
        self.logger.info("Create tf_idf matrix for the input word_sentence_list...")
        self.tf_idf_obj = TfidfVectorizer(use_idf = tfidf) # Create TfidfVectorizer object
        self.tf_idf_obj.fit(self.sentences)  # Transform and fit tf-idf to all sentences(could be paragraphs)
        self.tf_idf = self.tf_idf_obj.transform(self.sentences) # A matrix of tf-idf values for each word in each sentence
        self.word_index = self.tf_idf_obj.get_feature_names_out() # Keep track of words by index for lookups
        
        
        if supervised_docs:
            self.logger.info("Create tf_idf matrix for the input supervised_docs...")
            self.extra_tf_idf_obj = TfidfVectorizer(use_idf = tfidf) # Create TfidfVectorizer object
            self.extra_tf_idf_obj.fit(supervised_docs)  # Transform and fit tf-idf to all sentences(could be paragraphs)
            self.extra_tf_idf = self.extra_tf_idf_obj.transform(supervised_docs) 
            self.extra_word_index = self.extra_tf_idf_obj.get_feature_names_out() # Keep track of words by index for lookups
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

        self.logger.info("Starting learnVectors...")
        for row, col in zip(rows, cols):
            if curr_line == row: # Check that the current word belongs to the same paragraph (or sentence).
                try:
                    # Infer the vector of the current word by scaling the word's word2vec vector by its tf-idf value.
                    # Add that inferred vector to the current vector representing the current paragraph.
                    curr_vec += (self.word2vec_model.wv[(self.word_index[col])] * self.tf_idf[row, col])
                except Exception as e:
                    self.logger.error("Error in learnVectors (if curr_line == row): %s", e)
                    raise e
            else:
                # If we are on the next paragraph, yield the current vector and reset it.
                yield(curr_vec)
                curr_line = row
                curr_vec = np.zeros(self.dimensions)
                try:
                    curr_vec = self.word2vec_model.wv[(self.word_index[col])] * self.tf_idf[row, col]
                except Exception as e:
                    self.logger.error("Error in learnVectors (if curr_line != row): %s", e)
                    raise e
    
    def train(self):
        self.vectors = list(self.learnVectors())
        
    def getMostSimilar(
            self, 
            sentence, 
            top_n=10, 
            threshold=0.5, 
            sentences=None, 
            vectors=None
        ):
        '''
        Given a new sentence, find the closest top_n sentences from the corpus
 
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

        # if corpus and vectors are provided, use them
        # If not, use the learned vectors and original corpus.
        if sentences and vectors:
            corpus = sentences
            vecs = vectors
        else:
            corpus = self.sentences
            vecs = self.vectors  # This will call train() which, in turns, call learnVectors()
        
        # self.logger.info("Calculate cosine similarities between inferred vector and existing vectors...")
        cos_similarities = np.ravel(cosine_similarity(inferred_vector.reshape(1,-1), vecs))

        # Return a list of indices of the top_n most similar sentences.
        # self.logger.info("Get the top_n most similar sentences by cosine similarities...")
        most_similar = np.argpartition(-cos_similarities, top_n)[:top_n]

        # Return list of tuples, each tuple contains cosine similarities and the top nearest sentence
        # Only select the sentences whose cosine similarity is greater than the threshold
        results = [
            (cos_similarities[sentence_index], corpus[sentence_index]) 
            for sentence_index in most_similar 
            if cos_similarities[sentence_index] >= threshold
        ]

        return results
                           
    def inferVector(self, line):
        if self.extra_tf_idf_obj:
            return self.inferVector2(line)
        return self.inferVector1(line)
    

    def inferVector1(self, line):
        '''
        Given a new line, infer a custom vector representation using the provided corpus's tfidf matrix
 
        Args: 
            line : new sentence to be inferred

        Returns: 
            numpy.ndarray : vector representation of the line
        '''

        self.logger.info("Infer the tf-idf values for the words in the provided line")
        line = ' '.join(simple_preprocess(line)) # pre-process the line
        line_tf_idf = self.tf_idf_obj.transform([line])
        rows, cols = line_tf_idf.nonzero()
        
        new_vec = np.zeros(self.dimensions)
        # Apply the same sentence to vector conversion as above. 
        errors = 0
        for col in cols:

            # Ha: If the word is not in the word2vec model, then set tf-idf value as 0.
            # Not sure if this is a valid way to handle this case.
            if self.word_index[col] not in self.word2vec_model.wv:
                errors += 1
                new_vec += (line_tf_idf[0, col])
            else:    
                new_vec += (self.word2vec_model.wv[(self.word_index[col])] * line_tf_idf[0, col])
        
        # self.logger.warning("There are %s words not existing in the trained word2Vec model", errors)
        
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
        
        self.logger.info("Infer the tf-idf values for the words in the provided line")
        
        # Check each word in the line. If the word does not exist in the ground truth tf-idf vocab,
        # try to find similar word in the word2vec model, then replace the word with the similar word.
        replacement_words = []

        for word in line.split():
            if word not in self.extra_tf_idf_obj.vocabulary_:
                try:
                    similar_words = self.word2vec_model.similar_by_word(word, topn=10, restrict_vocab=None)
                    
                    for sim in similar_words:
                        if sim[0] in self.extra_tf_idf_obj.vocabulary_:
                            replacement_words.append((word, sim[0]))
                            break

                except Exception as e:
                    self.logger.error("Error in inferVector2 when finding similar replacement words: %s", e)
                    raise e
                    
        for old, new in replacement_words:
            line = line.replace(old, new)
        
        
        # infer the tf-idf values for the words in the line
        line_tf_idf = self.extra_tf_idf_obj.transform([line]) 
        rows, cols = line_tf_idf.nonzero()
        
        new_vec = np.zeros(self.dimensions)

        # Apply the same vector conversion as above. 

        for col in cols:
            try:    
                new_vec += (self.word2vec_model[(self.extra_word_index[col])] * line_tf_idf[0, col])
            except Exception as e:
                self.logger.error("Error in inferVector2 when converting vectors: %s", e)
                raise e
                            
        return np.asarray(new_vec)