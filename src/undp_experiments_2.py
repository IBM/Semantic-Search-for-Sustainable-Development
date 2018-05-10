from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from gensim.models import doc2vec
from gensim import models
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher as sm
from collections import OrderedDict
import os
import numpy as np
import itertools
import shelve
from collections import Counter
import parseundp
from CustomParVec import CustomParVec
from undp_experiments_Utils import getTargetDoc, getInfo, get_all_matches, loadTruth, ria, avgMatches, evaluateByTarget


#To start, we specify the paths that hold our policy documents and template2 RIA data. 
#We should also specify any documents we wish to exclude. We will learn our embeddings using the undp target descriptions 
#and all of the policy documents including the new country we wish to produce an RIA for.

documents_path = 'data/documents/'
template_data_path = 'data/template2/'
exclude_documents = []

#I have saved a dictionary of the target descriptions. We load it here.
#shelf = shelve.open('undp.db')
#targets_only = shelf['targets_only']
import pickle
targets_only = pickle.load(open('undp.pkl', 'rb'))
#targets_only['1.1'] gives the text for SDG 1 Target 1.1 and so on
#shelf.close()

#Next we create our corpus of Doc2Vec tagged documents in which each document is a paragraph/sentence from the documents 
#in the documents path as well as the target descriptions.
corpus = list(parseundp.read_corpus(documents_path, exclude_documents, targets_only))

#Once we have our corpus of Doc2Vec tagged documents, we create a list in which every entry is a list of words of that line.
words_by_line = [entry.words for entry in corpus]
#words_by_line actually lists the words in entire paragraphs (i.e. each line is one or more paragraphs) in the document
#Here we create instances of our custom paragraph vector model.

#Set values for various parameters
num_features = 4000    # Word vector dimensionality                      
min_word_count = 30   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 30          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

#We use the same parameters to create one instance that uses normalized bag of words scaling and another that uses tf-idf scaling.

par_vec_nbow = CustomParVec(words_by_line, num_workers, num_features, min_word_count, context, downsampling, False)
par_vec_tfidf = CustomParVec(words_by_line, num_workers, num_features, min_word_count, context, downsampling, True)

#We will also experiment with Google's pre-trained word2vec model which has 300 dimensions
model_google = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
par_vec_google = CustomParVec(words_by_line, num_workers, 300, min_word_count, context, downsampling, True, model_google)

#Let's place our models in a single list
par_vecs = [par_vec_google, par_vec_nbow, par_vec_tfidf]




#Data we will be testing with

policy_documents_liberia   = ['Liberia Agenda for Transformation.txt', 'Liberia Eco stabilization and recovery plan-april_2015.txt']
policy_documents_bhutan    = ['Eleventh-Five-Year-Plan_Vol-1.txt', '11th-Plan-Vol-2.txt']
policy_documents_namibia   = ['na-nbsap-v2-en.txt', 'Agri Book with cover1.txt', 'execution strategy for industrialisation.txt', 'INDC of Namibia Final pdf.txt', 'Namibia_Financial_Sector_Strategy.txt', 'Tourism Policy.txt', 'namibia_national_health_policy_framework_2010-2020.txt', 'nampower booklet_V4.txt', '826_Ministry of Education Strategic Plan 2012-17.txt', 'Namibia_NDP4_Main_Document.txt']
policy_documents_cambodia  = ['National Strategic Development Plan 2014-2018 EN Final.txt', 'Cambodia_EducationStrategicPlan_2014_2018.txt', 'Cambodia Climate Change Strategic Plan 2014_2023.txt', 'Cambodia Industrial Development Policy 2015_2025.txt', 'Cambodian Gender Strategic Plan - Neary Rattanak 4_Eng.txt', 'Draft_HealthStrategicPlan2016-2020.txt', 'Cambodia_national-disability-strategic-plan-2014-2018.txt', 'National_Policy_on_Green_Growth_2013_EN.txt', 'tourism_development_stategic_plan_2012_2020_english.txt', 'Labour Migration Policy for Cambodia 2015-2018.txt', 'kh-nbsap-v2-en.txt', 'financial-sector-development-strategy-2011-2020.txt', 'National_Social_Protection_Strategy_for_the_Poor_and_Vulnerable_Eng.txt']
policy_documents_mauritius = ['Agro-forestry Strategy 2016-2020.txt', 'VISION_14June2016Vision 2030DraftVersion4.txt', 'Updated Action Plan of the Energy Strategy 2011 -2025.txt', 'National Water Policy 2014.txt', 'National CC Adaptioin Policy Framework report.txt', 'MauritiusEnergy Strategy 2009-2025.txt', 'Mauritius Govertment programme 2015-2019.txt', 'CBD Strategy and Action Plan.txt']

exclude_ria_liberia   = ['liberia.xlsx']
exclude_ria_bhutan    = ['bhutan_template2.xlsx']
exclude_ria_namibia   = ['namibia_template2.xlsx']
exclude_ria_cambodia  = ['cambodia_template2.xlsx']
exclude_ria_mauritius = ['mauritius.xlsx']

all_exclude_ria = [exclude_ria_liberia, exclude_ria_bhutan, exclude_ria_namibia, exclude_ria_cambodia, exclude_ria_mauritius]
all_policy_documents = [policy_documents_liberia, policy_documents_bhutan, policy_documents_namibia, policy_documents_cambodia, policy_documents_mauritius]

#Experiment 2: Include matches form prior RIAs in semantic search
include_prior_matches = {}
vec = 1
for par_vec in par_vecs:
    for i in range(len(all_policy_documents)):
        exclude_ria = all_exclude_ria[i]
        policy_documents = all_policy_documents[i]
        target_matches = loadTruth(template_data_path, exclude_ria)
        targs, targ_vecs, sents = getInfo(par_vec, target_matches)
        print(exclude_ria[0][:-5]+str(vec))
        score_dict = ria(documents_path, policy_documents, par_vec, sents, targ_vecs, targs)
        include_prior_matches[exclude_ria[0][:-5]+str(vec)] = [score_dict]
    vec += 1
    
i = 0
for key, val in include_prior_matches.items():
    exclude_test = [file for file in os.listdir(template_data_path) if file not in all_exclude_ria[i]]
    test_development_matches = parseundp.extract_template_data(template_data_path, exclude_test)
    test_target_matches = parseundp.create_target_dictionary(test_development_matches)
    
    print(key, all_exclude_ria[i])
    match_by_sent = evaluateByTarget(val[0], test_target_matches, 301)
    include_prior_matches[key].append(match_by_sent)
    avg_new = avgMatches(match_by_sent, test_target_matches, 301)
    include_prior_matches[key].append(avg_new)

    i+=1
    if i % 5 == 0:
        i = 0

include_prior_matches['liberia_google'] = include_prior_matches.pop('liberia1')
include_prior_matches['liberia_nbow'] = include_prior_matches.pop('liberia2')
include_prior_matches['liberia_tfidf'] = include_prior_matches.pop('liberia3')
include_prior_matches['bhutan_google'] = include_prior_matches.pop('bhutan_template21')
include_prior_matches['bhutan_nbow'] = include_prior_matches.pop('bhutan_template22')
include_prior_matches['bhutan_tfidf'] = include_prior_matches.pop('bhutan_template23')
include_prior_matches['namibia_google'] = include_prior_matches.pop('namibia_template21')
include_prior_matches['namibia_nbow'] = include_prior_matches.pop('namibia_template22')
include_prior_matches['namibia_tfidf'] = include_prior_matches.pop('namibia_template23')
include_prior_matches['cambodia_google'] = include_prior_matches.pop('cambodia_template21')
include_prior_matches['cambodia_nbow'] = include_prior_matches.pop('cambodia_template22')
include_prior_matches['cambodia_tfidf'] = include_prior_matches.pop('cambodia_template23')
include_prior_matches['mauritius_google'] = include_prior_matches.pop('mauritius1')
include_prior_matches['mauritius_nbow'] = include_prior_matches.pop('mauritius2')
include_prior_matches['mauritius_tfidf'] = include_prior_matches.pop('mauritius3')

countries = ['liberia', 'bhutan', 'namibia', 'cambodia', 'mauritius']
num_sentences = 30
for key in include_prior_matches.keys():
    print('{0:10}  {1:10.5f}%'.format(key, include_prior_matches[key][2][num_sentences]*100))
    #print('-------------------------------------------------------------------------------')    


import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
sns.set_context('talk')
sns.set_style("white")
plt.figure(figsize=(15,11))

i =0
for key in include_prior_matches:
    if 'tfidf' in key:
        plt.plot(list(range(1, 31)), (np.asarray(sorted(include_prior_matches[key][2]))*100)[:30], label = key.split('.')[0].split('_')[0].upper())
plt.legend(title = 'Country', bbox_to_anchor=(1.1, 0.45), loc=1, borderaxespad=10)
plt.title('Percent Matches Vs. Number of Sentences')
plt.xlabel('Number of Sentences')
plt.ylabel('Percent Matches with Policy Experts')
plt.yticks(np.arange(0, 55, 5))
#plt.savefig('matches_update_30.jpeg')
plt.show()

plt.figure(figsize=(15,11))

examples = ['1.2', '3.3', '5.1', '9.3', '8.5', '15.2', '16.1']
for key in examples:
    plt.plot(list(range(1, 101)), (np.asarray(sorted(include_prior_matches['liberia_tfidf'][1][key]))*100)[:100], label = key)
plt.legend(title = 'Target', bbox_to_anchor=(1.1, .7), loc=1, borderaxespad=10)
plt.title('LIBERIA Percent Matches Vs. Number of Sentences by Target')
plt.xlabel('Number of Sentences')
plt.ylabel('Percent Matches with Policy Experts')
plt.yticks(np.arange(0, 105, 10))
#plt.savefig('liberia_target_update.jpeg')
plt.show()

