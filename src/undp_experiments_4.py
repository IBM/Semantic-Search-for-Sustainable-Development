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
from undp_experiments_Utils import getTargetDoc, getInfo, get_all_matches, ria, avgMatches, evaluateByTarget


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

#Experiment 4: Same as experiment 3 except using tf-idf scaling from the target description and prior matches only

use_special_tfidf = {}

for i in range(len(all_exclude_ria)):
    exclude_ria = all_exclude_ria[i]
    policy_documents = all_policy_documents[i]
    hm = list(getTargetDoc(template_data_path, exclude_ria).values())
    extra = [x[0] for x in hm]
    par_vec_tfidf_supervised = CustomParVec(words_by_line, num_workers, num_features, min_word_count, context, downsampling, True, None, extra)
    target_matches = getTargetDoc(template_data_path, exclude_ria)
    targs, targ_vecs, sents = getInfo(par_vec_tfidf_supervised, target_matches)
    score_dict_t = ria(documents_path, policy_documents, par_vec_tfidf_supervised, sents, targ_vecs, targs)
    
    exclude_test = [file for file in os.listdir(template_data_path) if file not in exclude_ria]
    test_development_matches = parseundp.extract_template_data(template_data_path, exclude_test)
    test_target_matches = parseundp.create_target_dictionary(test_development_matches)
    match_by_sent = evaluateByTarget(score_dict_t, test_target_matches, 300)
    avg_new = avgMatches(match_by_sent, test_target_matches, 300)
    use_special_tfidf[all_exclude_ria[i][0]] = [score_dict_t, match_by_sent, avg_new]
    print(all_exclude_ria[i][0])

use_special_tfidf['liberia_supervised_tfidf'] = use_special_tfidf.pop('liberia.xlsx')
use_special_tfidf['bhutan_supervised_tfidf'] = use_special_tfidf.pop('bhutan_template2.xlsx')
use_special_tfidf['namibia_supervised_tfidf'] = use_special_tfidf.pop('namibia_template2.xlsx')
use_special_tfidf['cambodia_supervised_tfidf'] = use_special_tfidf.pop('cambodia_template2.xlsx')
use_special_tfidf['mauritius_supervised_tfidf'] = use_special_tfidf.pop('mauritius.xlsx')

countries = ['liberia', 'bhutan', 'namibia', 'cambodia', 'mauritius']
num_sentences = 30
for key in use_special_tfidf.keys():
    print('{0:10}  {1:10.5f}%'.format(key, use_special_tfidf[key][2][num_sentences]*100))
    #print('-------------------------------------------------------------------------------')

import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
sns.set_context('talk')
sns.set_style("white")
plt.figure(figsize=(15,11))

i =0
for key in use_special_tfidf:
    if 'tfidf' in key:
        plt.plot(list(range(1, 31)), (np.asarray(sorted(use_special_tfidf[key][2]))*100)[:30], label = key.split('.')[0].split('_')[0].upper())
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
    plt.plot(list(range(1, 101)), (np.asarray(sorted(use_special_tfidf['liberia_supervised_tfidf'][1][key]))*100)[:100], label = key)
plt.legend(title = 'Target', bbox_to_anchor=(1.1, .7), loc=1, borderaxespad=10)
plt.title('LIBERIA Percent Matches Vs. Number of Sentences by Target')
plt.xlabel('Number of Sentences')
plt.ylabel('Percent Matches with Policy Experts')
plt.yticks(np.arange(0, 105, 10))
#plt.savefig('liberia_target_update.jpeg')
plt.show()






