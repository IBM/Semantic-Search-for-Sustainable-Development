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
from PyPDF2 import PdfFileWriter, PdfFileReader
from CustomParVec import CustomParVec
from undp_experiments_Utils import getTargetDoc, getInfo, get_all_matches, ria, evaluateByTarget, avgMatches, getUpdates, getResults, generateSpreadsheet, updateGroundTruth


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

#We use the same parameters to create one instance that uses normalized bag of words scaling and another that uses tf-idf scaling
par_vec_nbow = CustomParVec(words_by_line, num_workers, num_features, min_word_count, context, downsampling, False)
par_vec_tfidf = CustomParVec(words_by_line, num_workers, num_features, min_word_count, context, downsampling, True)

#We will also experiment with Google's pre-trained word2vec model which has 300 dimensions.
model_google = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
par_vec_google = CustomParVec(words_by_line, num_workers, 300, min_word_count, context, downsampling, True, model_google)

#Let's place our models in a single list
par_vecs = [par_vec_google, par_vec_nbow, par_vec_tfidf]

#data for any prior RIA we would like to test
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


#We can test a country for which a RIA was previously conducted here
exclude_ria = all_exclude_ria[0]
policy_documents = all_policy_documents[0]
target_matches = getTargetDoc(template_data_path, exclude_ria)
targs, targ_vecs, sents = getInfo(par_vec_tfidf, target_matches)
score_dict = ria(documents_path, policy_documents, par_vec_tfidf, sents, targ_vecs, targs)

exclude_test = [file for file in os.listdir(template_data_path) if file not in all_exclude_ria[0]]
test_development_matches = parseundp.extract_template_data(template_data_path, exclude_test)
test_target_matches = parseundp.create_target_dictionary(test_development_matches)
    
match_by_sent = evaluateByTarget(score_dict, test_target_matches, 301)
avg_new = avgMatches(match_by_sent, test_target_matches, 301)

avg_new[30]

#conduct a ria
#Retrieve data from prior RIAs
target_matches = getTargetDoc(template_data_path)
targs, targ_vecs, sents = getInfo(par_vec_tfidf, target_matches)

#Specify policy documents to be used for RIA of new country
#policy_documents_png = ['DRAFT National Lukautim Pikinini policy R1-12p.pdf', 'MIDTERM REVIEW PNG NATIONAL HIV AIDS STRATEGY 2011-15 - DRAFT REPORT.pdf', 'Mitigation Policy.pdf', 'MTDP2.pdf', 'National Education Plan 2015 - 2019 11 May  2015final.pdf', 'National Higher and Technical Education Plan2015-2024.pdf', 'National Policy fo4 Women and Genda Equality 2011-2015.pdf', 'NATIONAL STRATEGY FORRESPONSIBLESUSTAINABLEDEVELOPMENT1.from Dilli final.pdf', 'NPP2015-202.21stMay2015.pdf', 'pg-nbsap-01-en.pdf', 'PNG Interim Action Plan for Climate Change.pdf', 'PNG National DRRRM Framework for Action 2005_2015.pdf', 'PNG National Health Plan_Part1.pdf', 'PNG Organic Law 1998.pdf', 'PNG Protected Areas Policy-NEC Approved_Signed.pdf', 'PNG-NPS-GESI-Policy-.pdf', 'WaSH_POLICY 04.03.2015.pdf']
policy_documents_png = ['DRAFT National Lukautim Pikinini policy R1-12p.txt', 'MIDTERM REVIEW PNG NATIONAL HIV AIDS STRATEGY 2011-15 - DRAFT REPORT.txt', 'Mitigation Policy.txt', 'MTDP2.txt', 'National Education Plan 2015 - 2019 11 May  2015final.txt', 'National Higher and Technical Education Plan2015-2024.txt', 'National Policy fo4 Women and Genda Equality 2011-2015.txt', 'NATIONAL STRATEGY FORRESPONSIBLESUSTAINABLEDEVELOPMENT1.from Dilli final.txt', 'NPP2015-202.21stMay2015.txt', 'pg-nbsap-01-en.txt', 'PNG Interim Action Plan for Climate Change.txt', 'PNG National DRRRM Framework for Action 2005_2015.txt', 'PNG National Health Plan_Part1.txt', 'PNG Organic Law 1998.txt', 'PNG Protected Areas Policy-NEC Approved_Signed.txt', 'PNG-NPS-GESI-Policy-.txt', 'WaSH_POLICY 04.03.2015.txt']

policy_documents = policy_documents_png

#Conduct a RIA with the desired model
score_dict_tf = ria(documents_path, policy_documents, par_vec_tfidf, sents, targ_vecs, targs)

#Generate the results spreadsheet
for_devika_png = getResults(score_dict_tf, 10)
generateSpreadsheet(for_devika_png, 'RIA_PNG.xlsx')

#Update model for future RIA
updates = getUpdates('RIA_PNG.xlsx')
updates['4.6']

shelf = shelve.open('RIA_Data')
ground_truth = shelf['ground_truth']
shelf.close()
ground_truth['4.6']

updateGroundTruth(updates)

shelf = shelve.open('RIA_Data')
ground_truth = shelf['ground_truth']
shelf.close()
ground_truth['4.6']

