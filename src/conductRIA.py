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
import os
import parseundp
import custom_model as cpv
import undp_experiments_Utils as utils
from gensim import models
from custom_logger import CustomLogger
from pathlib import Path

#%%
proj_dir = Path(__file__).parent.absolute()
logger = CustomLogger(__name__, write_local=True)

#if you want to use the Google news pre-trained model, set below to true
USE_GOOGLE_NEWS = False

#To start, we specify the paths that hold our policy documents and template2 RIA data. 
#We should also specify any documents we wish to exclude. We will learn our embeddings using the undp target descriptions 
#and all of the policy documents including the new country we wish to produce an RIA for.

documents_path = proj_dir / 'data/documents/'
template_data_path = proj_dir / 'data/template2/'
shelf_name = str(proj_dir / 'undp')
exclude_documents = []

# If a dictionary of the target descriptions has already been created, load it, 
# Otherwise, create it

if os.path.exists(shelf_name + '.db'):
    targets_only = utils.open_shelve_object(shelf_name, 'targets_only')
else:

    # Create targets_only object
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    i = 0
    targets_only = {}
    with open(proj_dir / 'targets.txt', 'r', encoding='utf-8') as file:
        for line in file:
            x = line.split()
            if x[0][-1] not in alphabet:
                targets_only[x[0]] = ' '. join(x[1:])

    utils.create_shelve_object(shelf_name, targets_only, 'targets_only')

    # Create targets-matches object
    development_matches = parseundp.extract_template_data(template_data_path)
    target_matches = parseundp.create_target_dictionary(development_matches)
    utils.create_shelve_object(shelf_name, target_matches, 'targets')

#%%
# NOTE: change chunk_lines_number to the number of lines you want to chunk together
# to create an input paragraph for the paragraph vector model.
# Longer paragraph takes longer to train.

# Next we create our corpus of Doc2Vec tagged documents in which each document 
# is a paragraph/sentence from the documents 
# in the documents path as well as the target descriptions.
corpus = list(parseundp.read_corpus(documents_path, exclude_documents, targets_only, chunk_lines_number=1))

#%%
# Once we have our corpus of Doc2Vec tagged documents, we create a list 
# in which every member is a list of words of that line.
# words_by_line actually lists the words in entire paragraphs 
# (i.e. each line is one or more paragraphs) in the document
words_by_line = [entry.words for entry in corpus]


# Here we create instances of our custom paragraph vector model.
# Set values for various parameters
num_features = 4000    # Word vector dimensionality                      
min_word_count = 30   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 30          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# We use the same parameters to create one instance that uses 
# normalized bag of words scaling and another that uses tf-idf scaling

logger.info('Creating custom paragraph vector models...')
if USE_GOOGLE_NEWS:
    logger.info('Loading Google News pre-trained model...')
    model_google = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
    par_vec_google = cpv.CustomParVec(
        word_sentence_list=words_by_line,
        workers=num_workers,
        dimensions=300,
        min_word_count=min_word_count,
        context=context,
        downsampling=downsampling,
        tfidf=True,
        pre_trained=model_google
    )

    # Put models in a single list
    
else:
    par_vec_nbow = cpv.CustomParVec(
        word_sentence_list=words_by_line,
        workers=num_workers,
        dimensions=num_features,
        min_word_count=min_word_count,
        context=context,
        downsampling=downsampling,
        tfidf=False
    )

    par_vec_tfidf = cpv.CustomParVec(
        word_sentence_list=words_by_line,
        workers=num_workers,
        dimensions=num_features,
        min_word_count=min_word_count,
        context=context,
        downsampling=downsampling,
        tfidf=True
    )


# We will also experiment with Google's pre-trained word2vec model which has 300 dimensions.
    
#%%

# Data for any prior RIA we would like to test
policy_documents_liberia   = [
    'Liberia Agenda for Transformation.txt', 
    'Liberia Eco stabilization and recovery plan-april_2015.txt'
]
policy_documents_bhutan  = ['Eleventh-Five-Year-Plan_Vol-1.txt', '11th-Plan-Vol-2.txt']
policy_documents_namibia = [
    'na-nbsap-v2-en.txt', 
    'Agri Book with cover1.txt', 
    'execution strategy for industrialisation.txt', 
    'INDC of Namibia Final pdf.txt', 
    'Namibia_Financial_Sector_Strategy.txt', 
    'Tourism Policy.txt', 
    'namibia_national_health_policy_framework_2010-2020.txt', 
    'nampower booklet_V4.txt', 
    '826_Ministry of Education Strategic Plan 2012-17.txt', 
    'Namibia_NDP4_Main_Document.txt'
]

policy_documents_cambodia = [
    'National Strategic Development Plan 2014-2018 EN Final.txt',
    'Cambodia_EducationStrategicPlan_2014_2018.txt',
    'Cambodia Climate Change Strategic Plan 2014_2023.txt',
    'Cambodia Industrial Development Policy 2015_2025.txt',
    'Cambodian Gender Strategic Plan - Neary Rattanak 4_Eng.txt',
    'Draft_HealthStrategicPlan2016-2020.txt',
    'Cambodia_national-disability-strategic-plan-2014-2018.txt',
    'National_Policy_on_Green_Growth_2013_EN.txt',
    'tourism_development_stategic_plan_2012_2020_english.txt',
    'Labour Migration Policy for Cambodia 2015-2018.txt',
    'kh-nbsap-v2-en.txt',
    'financial-sector-development-strategy-2011-2020.txt',
    'National_Social_Protection_Strategy_for_the_Poor_and_Vulnerable_Eng.txt'
]

policy_documents_mauritius = [
    'Agro-forestry Strategy 2016-2020.txt',
    'VISION_14June2016Vision 2030DraftVersion4.txt',
    'Updated Action Plan of the Energy Strategy 2011 -2025.txt',
    'National Water Policy 2014.txt',
    'National CC Adaptioin Policy Framework report.txt',
    'MauritiusEnergy Strategy 2009-2025.txt',
    'Mauritius Govertment programme 2015-2019.txt',
    'CBD Strategy and Action Plan.txt'
]

exclude_ria_liberia   = ['liberia.xlsx']
exclude_ria_bhutan    = ['bhutan_template2.xlsx']
exclude_ria_namibia   = ['namibia_template2.xlsx']
exclude_ria_cambodia  = ['cambodia_template2.xlsx']
exclude_ria_mauritius = ['mauritius.xlsx']

all_exclude_ria = [exclude_ria_liberia, exclude_ria_bhutan, exclude_ria_namibia, exclude_ria_cambodia, exclude_ria_mauritius]
all_policy_documents = [policy_documents_liberia, policy_documents_bhutan, policy_documents_namibia, policy_documents_cambodia, policy_documents_mauritius]

#%%
#We can test a country for which a RIA was previously conducted here
logger.info("Testing Liberia")
exclude_ria = all_exclude_ria[0]
policy_documents = all_policy_documents[0]

#%%
target_matches = utils.getTargetDoc(template_data_path, exclude_ria)

#%%
targs, targ_vecs, sents = utils.getInfo(par_vec_tfidf, target_matches) 
score_dict = utils.ria(documents_path, policy_documents, par_vec_tfidf, sents, targ_vecs, targs)

exclude_test = [file for file in os.listdir(template_data_path) if file not in all_exclude_ria[0]]
test_development_matches = parseundp.extract_template_data(template_data_path, exclude_test)
test_target_matches = parseundp.create_target_dictionary(test_development_matches)


match_by_sent = utils.evaluateByTarget(score_dict, test_target_matches, 301)
avg_new = utils.avgMatches(match_by_sent, test_target_matches, 301)

avg_new[30]

#%%

#conduct a ria
#Retrieve data from prior RIAs


target_matches = utils.getTargetDoc(template_data_path)
targs, targ_vecs, sents = utils.getInfo(par_vec_tfidf, target_matches)

#Specify policy documents to be used for RIA of new country
policy_documents_png = [
    'DRAFT National Lukautim Pikinini policy R1-12p.txt', 
    'MIDTERM REVIEW PNG NATIONAL HIV AIDS STRATEGY 2011-15 - DRAFT REPORT.txt', 
    'Mitigation Policy.txt', 
    'MTDP2.txt', 
    'National Education Plan 2015 - 2019 11 May  2015final.txt', 
    'National Higher and Technical Education Plan2015-2024.txt', 
    'National Policy fo4 Women and Genda Equality 2011-2015.txt', 
    'NATIONAL STRATEGY FORRESPONSIBLESUSTAINABLEDEVELOPMENT1.from Dilli final.txt', 
    'NPP2015-202.21stMay2015.txt', 'pg-nbsap-01-en.txt', 
    'PNG Interim Action Plan for Climate Change.txt', 
    'PNG National DRRRM Framework for Action 2005_2015.txt', 
    'PNG National Health Plan_Part1.txt', 
    'PNG Organic Law 1998.txt', 
    'PNG Protected Areas Policy-NEC Approved_Signed.txt', 
    'PNG-NPS-GESI-Policy-.txt', 
    'WaSH_POLICY 04.03.2015.txt'
]

policy_documents = policy_documents_png

#Conduct a RIA with the desired model
score_dict_tf = utils.ria(documents_path, policy_documents, par_vec_tfidf, sents, targ_vecs, targs)
#%%
#Generate the results spreadsheet
ria_results = utils.getResults(score_dict_tf, 10)
logger.info("Write results to spreadsheet")
utils.generateSpreadsheet(ria_results, 'RIA_PNG.xlsx')

