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

import os
import re
import pandas as pd
import numpy as np
from gensim.models import doc2vec
from gensim.utils import simple_preprocess

'''
    Parse through every xlsx file in the specified path. For every sheet in each file, 
    create a pandas data frame and if a column has key words representing undp goals, keep the information in 
    that column.

    Args:
        path (string)          : Directory to the xlsx documents (Current directory by default).
        exclude (list[string]) : List of files to be excluded from data extraction

    Returns:
        data (list[pandas.DataFrame]) : list of pandas data frames. Each data frame has two columns, one being 
                                        the target undpand the other being text identified as matching the target.
'''
def extract_template_data(path = '.', exclude = []):
    data = []
    found = False
    for file in os.listdir(path):
        if file[-4:] == 'xlsx' and '~' not in file and file not in exclude:
            file_path = os.path.join(path, file)
            xls = pd.ExcelFile(file_path)
            for sheet in range(len(xls.sheet_names)):
                template = pd.read_excel(file_path, header = None, sheetname = sheet)
                template.fillna('', inplace = True)
                for col in range(len(template.columns)):
                    try:
                        if 'Identify closest' in str(template.iloc[0][col]) or 'Identify closest' in str(template.iloc[1][col]):
                            keep = col
                            found = True
                        elif 'National Development Plan' in str(template.iloc[1][col]) and not found:
                            keep = col
                            found = True
                    except:
                        continue
                if found:
                    data.append(template[[template.keys()[1], template.keys()[keep]]])
                found = False

    return data


'''
    Creates a dictionary of target : list[sentences], where each sentence was identified to match the target.

    Args:
        development_matches (pandas DataFrame): Data frame with 2 columns, one being the undp target and the other 
                                                being the sentence/paragraph identified to match the target

    Returns:
        target_matches (dict): Dictionary of target:list[sentences]
'''
def create_target_dictionary(development_matches):
    sgd_target_pattern = r'[0-9]+\.[0-9]+' # pattern to match target format
    target_matches = {}
    for development_match in development_matches:
        development_match.replace(np.nan, '', regex=True, inplace = True)
        target = None

        for row in development_match.itertuples():
            match = re.search(sgd_target_pattern, str(row[1]), flags=0)
            if match: # If we found a undp target
                target = match.group()
                if target in target_matches: # Add sentence to the set for that target's key
                    target_matches[target].add(row[1][len(target):])
                else:
                    target_matches[target] = set({row[1][len(target):]})
            # Continue adding to the current target's key if there is text in the data frame
            if target != None and row[2] != '':
                target_matches[target].add(row[2])
        
    return target_matches


'''
    Parse through every english text file in the specified path. For every line in each file,
    pre-process the line (convert to lower case, remove punctuation). Only yield doc2Vec 
    TaggedDocuments for every processed line in which there are more than 10 words and over
    half the words have more than 3 characters.

    Args:
        path (string)          : Directory to the text documents (Current directory by default).
        exclude (list[string]) : List of files to be excluded

    Yields:
        doc2vec.TaggedDocument : Doc2Vec Tagged Document object of the next processed line with a unique id. 
'''
def read_corpus(path = '.', exclude = [], targets = None):
    i= 0
    for file in os.listdir(path):
        if file[-4:] == '.txt' and file not in exclude and 'no_en' not in file: # ensure file is an english txt file
            print(file)
            with open(os.path.join(path, file),  encoding="utf8") as document_text:
                for line in document_text:
                    count = 0
                    words = simple_preprocess(line)
                    for word in words: # count the number of words with <= 3 characters
                        if len(word) <= 3:
                            count += 1
                    if count < len(words)/2 and len(words) > 10: # exclude lines in which 1/2 the words have less 
                        yield(doc2vec.TaggedDocument(words, [i])) # than 3 characters or have less than 10 words
                        i+=1
    if targets:
        for key, val in targets.items():
            yield(doc2vec.TaggedDocument(simple_preprocess(val), [i]))
            i+=1