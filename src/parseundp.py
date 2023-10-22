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
from custom_logger import CustomLogger
from pathlib import Path
from typing import Any, Optional

logger = CustomLogger(__name__, write_local=True)


def extract_template_data(path = '.', exclude = []):
    """
        Parse through every xlsx file in the specified path. For every sheet in each file, 
        create a pandas data frame and if a column has key words representing undp goals, keep the information in 
        that column.

        Args:
            path (string)          : Directory to the xlsx documents (Current directory by default).
            exclude (list[string]) : List of files to be excluded from data extraction

        Returns:
            data (list[pandas.DataFrame]) : list of pandas data frames. Each data frame has two columns, one being 
                                            the target undp and the other being text identified as matching the target.
    """
    data = []
    found = False
    for file in os.listdir(path):
        if file[-4:] == 'xlsx' and '~' not in file and file not in exclude:
            file_path = os.path.join(path, file)
            xls = pd.ExcelFile(file_path)
            logger.info(f"Reading {file}")

            for sheet in range(len(xls.sheet_names)):
                template = pd.read_excel(file_path, header = None, sheet_name = sheet)
                template = template.apply(lambda x: x.astype(str), axis=0) # convert all columns to string
                template.fillna('', inplace = True)

                for col in range(len(template.columns)):
                    try:
                        if 'Identify closest' in str(template.iloc[0][col]) or 'Identify closest' in str(template.iloc[1][col]):
                            keep = col
                            found = True
                        elif 'National Development Plan' in str(template.iloc[1][col]) and not found:
                            keep = col
                            found = True
                    except Exception as e:
                        logger.info(f"---Sheet {sheet}, shape: {template.shape}")
                        logger.error(e)
                        continue

                if found:
                    data.append(template[[template.keys()[1], template.keys()[keep]]])
                found = False

    return data


def create_target_dictionary(development_matches):
    """
        Creates a dictionary of target : list[sentences], where each sentence was identified to match the target.

        Args:
            development_matches (pandas DataFrame): Data frame with 2 columns, one being the undp target and the other 
                                                    being the sentence/paragraph identified to match the target

        Returns:
            target_matches (dict): Dictionary of target:list[sentences]
    """
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
            if target is not None and row[2] != '':
                target_matches[target].add(row[2])
        
    return target_matches


def read_corpus(
        path: str = '.', 
        exclude: Optional[list] = None, 
        targets: Optional[dict] = None, 
        chunk_lines_number: int = 1
    ):
    """
        Parse through every english text file in the specified path. For every line in each file,
        pre-process the line (convert to lower case, remove punctuation). Only yield doc2Vec 
        TaggedDocuments for every processed line in which there are more than 10 words and over
        half the words have more than 3 characters.

        Args:
            path (string)          : Directory to the text documents (Current directory by default).
            exclude (list[string]) : List of files to be excluded
            targets (dict)         : A dictionary containing target texts
            chunk_lines_number (int)  : Number of lines to chunk together to form a single TaggedDocument. 
                If using default value 1, each TaggedDocument represents one line. 
                If using a value > 1, each TaggedDocument represents a chunk of lines (Paragraph)

        Yields:
            doc2vec.TaggedDocument : Doc2Vec Tagged Document object of the next processed line with a unique id. 
    """
    i = 0
    short_word_thres = 3

    txt_files = [str(file.name) for file in Path.glob(Path(path), '*.txt') if 'no_en' not in file.name]
    
    # Exclude files if specified
    if exclude is not None:
        txt_files = [file for file in txt_files if file not in exclude]
    
    # Iterate through files
    for file in txt_files:
        
        with open(os.path.join(path, file),  encoding="utf8") as file:
            chunking_counter = 0
            lines_chunk = []
            
            # Go through lines in file, check if it satisfies the criteria, and add to the chunk.
            # Once the number of lines added equal to the chunk_lines_number, 
            # or when reaching the end of file, yield the chunk
            for line in file:
                
                line_words = simple_preprocess(line)
                short_words = [len(w) < short_word_thres for w in line_words]

                # Filter out some sentences
                if sum(short_words) < len(line_words)/2 and len(line_words) > 10:
                    
                    lines_chunk += line_words
                    chunking_counter += 1
                    
                    if chunking_counter == chunk_lines_number:
                        chunk_output = lines_chunk
                        lines_chunk = []
                        yield (doc2vec.TaggedDocument(chunk_output, [i]))
                        i += 1
                else:
                    # yield the chunk when reaching
                    chunk_output = lines_chunk
                    yield (doc2vec.TaggedDocument(chunk_output, [i]))

    if targets:
        for key, val in targets.items():
            yield (doc2vec.TaggedDocument(simple_preprocess(val), [i]))
            i += 1