#!/usr/bin/env python

import os
import sys
from pathlib import Path
import argparse

from collections import Counter
import re
import random

import pandas as pd
import numpy as np

import nltk
nltk.download('names')
from nltk.corpus import names

from helper import preprocessing_v2 as pre

def parse_and_join_data(cwd, path):
    
    '''
    Parse raw data and join all reviews in a single dataframe.
    '''
    
    data_list = []
    
    split_dir = Path(os.path.join(cwd,path))

    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            file_name = re.split('_|\.', str(text_file.name))
            
            review_id = file_name[0]
            review_score = int(file_name[1])
            text = text_file.read_text()
            label = 0 if label_dir == "neg" else 1
            
            data_list.append([review_id, review_score, text, label])
                    
    return pd.DataFrame(data_list, columns = ['review_id', 'review_score', 'review_text', 'label'])

def create_vocab(cwd, path):
    
    '''
    Parse the word files and create a vocab list for both female and male
    gendered words, plus lists of female and male gendered names from the NLTK library.
    '''
    with open(os.path.join(cwd, path, 'female_word_file.txt')) as file:
        f_vocab = []
        for w in file.read().split():
            f = w.replace('_', ' ')
            f_vocab.append(f)
    
    with open(os.path.join(cwd, path, 'male_word_file.txt')) as file:
        m_vocab = []
        for w in file.read().split():
            m = w.replace('_', ' ')
            m_vocab.append(m)

    f_names = [name.lower() for name in names.words('female.txt')]
    m_names = [name.lower() for name in names.words('male.txt')]
    
            
    return f_vocab, m_vocab, f_names, m_names

def main(args):
    
    '''
    Preprocess the source data to create neutral, female, and male 
    gendered versions for each review.
    '''
    
    cwd = os.getcwd()
    
    # Parse the data and create the df
    if args.data_path:
        print("Parsing data...")
        df = parse_and_join_data(cwd, args.data_path)
    else:
        raise Exception("No data paths provided. Please use command line argument -d/--data_paths \
                        and specify comma delimited paths to data files.")
    
    # Parse the vocab files and create the vocab and names lists
    if args.vocab_path:
        print("Parsing vocab lists...")
        f_vocab, m_vocab, f_names, m_names = create_vocab(cwd, args.vocab_path)
    else:
        raise Exception("No vocab path provided. Please use command line argument -v/--vocab_path \
                        and specify path to vocab files.")
    
    # Create total vocab and names set
    tot_vocab = set(f_vocab + m_vocab + f_names + m_names)
    tot_m_vocab = set(m_vocab + m_names)
    tot_f_vocab = set(f_vocab + f_names)
    
    # Create mappings between male and female word counterparts
    print("Creating vocab mappings...")
    f_map, m_map = pre.gender_mapping(f_vocab, m_vocab, f_names, m_names)
    
    # Create separate regex objects for male and female vocabs
    m_regex = re.compile('|'.join(r'\b%s\b' %s for s in map(re.escape, tot_m_vocab)))
    f_regex = re.compile('|'.join(r'\b%s\b' %s for s in map(re.escape, tot_f_vocab)))
    
    # Convert all text to neutral reviews
    print("Processing neutral review dataset...")
    df[['neutral_review_text', 'neutral_sub_count']] = df.apply(pre.preprocess_unk, result_type='expand', axis=1, 
                                                                vocab=tot_vocab)
    
    # Convert all text to female gendered reviews
    print("Processing female gendered review dataset...")
    df[['female_review_text', 'male_sub_count']] = df.apply(pre.preprocess_gendered_swap, result_type='expand', axis=1, 
                                                            vocab_map=m_map, regex=m_regex)
    
    # Convert all text to male gendered reviews
    print("Processing male gendered review dataset...")
    df[['male_review_text', 'female_sub_count']] = df.apply(pre.preprocess_gendered_swap, result_type='expand', axis=1,
                                                        vocab_map=f_map, regex=f_regex)
    
    # Reorder the columns
    cols = df.columns.to_list()
    label_index = cols.index('label')
    cols = cols[:label_index] + cols[label_index+1:] + [cols[label_index]]
    
    # Assign reordered df
    new_df = df[cols]
    
    # Save to csv
    new_df_path = os.path.join(cwd, args.output)                  
    new_df.to_csv(new_df_path, index=False)
    print(f"Saved new dataset to {new_df_path}")
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", help="path to data files")
    parser.add_argument("-v", "--vocab_path", help="path to vocab files")
    parser.add_argument("-o", "--output", help="filepath to save output file (requires .csv filename)")
    args = parser.parse_args()
            
    main(args)
    

    
