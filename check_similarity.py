# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:27:43 2022

@author: SEVENDI ELDRIGE RIFKI POLUAN
"""

import math
import string
import sys
import os
import numpy as np
import pandas as pd
  
class SimilariyCheck(object):
    
    def __init__(self): 
        self.root_path = './'
        self.query = 'qlearningAgents.py'
    
    def read_file(self, filename):  
        try:
            print('OPENING ', filename)
            with open(filename, 'rb') as f:
                data = f.read().decode()
            return data 
        except IOError:
            print("Error opening or reading input file: ", filename)
            sys.exit()
      
    def get_words_from_line_list(self, text):  
        translation_table = str.maketrans(string.punctuation+string.ascii_uppercase,
                                         " " * len(string.punctuation) + string.ascii_lowercase)
        text = text.translate(translation_table)
        word_list = text.split() 
        return word_list 
       
    def count_frequency(self, word_list):  
        D = {} 
        for new_word in word_list: 
            if new_word in D:
                D[new_word] = D[new_word] + 1 
            else:
                D[new_word] = 1  
        return D
       
    def word_frequencies_for_file(self, filename):  
        line_list = self.read_file(filename)
        word_list = self.get_words_from_line_list(line_list)
        freq_mapping = self.count_frequency(word_list) 
        # print("File", filename, ":", )
        # print(len(line_list), "lines, ", )
        # print(len(word_list), "words, ", )
        # print(len(freq_mapping), "distinct words") 
        return filename, len(line_list), len(word_list), len(freq_mapping), freq_mapping
 
    def dotProduct(self, D1, D2): 
        Sum = 0.0 
        for key in D1: 
            if key in D2:
                Sum += (D1[key] * D2[key]) 
        return Sum
       
    def vector_angle(self, D1, D2): 
        numerator = self.dotProduct(D1, D2)
        denominator = math.sqrt(self.dotProduct(D1, D1) * self.dotProduct(D2, D2))
        return math.acos(numerator / denominator)
       
    def document_similarity(self, filename_1, filename_2):  
        filename1, num_lines1, num_words1, num_distinct_words1, sorted_word_list_1 = self.word_frequencies_for_file(filename_1)
        filename2, num_lines2, num_words2, num_distinct_words2, sorted_word_list_2 = self.word_frequencies_for_file(filename_2)
        distance = self.vector_angle(sorted_word_list_1, sorted_word_list_2)
        # print("The distance between the documents is: % 0.6f (radians)"% distance)
        return distance
    
    def get_file_path(self): 
        get_file_paths = {}
        for root, dirs, files in os.walk(self.root_path):
            for file in files: 
                if self.query in file:
                    file_path = os.path.join(root, file)
                    get_file_path = {}
                    get_file_path['filename'] = file
                    get_file_path['address'] = file_path
                    get_file_paths[file.replace(self.query, '').strip()] = get_file_path
        return get_file_paths
     
    
    def main(self):
        file_paths = self.get_file_path()
        file_list = sorted([x for x in file_paths.keys()])
        
        # CHECK THE SIMILARITY BY LOOPING ALL THE FILES
        save = []
        for x in sorted(file_list): 
            s_save = [x]
            for y in sorted(file_list): 
                path_1 = file_paths[x]['address']
                path_2 = file_paths[y]['address']
                sim = self.document_similarity(path_1, path_2) 
                s_save.append(sim)
            save.append(s_save)
        
        # SAVE THE FILE INTO CSV
        ar_to_save = pd.DataFrame(save)
        file_list.insert(0, 'STUDENT ID')
        ar_to_save.columns = file_list 
        ar_to_save.to_csv(os.path.join(self.root_path, 'similarity_results.csv'))

app = SimilariyCheck()
app.main()