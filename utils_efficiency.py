import itertools
import openai
import pickle
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from checkpoint import *
from g4f import Provider

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertModel, BertTokenizer

model_name = 'bert-base-uncased'  # Example model name
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def default_dic():
    return defaultdict(int)



def default_array():
    return np.zeros((1,2))

# store and load pickle
def store_variable_with_pickle(file_path, variable):
    with open(file_path, 'wb') as file:
        pickle.dump(variable, file)

def load_variable_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        variable = pickle.load(file)
    return variable

def get_sentence_embeddings(sentences):
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_inputs)

    embeddings = model_output.last_hidden_state[:, 0, :].numpy()  # Extract CLS token embeddings
    return embeddings

def cluster_sentences(sentences, cluster_size=5, minimize = True):
    embeddings = get_sentence_embeddings(sentences)
    n_clusters = len(sentences) // cluster_size
    if minimize:
        # using KMeans to minimize the distance
        similarity_matrix = cosine_similarity(embeddings)

        # Using k-means clustering to group sentences into clusters
        kmeans = KMeans(n_clusters = n_clusters)
        cluster_labels = kmeans.fit_predict(similarity_matrix)

        clusters = {}
        for idx, cluster_label in enumerate(cluster_labels):
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(sentences[idx])
        
        return list(itertools.chain.from_iterable(clusters.values()))
    else:
        agg_clustering = AgglomerativeClustering(n_clusters = n_clusters, linkage="complete")
        cluster_assignments = agg_clustering.fit_predict(embeddings)
        clusters = {}
        for idx, cluster_label in enumerate(cluster_assignments):
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(sentences[idx])
        return list(itertools.chain.from_iterable(clusters.values()))

def run_prompt(sentences, model, file_name, resume = False):
    check_point_name = file_name[:-4] + "_checkpoint.pkl"
    check_point = load_checkpoint(check_point_name)
    if check_point == None:
        file_seperate = open(file_name,'w')
        start_i = 0
        total_time = 0
    else:
        file_seperate = open(file_name,'a')
        start_i = check_point[0]
        total_time = check_point[1]
        delete_checkpoint(check_point_name)

    # file_seperate_name = "efficiency_res/" + args_file_name + "/sepearte" + str(key) + ".txt"
    exception_or_not = False
    for i, word in enumerate(sentences):
        if i < start_i:
            continue
        try:
            answer, polish_time = estimate_time(word, model = model)
            # print(answer)
        except:
            exception_or_not = True
            check_point = [i, total_time]
            generate_check_point(check_point_name, check_point)
            break
        
        
        total_time += polish_time
        
        file_seperate.write("WORD:\n" + word + "\n")
        file_seperate.write("ANSWER:\n" + answer + "\n")
        file_seperate.write("TIME:\n" + str(polish_time) + "\n")
        # time.sleep(5)
    if exception_or_not == False:
        file_seperate.write("TOTAL TIME:" + str(total_time))
    file_seperate.close()
    return total_time, exception_or_not

def estimate_time(input, model = "gpt-3.5-turbo"):
    """ Given an input, estimate the running time for run it in relvant model

    Args:
        input (str): The input sentence
        model (str, optional): The model to run. Defaults to "gpt-3.5-turbo". choices = ['g4f', 'gpt-3.5-turbo', 'gpt-4']

    Returns:
        _type_: _description_
    """
    if model == "g4f":
        start_time = time.time()
        response = g4f.ChatCompletion.create(model='gpt-3.5-turbo',provider=g4f.Provider.Liaobots, messages=[
                                        {"role": "user", "content": input}], stream=False)
        end_time = time.time()
    else:
        messages = []
        messages.append({"role":"user","content": input})
        start_time = time.time()
        response=openai.ChatCompletion.create(
            model=model,
            messages= messages
            )
        end_time = time.time()
    reply = response["choices"][0]["message"]["content"]
    return reply, end_time - start_time

def get_flattened_string(splitted_s):
    """ splitted_s is a grouped of questions/answers, flattend it to be a list of strings

    Args:
        splitted_s (str): _description_

    Returns:
        _type_: _description_
    """
    pattern = r'^\d+\. '
    # tmp_string = "START" for corner case "The answer are: 1.xx 2.xx"
    tmp_string, flattend_strings = "START", []
    for ss in splitted_s:
        ss = ss.lstrip()
        if re.match(pattern, ss) and tmp_string != "":
            if tmp_string != "START":
                flattend_strings.append(tmp_string)
            tmp_string = ss
        else:
            if tmp_string != "START":
                tmp_string += ss
    if tmp_string != "":
        flattend_strings.append(tmp_string)
    return flattend_strings

# extract numbers
def remove_numbered_list(string, added_string = "Return the answer of each question with their numerical itemize."):
    # Remove added_string
    string = [s.replace(added_string, "") for s in string]
    # Regular expression pattern to match numbered list formats (e.g., "1. ", "2. ", "3. ", etc.)
    pattern = r'^\d+\. '
    final_strings, flattend_lengths = [], []
    tmp_tester_strings = []
    for i, s in enumerate(string):
        s = s.replace("O\n2","O2")
        s = s.replace("O\n3","O3")
        splitted_s = s.split("\n")
        # if i == 22:
        #     print(1)
        flattend_strings = get_flattened_string(splitted_s)
        # add flattend_strings to final_strings
        tmp_tester_strings.append(flattend_strings)
        len_s = len(flattend_strings)
        flattend_lengths.append(len_s)
        final_strings += flattend_strings
    # pattern = r'\b\d+\.\s'
    # delete all the 1. 2. 3.
    for i, flattend_string in enumerate(final_strings):
        final_strings[i] = re.sub(pattern, '', flattend_string)
    for i, tmp_tester_string in enumerate(tmp_tester_strings):
        for j, s in enumerate(tmp_tester_string):
            tmp_tester_strings[i][j] = re.sub(pattern, '', s)
    return final_strings, tmp_tester_strings


def custom_reorder_list(l):
    result = []
    mid_index = len(l) // 2

    for i in range(mid_index):
        result.append(l[i])
        result.append(l[-(i + 1)])

    # If the length of the input list is odd, append the middle element.
    if len(l) % 2 == 1:
        result.append(l[mid_index])

    return result


# def extract_strings_from_tags(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     word_strings = []
#     answer_strings = []
#     time_strings = []

#     current_tag = None

#     for line in lines:
#         # if "Texas" in line:
#         #     print(1)
#         if line.startswith("WORD"):
#             current_tag = "WORD"
#             text = line.split(":")
#             if len(text) > 1 and text[1][:6] != "Return":
#                 word_strings.append(text[1].replace('\n', ''))
#         elif line.startswith("ANSWER"):
#             current_tag = "ANSWER"
#             text = line.split(":")
#             if len(text) > 1 and len(text[1]) != 0:
#                 answer_strings.append(text[1].replace('\n', ''))
#         elif line.startswith("TIME"):
#             current_tag = "TIME"
#         else:
#             if current_tag == "WORD":
#                 word_strings.append(line.strip())
#             elif current_tag == "ANSWER" and line.strip()!='':
#                 answer_strings.append(line.strip())
#             elif current_tag == "TIME":
#                 time_strings.append(line.strip())        
#                 if len(word_strings) != len(answer_strings):
#                     print(1)
#     return word_strings, answer_strings, time_strings

def get_tag(line):
    if line.startswith("WORD"):
        current_tag = "WORD"
    elif line.startswith("ANSWER"):
        current_tag = "ANSWER"
    elif line.startswith("TIME"):
        current_tag = "TIME"
    return current_tag

def extract_strings_from_tags(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    total_time = 0
    word_strings = []
    answer_strings = []
    time_strings = []

    current_tag = None
    current_string = ""

    for line in lines:
        # If a new tag is encountered, add the current string to the corresponding list and reset the current_string.
        if line.startswith(("WORD", "ANSWER", "TIME")):
            if current_tag == "WORD":
                # corner case in squad
                current_string = current_string.replace("O\n2","O2")
                word_strings.append(current_string.strip())
            elif current_tag == "ANSWER":
                # corner case in squad
                current_string = current_string.replace("O\n2","O2")
                answer_strings.append(current_string.strip())
            elif current_tag == "TIME":
                time_strings.append(float(current_string.strip()))
                if len(word_strings) != len(answer_strings):
                    print(1)
            current_string = ""
        if line.startswith("WORD"):
            current_tag = "WORD"
        elif line.startswith("ANSWER"):
            current_tag = "ANSWER"
        elif line.startswith("TIME"):
            current_tag = "TIME"
        else:
            if line.startswith("TOTAL TIME"):
                total_time = float(line.split(":")[1])
            if current_tag == "WORD":
                current_string += line
            elif current_tag == "ANSWER":
                current_string += line
            elif current_tag == "TIME":
                current_string += line
    return word_strings, answer_strings, time_strings, total_time