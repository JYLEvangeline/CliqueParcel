from collections import defaultdict
from datasets import load_dataset
from utils_efficiency import *

import argparse
import datasets
import csv
import numpy as np
import openai
import random

pre_defined_added_string= "Return the answer of each question with their numerical itemize.\n " 
pre_defined_added_string = "Return the answer of each question with their numerical itemize. You must return with numerical itemize!! Remember to start the answer of each question with 1. xxx\n 2. xxx\n ... \n "# new dataset gpt 3.5
openai.api_key= "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"
datasets.logging.set_verbosity_error()
datasets.logging.disable_progress_bar()
def parse_arguments():
    parser = argparse.ArgumentParser(description="group efficiency")
    parser.add_argument("--length_of_quries", type = int, default = 8)
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--label", type = str, default = "coarse_label") # coarse_label, fine_label for trec
    parser.add_argument("--length_num", type = int, default = 256)
    parser.add_argument("--dataset", default = "MMLU", choices = ["trec","squad","hotpot_qa", "CSQA", "GSM8K","MATH", "ANLI", "MMLU"])
    # dataset :
    # Comprehension: SQUAD, hotpot_qa
    # MATH inference: GSM8K, MATH
    parser.add_argument("--model", default = "gpt-3.5-turbo", choices = ["gpt-3.5-turbo", "gpt-4"])
    parser.add_argument("--mode", default = "maximum_diff", choices = ["group", "seperate", "full_random", "semantic_sim", "concept_plus_semantic_sim", "avg_length", "seq_length", "maximum_diff", 'random_plus_avg_length'])
    parser.add_argument("--i_th_experiment", type = str, default = "12.query_length")
    parser.add_argument("--resume", type = bool, default = True)
    args = parser.parse_args()
    return args


def group_strings(strings, group_size, added_string = ""):
    numbered_groups = []
    grouped_strings = [strings[i:i+group_size] for i in range(0, len(strings), group_size)]
    for i, group in enumerate(grouped_strings):
        numbered_group = [f"{j+1}. {sentence}" for j, sentence in enumerate(group)]
        numbered_groups.append(numbered_group)
    numbered_groups = [added_string + '\n '.join(numbered_group) for numbered_group in numbered_groups]
    return numbered_groups

def merge_dataset_by_label(dataset, key_name_for_merged_label, key_name_for_text):
    merged_dict = defaultdict(list)
    for text in dataset:
        merged_dict[text[key_name_for_merged_label]].append(text[key_name_for_text])
    return dict(merged_dict)

def get_dataset_prompts(random_seed = 42, dataset_description = {"name":"trec", "label":"fine_label", "length_num": 500}):
    random.seed(random_seed)
    np.random.seed(random_seed)
    if dataset_description["name"] == "trec":
        dataset = load_dataset("trec")
        train_dataset = dataset["test"]
        length_num = dataset_description["length_num"]
        random_sample = random.sample(list(train_dataset), length_num)
        merged_prompts = merge_dataset_by_label(random_sample, dataset_description["label"], "text")
        non_merged_prompts = [r['text'] for r in random_sample]
        answer = []
    if dataset_description['name'] == 'squad':
        dataset = load_dataset("squad")
        train_dataset = dataset["validation"] 
        length_num = dataset_description["length_num"]
        random_sample = random.sample(list(train_dataset), length_num)
        # add coarse label to random_sample
        words = set(['what', 'when', 'where', 'who', 'why', 'how', "which", "whose"])
        dictionary = {'what':0, 'when':1, 'where':2, 'who':3, 'why':4, 'how':5}
        for i, qa in enumerate(random_sample):
            list_of_q = re.sub(r'[^a-zA-Z ]', ' ', qa['question']).lower().split()
            q = set(list_of_q)
            label = list(words.intersection(q))
            # corner case: length of label larger than 1
            if len(label) > 1:
                indices_of_labels = sorted([(list_of_q.index(l),l) for l in label])
                label = [indices_of_labels[0][1]]
            if len(label) == 0:
                label = ["what"]
            label = label[0]
            # combine which + what, whose + who
            if label == "which":
                label = "what"
            if label == "whose":
                label = "who"
            qa['coarse_label'] = dictionary[label]
            # renew with qa
            random_sample[i] = qa
        # add text to random_sample
        random_sample = [{**qa, "text": qa['context'] + qa['question']} for qa in random_sample]
        merged_prompts = merge_dataset_by_label(random_sample, "coarse_label", "text")
        non_merged_prompts = [aa['context'] + aa['question'] for aa in random_sample]
        answer = {qa["text"]: list(set(qa['answers']['text'])) for qa in random_sample}
    if dataset_description['name'] == 'hotpot_qa':
        short_answer = 'Short answer:'
        short_answer = ''
        dataset = load_dataset('hotpot_qa', 'distractor')
        train_dataset = dataset["validation"] 
        length_num = dataset_description["length_num"]
        random_sample_tmp = random.sample(list(train_dataset), length_num)
        # add coarse label to random_sample
        dictionary = {'comparison':0, 'bridge':1}
        # add text to random_sample  jy check
        # q = [random_sample_tmp[0]['context']['sentences'][id] for id in random_sample_tmp[0]['supporting_facts']['sent_id']]

        random_sample = [{'text': short_answer + ' '.join([' '.join(qq) for qq in [item['context']['sentences'][id] for id in item['supporting_facts']['sent_id']]])
                           + item['question'], 
                           'answer': item['answer'], 
                           'coarse_label': dictionary[item['type']]} 
                           for item in random_sample_tmp]
        merged_prompts = merge_dataset_by_label(random_sample, "coarse_label", "text")
        non_merged_prompts = [aa['text'] for aa in random_sample]
        answer = {qa["text"]: qa['answer'] for qa in random_sample}
    if dataset_description['name'] == 'CSQA':
        dataset = load_dataset('commonsense_qa')
        train_dataset = dataset["validation"] 
        length_num = dataset_description["length_num"]
        random_sample_tmp = random.sample(list(train_dataset), length_num)
        # no coarse label
        dictionary = {'NA':0}
        # add text to random_sample
        # q = [for random_sample_i in random_sample_tmp]
        random_sample = [{'text': random_sample_i['question'] + ' Select your answer from following options. The format should be "The answer is xxx"\n' + 
                          '\n'.join([f'{label}: {text}' for 
                                     label, text in zip(random_sample_i['choices']['label'], random_sample_i['choices']['text'])]) , 
                           'answer': [random_sample_i['choices']['text'][random_sample_i['choices']['label'].index(random_sample_i['answerKey'])]],  # ['relaxation']
                           'coarse_label': 0} 
                           for random_sample_i in random_sample_tmp]
        merged_prompts = merge_dataset_by_label(random_sample, "coarse_label", "text")
        non_merged_prompts = [aa['text'] for aa in random_sample]
        answer = {qa["text"]: qa['answer'] for qa in random_sample}
    if dataset_description['name'] == 'GSM8K':
        dataset = load_dataset('gsm8k','main')
        train_dataset = dataset["test"] 
        length_num = dataset_description["length_num"]
        random_sample_tmp = random.sample(list(train_dataset), length_num)
        # no coarse label
        dictionary = {'NA':0}
        # add text to random_sample
        # q = [for random_sample_i in random_sample_tmp]
        random_sample = [{'text': random_sample_i['question'] + ' '  + 'The answer should start with "The answer is xxx"', 
                           'answer': random_sample_i['answer'].split("\n####")[-1].strip(), 
                           'coarse_label': 0} 
                           for random_sample_i in random_sample_tmp]
        merged_prompts = merge_dataset_by_label(random_sample, "coarse_label", "text")
        non_merged_prompts = [aa['text'] for aa in random_sample]
        answer = {qa["text"]: qa['answer'] for qa in random_sample}
    if dataset_description['name'] == 'MATH':
        dataset = load_dataset('math_dataset', 'arithmetic__mixed')
        train_dataset = dataset["test"] 
        length_num = dataset_description["length_num"]
        random_sample_tmp = random.sample(list(train_dataset), length_num)
        # no coarse label
        dictionary = {'NA':0}
        # add text to random_sample
        # q = [for random_sample_i in random_sample_tmp]
        random_sample = [{'text': random_sample_i['question'] + 
                          ' The answer should start with "The answer is xxx"\n', 
                           'answer': [str(round(eval(random_sample_i['answer'].strip("b'\\n")), round_num)) for round_num in [0,1,2,3]], 
                           'coarse_label': 0} 
                           for random_sample_i in random_sample_tmp]
        merged_prompts = merge_dataset_by_label(random_sample, "coarse_label", "text")
        non_merged_prompts = [aa['text'] for aa in random_sample]
        answer = {qa["text"]: qa['answer'] for qa in random_sample}
    if dataset_description['name'] == 'ANLI':
        dataset = load_dataset('anli')
        train_dataset = dataset["test_r1"] 
        length_num = dataset_description["length_num"]
        random_sample_tmp = random.sample(list(train_dataset), length_num)
        # no coarse label
        dictionary = {'NA':0}
        # add text to random_sample
        # q = [for random_sample_i in random_sample_tmp]
        answer_dict = {0: 'entailment', 1: 'neural', 2: 'contradiction'}
        random_sample = [{'text': f'Suppose the premise is {random_sample_i["premise"]}. The hypothesis is {random_sample_i["hypothesis"]} What is the relationship between hypothesis and premise? Contradiction, entailment or neural?', 
                           'answer': answer_dict[random_sample_i['label']], 
                           'coarse_label': 0} 
                           for random_sample_i in random_sample_tmp]
        merged_prompts = merge_dataset_by_label(random_sample, "coarse_label", "text")
        non_merged_prompts = [aa['text'] for aa in random_sample]
        answer = {qa["text"]: qa['answer'] for qa in random_sample}
    if dataset_description['name'] == 'MMLU':
        dataset = load_dataset("lukaemon/mmlu","elementary_mathematics", split = "test")
        train_dataset = dataset
        length_num = dataset_description["length_num"]
        random_sample_tmp = random.sample(list(train_dataset), length_num)
        # no coarse label
        dictionary = {'NA':0}
        choices = ['A','B','C','D']
        # add text to random_sample
        # q = [for random_sample_i in random_sample_tmp]
        random_sample = [{'text': random_sample_i['input'] + ' Select your answer from following options: ' + '; '.join([f'{choice}: {random_sample_i[choice]}' for choice in choices]) + 'The answer should start with "The answer is xxx"\n', 
                           'answer': random_sample_i[random_sample_i['target']], 
                           'coarse_label': 0} 
                           for random_sample_i in random_sample_tmp]
        merged_prompts = merge_dataset_by_label(random_sample, "coarse_label", "text")
        non_merged_prompts = [aa['text'] for aa in random_sample]
        answer = {qa["text"]: qa['answer'] for qa in random_sample}

    return merged_prompts, non_merged_prompts, answer


def get_time_for_a_list_of_sentences(args):
    """ get the running time for a list of sentences

    Args:
        args (args): the args fo this file
        sentences ([str]): a list of a sentences(in label, seperate and semantic_sim)
        length_of_quries (int, optional): The length of group size when we group prompts together. Defaults to 5.
        key (str, optional): The group name. Defaults to "0". Only availabel in label, seperate and semantic_sim mode.

    Returns:
        _type_: _description_
    """
    dic_length_of_quires = {'trec': 64, 'squad': 16, 'hotpot_qa': 8, 'CSQA': 32, 'GSM8K': 16, 'MATH': 32, 'ANLI': 4, 'MMLU': 16}
    label = args.label
    if args.dataset == "trec":
        dataset_description = {"name":"trec", "label":label, "length_num": args.length_num}
        args.length_of_quries = 64
    elif args.dataset == 'squad':
        dataset_description = {"name":"squad", "length_num": args.length_num}
        args.length_of_quries = 16
    elif args.dataset == 'hotpot_qa':
        dataset_description = {"name":"hotpot_qa", "length_num": args.length_num}
        args.length_of_quries = 4
    elif args.dataset == 'CSQA':
        dataset_description = {"name":"CSQA", "length_num": args.length_num}
        args.length_of_quries = 32
    elif args.dataset == 'GSM8K':
        dataset_description = {"name":"GSM8K", "length_num": args.length_num}
        args.length_of_quries = 16
    elif args.dataset == 'MATH':
        dataset_description = {"name":"MATH", "length_num": args.length_num}
        args.length_of_quries = 32
    elif args.dataset == 'ANLI':
        dataset_description = {"name":"ANLI", "length_num": args.length_num}
        args.length_of_quries = 4
    elif args.dataset == "MMLU":
        dataset_description = {"name":"MMLU", "length_num": args.length_num}
        args.length_of_quries = 16
    
    args.length_of_quries = dic_length_of_quires[args.dataset]
    # jy: move later
    args_file_name = '/'.join([str(args.length_num), args.model, args.dataset, args.label, args.i_th_experiment + str(args.length_of_quries)])
    if not os.path.exists("efficiency_res/" + args_file_name):
        os.makedirs("efficiency_res/" + args_file_name)
        print("efficiency_res/" + args_file_name)
    
    exception_or_not = False
    
    length_of_quries = args.length_of_quries
    merged_prompts, non_merged_prompts, answer = get_dataset_prompts(dataset_description = dataset_description)
    total_time = []
    if args.mode == 'seperate':
        # process them one by one
        for key, sentences in merged_prompts.items():
            file_seperate_name = "efficiency_res/" + args_file_name + "/seperate" + str(key) + ".txt"
            if args.resume == True:
                check_point_name = file_seperate_name[:-4] + "_checkpoint.pkl"
                check_point = load_checkpoint(check_point_name)
                # if there doesn't exist a check_point, then we could skip this part
                if check_point == None:
                    continue
            total_time_seperate, exception_or_not = run_prompt(sentences, args.model, file_seperate_name, resume = args.resume)
            total_time.append(total_time_seperate)
    if args.mode == 'group':
        # process them with a concept group
        for key, sentences in merged_prompts.items():
            file_groups_name = "efficiency_res/" + args_file_name + "/group" + str(key) + ".txt"
            if args.resume == True:
                check_point_name = file_groups_name[:-4] + "_checkpoint.pkl"
                check_point = load_checkpoint(check_point_name)
                # if there doesn't exist a check_point, then we could skip this part
                if check_point == None:
                    continue
            grouped_sentences = group_strings(sentences, length_of_quries, added_string= pre_defined_added_string)
            total_time_group, exception_or_not  = run_prompt(grouped_sentences, args.model, file_groups_name)
            total_time.append(total_time_group)

    if args.mode == "random":
        # process them with a random group
        file_random_name = "efficiency_res/" + args_file_name + "/random" + ".txt"
        if args.resume == True:
            check_point_name = file_random_name[:-4] + "_checkpoint.pkl"
            check_point = load_checkpoint(check_point_name)
            # if there doesn't exist a check_point, then we could skip this part
            if check_point == None:
                return 0, False
        grouped_sentences = group_strings(non_merged_prompts, length_of_quries, added_string= pre_defined_added_string)
        
        total_time_random, exception_or_not = run_prompt(grouped_sentences, args.model, file_random_name)
        total_time.append(total_time_random)

    if args.mode == "full_random":
        # process them with a random group
        file_random_name = "efficiency_res/" + args_file_name + "/full_random" + ".txt"
        if args.resume == True:
            check_point_name = file_random_name[:-4] + "_checkpoint.pkl"
            check_point = load_checkpoint(check_point_name)
            # if there doesn't exist a check_point, then we could skip this part
            if check_point == None:
                return 0, False
        full_random_prompts = random.sample(non_merged_prompts, len(non_merged_prompts))
        grouped_sentences = group_strings(full_random_prompts, length_of_quries, added_string= pre_defined_added_string)
        
        total_time_random, exception_or_not = run_prompt(grouped_sentences, args.model, file_random_name)
        total_time.append(total_time_random)

    if args.mode == "semantic_sim":
        file_random_name = "efficiency_res/" + args_file_name + "/semantic_sim" + ".txt"
        # process them with a semantic_simlar group
        if args.resume == True:
            check_point_name = file_random_name[:-4] + "_checkpoint.pkl"
            check_point = load_checkpoint(check_point_name)
            # if there doesn't exist a check_point, then we could skip this part
            if check_point == None:
                return 0, False
        sentences = cluster_sentences(non_merged_prompts, cluster_size = args.length_of_quries)
        grouped_sentences = group_strings(sentences, length_of_quries, added_string= pre_defined_added_string)
        
        total_time_random, exception_or_not = run_prompt(grouped_sentences, args.model, file_random_name)
        total_time.append(total_time_random)

    if args.mode == "concept_plus_semantic_sim":
        for key, sentences in merged_prompts.items():
            print(key)
            file_groups_name = "efficiency_res/" + args_file_name + "/concept_plus_semantic_sim" + str(key) + ".txt"
            if args.resume == True:
                check_point_name = file_groups_name[:-4] + "_checkpoint.pkl"
                check_point = load_checkpoint(check_point_name)
                # if there doesn't exist a check_point, then we could skip this part
                if check_point == None:
                    continue
            # add group by semantic_sim
            try:
                sentences = cluster_sentences(sentences, cluster_size = args.length_of_quries)
            except:
                # not enough for clustering
                sentences = sentences
            grouped_sentences = group_strings(sentences, length_of_quries, added_string= pre_defined_added_string)
            total_time_group, exception_or_not  = run_prompt(grouped_sentences, args.model, file_groups_name)
            total_time.append(total_time_group)
    if args.mode == "avg_length":
        file_random_name = "efficiency_res/" + args_file_name + "/avg_length" + ".txt"
        # group by average length
        if args.resume == True:
            check_point_name = file_random_name[:-4] + "_checkpoint.pkl"
            check_point = load_checkpoint(check_point_name)
            # if there doesn't exist a check_point, then we could skip this part
            if check_point == None:
                return 0, False
        non_merged_prompts_length = [[len(prompts.split()), prompts] for prompts in non_merged_prompts]
        sorted_non_merged_prompts_length = sorted(non_merged_prompts_length, key=lambda x: x[0])
        sorted_non_merged_prompts = [prompts[1] for prompts in sorted_non_merged_prompts_length]
        reordered_non_merged_prompts = custom_reorder_list(sorted_non_merged_prompts)
        grouped_sentences = group_strings(reordered_non_merged_prompts, length_of_quries, added_string= pre_defined_added_string)
        file_random_name = "efficiency_res/" + args_file_name + "/avg_length" + ".txt"
        total_time_random, exception_or_not = run_prompt(grouped_sentences, args.model, file_random_name, resume = args.resume)
        total_time.append(total_time_random)
    if args.mode == "seq_length":
        # group by average length
        file_random_name = "efficiency_res/" + args_file_name + "/seq_length" + ".txt"
        if args.resume == True:
            check_point_name = file_random_name[:-4] + "_checkpoint.pkl"
            check_point = load_checkpoint(check_point_name)
            # if there doesn't exist a check_point, then we could skip this part
            if check_point == None:
                return 0, False
        non_merged_prompts_length = [[len(prompts.split()), prompts] for prompts in non_merged_prompts]
        sorted_non_merged_prompts_length = sorted(non_merged_prompts_length, key=lambda x: x[0])
        sorted_non_merged_prompts = [prompts[1] for prompts in sorted_non_merged_prompts_length]
        grouped_sentences = group_strings(sorted_non_merged_prompts, length_of_quries, added_string= pre_defined_added_string)
        file_random_name = "efficiency_res/" + args_file_name + "/seq_length" + ".txt"
        total_time_random, exception_or_not = run_prompt(grouped_sentences, args.model, file_random_name, resume = args.resume)
        total_time.append(total_time_random)
    if args.mode == "maximum_diff":
        # group by maximum difference
        file_random_name = "efficiency_res/" + args_file_name + "/maximum_diff" + ".txt"
        if args.resume == True:
            check_point_name = file_random_name[:-4] + "_checkpoint.pkl"
            check_point = load_checkpoint(check_point_name)
            # if there doesn't exist a check_point, then we could skip this part
            if check_point == None:
                return 0, False
        sentences = cluster_sentences(non_merged_prompts, cluster_size = args.length_of_quries, minimize = False)
        grouped_sentences = group_strings(sentences, length_of_quries, added_string= pre_defined_added_string)
        total_time_random, exception_or_not = run_prompt(grouped_sentences, args.model, file_random_name, resume = args.resume)
        total_time.append(total_time_random)
    if args.mode == "random_plus_avg_length":
        # group by maximum difference
        file_random_name = "efficiency_res/" + args_file_name + "/random_plus_avg_length" + ".txt"
        if args.resume == True:
            check_point_name = file_random_name[:-4] + "_checkpoint.pkl"
            check_point = load_checkpoint(check_point_name)
            # if there doesn't exist a check_point, then we could skip this part
            if check_point == None:
                return 0, False
        random_num = 128
        random_non_merged_prompts = [non_merged_prompts[i:i + random_num] for i in range(0, len(non_merged_prompts), random_num)]
        random_non_merged_prompts_length = [[[len(prompt.split()),prompt] for prompt in prompts] for prompts in random_non_merged_prompts]
        sorted_random_non_merged_prompts_length = [sorted(inner_list, key=lambda x: x[0]) for inner_list in random_non_merged_prompts_length]

        sorted_random_non_merged_prompts = [[prompt[1] for prompt in prompts] for prompts in sorted_random_non_merged_prompts_length]
        reordered_random_non_merged_prompts = [custom_reorder_list(l) for l in sorted_random_non_merged_prompts]
        reordered_random_non_merged_prompts = [item for sublist in reordered_random_non_merged_prompts for item in sublist]
        grouped_sentences = group_strings(reordered_random_non_merged_prompts, length_of_quries, added_string= pre_defined_added_string)
        total_time_random, exception_or_not = run_prompt(grouped_sentences, args.model, file_random_name, resume = args.resume)
        total_time.append(total_time_random)
    return total_time, exception_or_not

def main():
    args = parse_arguments()
    name = 'pickle/' + '-'.join([str(val1)+ ":" +str(val2) for val1, val2 in list(vars(args).items())])
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    total_time, exception_or_not = get_time_for_a_list_of_sentences(args)
    print(exception_or_not)
    res = [total_time, args]
    store_variable_with_pickle(name, res)
    
if __name__ == '__main__':
    main()
