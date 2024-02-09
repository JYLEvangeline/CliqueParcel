import argparse
import csv
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import seaborn as sns
from collections import defaultdict
from gpt_efficiency_main import get_dataset_prompts
from itertools import chain
from scipy.stats import linregress
from utils_efficiency import *
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu



def parse_arguments():
    parser = argparse.ArgumentParser(description="group efficiency")
    parser.add_argument("--length_of_quries", type = int, default = 32)
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--label", type = str, default = "coarse_label")
    parser.add_argument("--dataset", default = "squad", choices = ["trec","squad","hotpot_qa"])
    parser.add_argument("--model", type = str, default = "gpt-4")
    parser.add_argument("--i_th_experiment", type = str, default = "12.query_length")
    parser.add_argument("--length_num", type = int, default = 256)
    args = parser.parse_args()
    return args

# final version for draw

# modes = ["seperate", "group", "random", "semantic_sim", "concept_plus_semantic_sim"]
dic_length_of_quires = {'trec': 64, 'squad': 16, 'hotpot_qa': 4}
dic_length_of_quires = {'trec': 64, 'squad': 16, 'hotpot_qa': 8, 'CSQA': 32, 'GSM8K': 16, 'MATH': 32, 'ANLI': 4, 'MMLU': 16} # hotpotqa 8 or 4?
modes =["seperate", "group", "random", "full_random", "semantic_sim", "concept_plus_semantic_sim", "avg_length", "maximum_diff", 'random_plus_avg_length'] # remove semantic sim debug
tmp_modes = "CC & RC & SSC  & CpSC & ALC    & MDC & SpALC".split('&')
tmp_modes = [m.strip() for m in tmp_modes]
seperate_index = modes.index('seperate')
key_for_label = {"coarse_label": 6, "fine_label": 50}
args = parse_arguments()


def trec_pattern(input_string):
    # input_string = '"34" | "40", "Celcius" | "F" | "Fahrenheit" | "C"'

    values = []  # To store the final result
    current_value = []  # To store the current inner list
    current_string = ""
    inside_quotes = False  # To track whether we are inside double quotes

    for char in input_string:
        if char == '"':
            inside_quotes = not inside_quotes
        elif char == ',' and not inside_quotes:
            if len(current_string) != 0:
                current_value.append(current_string)
                current_string = ""
            values.append(current_value)
            current_value = []
        elif char == '|' and not inside_quotes:
            current_value.append(current_string)
            current_string = ""
        elif inside_quotes:
            current_string += char
    if current_string:
        current_value.append(current_string)
    if current_value:
        values.append(current_value)

    # Remove double quotes from each value in the result
    # result = [[''.join(value) for value in values]]

    return values

def error_bar():
    data = load_variable_from_pickle("length_of_quries" + str(5) + ".pickle")[0]
    keys = list(data.keys())
    values = [item[0]*100 for item in data.values()]
    errors = [item[1]*100 for item in data.values()]

    # Create the plot with error bars
    plt.errorbar(keys, values, yerr=errors, fmt='o')
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.title('Individual vs Group')


    # Show the plot
    plt.savefig("Figure/length_of_quries" + str(5) + ".eps")
    plt.close()

def ratio_bar():
    data = load_variable_from_pickle("length_of_quries" + str(5) + ".pickle")[0]
    keys = list(data.keys())
    # get ratio
    values1 = [item[0]/item[1] for item in data.values()]
    # get groups
    merged_prompts, non_merged_prompts = get_dataset_prompts()
    data2 = {}
    for key, val in merged_prompts.items():
        data2[key] = math.ceil(len(val)/5)
    data.update(data2)
    values2 = list(data.values())

    # Create the plot with error bars
    bar_width = 0.35

    # Calculate the positions of the bars on the x-axis
    x = np.arange(len(keys))

    # Create the plot
    plt.bar(x, values1, width=bar_width, label='Ratio')
    plt.bar(x + bar_width, values2, width=bar_width, label='Group Numbers')

    # Add labels, title, and legend
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.title('Dictionary Plot with Two Bars per Key')
    plt.xticks(x + bar_width / 2, keys)
    plt.legend()

    # Show the plot
    plt.xlabel('Keys')
    plt.ylabel('Ratio')
    plt.title('The ratio of individual divided by group')


    # Show the plot
    plt.savefig("Figure/ratio_length_of_quries" + str(5) + ".eps")
    plt.close()

def heatmap(correlation_matrix, file_name):
    tmp_modes = "CC &  RC & SSC  & CpSC & ALC    & MDC & SpALC".split('&')
    tmp_modes = [m.strip() for m in tmp_modes]
    file_name = file_name.replace(".","_")
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5, xticklabels = tmp_modes, yticklabels = tmp_modes, fmt=".3f")
    plt.title('Correlation Matrix Heatmap')
    plt.savefig("Figure/" + file_name + ".eps")
    plt.close()

def scatterplot(categories, data, labels, file_name, xlabel = "Methods", ylabel = "Accuracy", show_y = True, xticks = True, y_lim = True, regression = True):
    """ scatter plot

    Args:
        data (dict): _description_
        file_name (str): _description_
    """
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'brown']

    file_name = file_name.replace(".","_")
    n = len(data)  # Number of datasets

    # Create a bar plot
    for i, i_th_data in enumerate(data):
        if len(i_th_data) == 1:
            x = range(1, len(i_th_data[0])+1)
            y = i_th_data[0]
        elif len(i_th_data) == 2:
            x = i_th_data[0]
            y = i_th_data[1]
        plt.scatter(x, y, label= labels[i], color = colors[i])
        if regression == True:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            regression_line = slope * x + intercept
            plt.plot(x, regression_line, color=  colors[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(i_th_data) == 1 and xticks == False:
        plt.xticks(range(len(i_th_data[0])), modes)
    elif isinstance(xticks, list):
        plt.xticks(range(len(i_th_data[0])), xticks)


    # set ylim
    if y_lim:
        max_value = np.max(np.concatenate(data))
        # Set y-axis limits based on the maximum value
        plt.ylim(0, max_value * 1.2)
    plt.legend()
    plt.savefig("Figure/" + file_name + ".eps")
    plt.close()


def lineplot(categories, data, labels, file_name, xlabel = "Methods", ylabel = "Accuracy", show_y = True, xticks = True, y_lim = True, marker = False):
    """ plot a line

    Args:
        data (dict): _description_
        file_name (str): _description_
    """
    file_name = file_name.replace(".","_")
    n = len(data)  # Number of datasets

    plt.style.use('ggplot')
    bar_width = 0.8/n # Width of each bar
    bar_positions = np.arange(len(categories))  # X positions of bars

    # Create a bar plot
    for i, i_th_data in enumerate(data):
        
        # if len(i_th_data) == 1:
        #     x = range(len(i_th_data[0]))
        #     y = i_th_data[0]
        #     plt.plot(i_th_data[0], label= labels[i])
        # elif len(i_th_data) == 2:
        #     plt.plot(i_th_data[0], i_th_data[1], label= labels[i])

        if len(i_th_data) == 1:
            x = range(len(i_th_data[0]))
            y = i_th_data[0]
        elif len(i_th_data) == 2:
            x = i_th_data[0]
            y = i_th_data[1]
        if len(i_th_data) == 1:
            x = range(len(i_th_data[0]))
            y = i_th_data[0]
        plt.plot(x, y, label= labels[i], linewidth=3)
        if marker == True:
            max_x = x[np.argmax(y)]
            max_y = max(y)
            # Annotate the maximum point with an "X" marker
            plt.scatter(max_x, max_y, color='red', marker='x', s=100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(i_th_data) == 1 and xticks == False:
        plt.xticks(range(len(i_th_data[0])), modes)
    elif isinstance(xticks, list):
        plt.xticks(range(len(i_th_data[0])), xticks)


    # set ylim
    if y_lim:
        max_value = np.max(np.concatenate(data))
        # Set y-axis limits based on the maximum value
        plt.ylim(0, max_value * 1.2)
    plt.legend()
    plt.savefig("Figure/" + file_name + ".eps")
    plt.close()

def barplot(categories, data, labels, file_name, xlabel = "Methods", ylabel = "Accuracy", show_y = True):
    """ plot a bar

    Args:
        data (dict): _description_
        file_name (str): _description_
    """
    file_name = file_name.replace(".","_")
    n = len(data)  # Number of datasets


    bar_width = 0.8/n # Width of each bar
    bar_positions = np.arange(len(categories))  # X positions of bars

    # Create a bar plot
    for i in range(n):
        plt.bar(bar_positions + i * bar_width, data[i], width=bar_width, label= labels[i])
        if show_y == True:
            for j, value in enumerate(data[i]):
                plt.text(bar_positions[j] + i * bar_width, value * 1.02, '{:.2f}'.format(value), ha='center', va='bottom')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(bar_positions + (n-1) * bar_width / 2, categories)
    # set ylim
    max_value = np.max(np.concatenate(data))
    # Set y-axis limits based on the maximum value
    plt.ylim(0, max_value * 1.2)
    plt.legend()
    plt.savefig("Figure/" + file_name + ".eps")
    plt.close()

def sum_up():
    data = load_variable_from_pickle("length_of_quries" + str(5) + ".pickle")[0]
    keys = list(data.keys())
    values = [item[0]*100 for item in data.values()]
    errors = [item[1]*100 for item in data.values()]
    data = load_variable_from_pickle("random" + str(5) + ".pickle")
    print(sum(values), sum(errors), data)

# error_bar()
# ratio_bar()
# sum_up()
def compare_distance(answer_dicts):
    res = np.zeros((len(modes), len(modes)))
    length = 0
    for key in answer_dicts["group"]:
        try:
            answer_key = [answer_dicts[mode][key] for mode in modes]
            length += 1
        except:
            continue
        embeddings = get_sentence_embeddings(answer_key)
        similarity_matrix = cosine_similarity(embeddings)
        res += similarity_matrix
    return res/length

def get_ground_truth(args):
    if args.dataset == "trec":
        dataset_description = {"name":"trec", "label":args.label, "length_num": args.length_num}
    elif args.dataset == 'squad':
        dataset_description = {"name":"squad", "length_num": args.length_num}
    elif args.dataset == 'hotpot_qa':
        dataset_description = {"name":"hotpot_qa", "length_num": args.length_num}
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
    _, _, answer = get_dataset_prompts(dataset_description = dataset_description)
    return answer
    

def get_completeness1(answer_dicts, accuracy):
    i = 0
    res = []
    for key in answer_dicts["group"]:
        try:
            # 
            answer_key = [answer_dicts[mode][key] for mode in modes]
            len_answer_key = np.array([len(a.split()) for a in answer_key])
            len_answer_key -= len_answer_key[seperate_index]
            len_answer_key = np.array([abs(a) for a in len_answer_key])
            
        except:
            continue
        embeddings = get_sentence_embeddings(answer_key)
        similarity_matrix = np.array(cosine_similarity(embeddings))[seperate_index]
        res_i = similarity_matrix / (len_answer_key+1)
        res.append(res_i)
    return sum(np.array(res)) * accuracy/len(res)

def get_completeness(answer_dicts, accuracy):
    res = []
    for key in answer_dicts["group"]:
        try:
            # 
            answer_key = [answer_dicts[mode][key] for mode in modes]
            candidates = [tmp.split() for tmp in answer_key]
            reference = [candidates[seperate_index]]
            bleu_scores = np.array([sentence_bleu(reference, candidate) for candidate in candidates])
            r = Rouge()
            rouge_scores = [r.get_scores(candidate, answer_key[seperate_index]) for candidate in answer_key]
            rouge_scores = np.array([rouge_score[0]['rouge-l']['r'] for rouge_score in rouge_scores])
            # b/r_scores = 
            acc = [accuracy[mode][key] for mode in modes]

        except:
            continue
        embeddings = get_sentence_embeddings(answer_key)
        similarity_matrix = np.array(cosine_similarity(embeddings))[seperate_index]
        res_i = similarity_matrix * (bleu_scores + rouge_scores) * acc
        res.append(res_i)
    return [sum(np.array(res))/len(res)]

def get_accuracy(ground_truth, answer_dicts, trec_or_not = False, dataset = None):
    """ get the accuracy of answers

    Args:
        ground_truth (_type_): The ground truth answer of question
        answer_dicts (_type_): The answers generated
        trec_or_not (bool, optional): Whether it is TREC dataset, if it is, read the manually labeling csv file.

    Returns:
        _type_: _description_
    """
    details = defaultdict(default_dic)
    res = defaultdict(int)
    total = defaultdict(int)
    if trec_or_not:
        ground_truth =  {}
        with open('output_trec.csv', 'r') as csvfile:
            # Create a CSV reader
            csvreader = csv.reader(csvfile)
            # Iterate through the rows and print them
            for row in csvreader:
                ground_truth[row[0]] = row[1]
        for question, answer_ground_truth in ground_truth.items():
            answer_ground_truth_list = trec_pattern(answer_ground_truth)
            for mode, qapairs_of_cur_mode in answer_dicts.items():
                if question in qapairs_of_cur_mode or question + ' ' in qapairs_of_cur_mode:
                    total[mode] += 1
                    # get the answer of this question for current mode
                    try:
                        answer = qapairs_of_cur_mode[question]
                    except:
                        answer = qapairs_of_cur_mode[question + ' ']
                    # check whether answer includes the ground truth
                    correct = True
                    for answer_ground_truth_item in answer_ground_truth_list:
                        if any([a.lower() in answer.lower() for a in answer_ground_truth_item]) == False:
                            correct = False
                            break
                    q = question if question in qapairs_of_cur_mode else question + ' '
                    if correct:
                        res[mode] += 1
                        # for details
                        details[mode][q] = 1
                    else:
                        details[mode][q] = 0
    
    for i, question in enumerate(ground_truth):
        answer_ground_truth = ground_truth[question]
        if dataset in ["trec","squad","hotpot_qa"]:
            question = question.replace("O\n2","O2")
            question = question.replace("O\n3","O3")
            question = question.strip()
            pattern = r'\b\d+\.\s'
        else:
            question = question.replace("\n","")
        # question = re.sub(pattern, '', question.lstrip())
        for mode, qapairs_of_cur_mode in answer_dicts.items():
            if question in qapairs_of_cur_mode or question + ' ' in qapairs_of_cur_mode:
                total[mode] += 1
                # get the answer of this question for current mode
                try:
                    answer = qapairs_of_cur_mode[question]
                except:
                    try:
                        answer = qapairs_of_cur_mode[question + ' ']
                    except:
                        answer = qapairs_of_cur_mode[question + '\n']
                # check whether answer includes the ground truth
                q = question if question in qapairs_of_cur_mode else question + ' '
                for possible_answer in answer_ground_truth:
                    if possible_answer.lower() in answer.lower():
                        res[mode] += 1
                        # for details
                        details[mode][q] = 1
                        break
                
            else:
                print(1)

    return {key: res[key] / total.get(key, 1) for key in res}, details

# def get_sentence_embeddings(sentences):
#     embeddings = []
#     for sentence in sentences:
#         embedding = model.encode(sentence)

def analyze_res(args, i_th_experiment, added_string = "Return the answer of each question with their numerical itemize. You must return with numerical itemize!!\n "):
    pre_time_result = load_variable_from_pickle('length_' + str(args.length_of_quries)+'_pre_time_caculator.pickle')
    if args.dataset == 'hotpot_qa':
        if args.model == "gpt-3.5-turbo":
            args.length_of_quries = 4
        else:
            args.length_of_quries = 8
    args_file_name = '/'.join([str(args.length_num), args.model, args.dataset, args.label, i_th_experiment + str(args.length_of_quries)])
    if args.dataset in ['trec','hotpot_qa','squad']:
        keys = key_for_label[args.label]
    else:
        keys = 1
    # parse txt file and get the answers
    answer_dicts = {}
    time_dicts = defaultdict(list)
    total_time_dicts = defaultdict(float)
    length_dicts = defaultdict(default_array)
    for mode in modes:
        answer_dict = {}
        l = [0,0]
        # total_time_dicts[mode] += pre_time_result[args.dataset][args.length_num][mode]
        # # # add bert to group
        # if mode == 'group':
        #     #total_time_dicts[mode] += pre_time_result[args.dataset][args.length_num]['semantic_sim']
        if mode in ['group', 'seperate', 'concept_plus_semantic_sim']:
            
            for key in range(keys):
                file_path = "efficiency_res/" + args_file_name + "/" + mode + str(key) + ".txt"
                try:
                    word_strings, answer_strings, time, total_time = extract_strings_from_tags(file_path)
                    l[0] += sum([len(a.split()) for a in answer_strings])
                    l[1] += len(answer_strings)
                except:
                    continue
                # get token_length results
                length_dicts[mode] += np.array([sum([len(word_string.split()) for word_string in word_strings]), sum([len(answer_string.split()) for answer_string in answer_strings])])
                time_dicts[mode] += time
                total_time_dicts[mode] += total_time
                # extra operations for grouped prompts
                if mode == 'group' or mode == 'concept_plus_semantic_sim':
                    # remove number and flatten
                    word_strings, tester_word_strings = remove_numbered_list(word_strings, added_string = added_string) # jy: waiting to add
                    answer_strings, tester_answer_strings = remove_numbered_list(answer_strings, length_of_quries = args.length_of_quries)     
                if mode == 'seperate':
                    word_strings = [w.replace("\n","") for w in word_strings]           
                    answer_strings = [a.replace("\n","") for a in answer_strings]
                # add word and answer to dict
                if len(word_strings) != len(answer_strings):
                    word_strings, answer_strings = [], []
                    for t_w, t_a in zip(tester_word_strings, tester_answer_strings):
                        if len(t_w) == len(t_a):
                            word_strings += t_w
                            answer_strings += t_a
                for w, a in zip(word_strings, answer_strings):
                    answer_dict[w.lstrip()] = a.lstrip()
        else:
            file_path = "efficiency_res/" + args_file_name + "/" + mode + ".txt"
            word_strings, answer_strings, _, total_time = extract_strings_from_tags(file_path)
            total_time_dicts[mode] += total_time
            # get token length results
            length_dicts[mode] += np.array([sum([len(word_string.split()) for word_string in word_strings]), sum([len(answer_string.split()) for answer_string in answer_strings])])
            # extra operations for grouped prompts. Random and semantic_sim are both grouped prompts
            # remove number and flatten
            word_strings, tester_word_strings = remove_numbered_list(word_strings, added_string = added_string)
            answer_strings, tester_answer_strings = remove_numbered_list(answer_strings, length_of_quries = args.length_of_quries)
            
            # add word and answer to dict
            if len(word_strings) != len(answer_strings):
                word_strings, answer_strings = [], []
                for t_w, t_a in zip(tester_word_strings, tester_answer_strings):
                    if len(t_w) == len(t_a):
                        word_strings += t_w
                        answer_strings += t_a

            for w, a in zip(word_strings, answer_strings):
                answer_dict[w.lstrip()] = a.lstrip()
        answer_dicts[mode] = answer_dict

    catogories = list(answer_dicts.keys())
    # draw accuracy between methods and ground truth
    if args.dataset == "trec":
        res, details = get_accuracy([], answer_dicts, trec_or_not = True, dataset = args.dataset)
    else:
        ground_truth = get_ground_truth(args)
        res, details = get_accuracy(ground_truth, answer_dicts, dataset = args.dataset)
        
    accuracy = [list(res.values())]
    maximum_index = list(accuracy[0]).index(max(accuracy[0])) #[1:] to remove seperate
    print_acc = [f"{round(num, 4):.4f}" for num in accuracy[0]]
    print_acc[maximum_index] = '\\textbf{' + print_acc[maximum_index] + '}'
    print(args.dataset + ' ' + args.model + '&' + '&'.join(print_acc))

    file_name =  str(args.length_num) + '/accuracy:' + '_'.join([args.model, args.dataset, args.label, i_th_experiment + str(args.length_of_quries)])
    barplot(catogories, accuracy, ['Accuracy'], file_name)
    
    # draw completeness

    file_name = str(args.length_num) + '/completeness:' + '_'.join([args.model, args.dataset, args.label, i_th_experiment + str(args.length_of_quries)])
    completeness = get_completeness(answer_dicts, details)
    maximum_index = list(completeness[0]).index(max(completeness[0][1:]))
    print_com = [f"{round(num, 2):.4f}" for num in completeness[0]]
    print_com[maximum_index] = '\\textbf{' + print_com[maximum_index] + '}'
    print('completeness')
    print(args.dataset + ' ' + args.model + '&' + '&'.join(print_com))

    # return 
    # draw tokens
    file_name = str(args.length_num) + '/token_length:' + '_'.join([args.model, args.dataset, args.label, i_th_experiment + str(args.length_of_quries)])
    # data = [[sum(i[0]), sum(i[1])] for i in length_dicts.values()]
    token_length =  np.array(list(length_dicts.values())).T
    token_length = np.resize(token_length,(2,len(modes)))
    barplot(catogories, token_length, ['Total Prompt Length', 'Total Result Length'], file_name, ylabel = "Token Length",)
    
                                                                    

    # draw price
    prices = {'gpt-4':[0.03, 0.06], 'gpt-3.5-turbo': [0.0015, 0.002]}
    file_name = str(args.length_num) + '/price:' + '_'.join([args.model, args.dataset, args.label, i_th_experiment + str(args.length_of_quries)])
    cur_price = prices[args.model]
    # based on openai policy https://openai.com/pricing
    price = [(token_length[0,:] * cur_price[0] + token_length[1,:] * cur_price[1])]
    barplot(catogories, price, ['Total Price Of Each Model'], file_name, ylabel = "Price($0.001)",)


    # draw running time
    file_name = str(args.length_num) + '/total_time:' + '_'.join([args.model, args.dataset, args.label, i_th_experiment + str(args.length_of_quries)])
    running_time = [list(total_time_dicts.values())]
    barplot(catogories, running_time, ['Running time'], file_name, ylabel = "Running Time",)
    
    # draw efficiency ratio
    file_name = str(args.length_num) + '/efficiency_ratio:' + '_'.join([args.model, args.dataset, args.label, i_th_experiment + str(args.length_of_quries)])
    index_of_seperate = modes.index('seperate')
    accuracy_ratio = np.array(accuracy)/accuracy[0][index_of_seperate]
    
    print("accuracy ratio")
    print_acc_ratio = [f"{round(num, 2):.2f}" for num in accuracy_ratio[0]]
    print('&'.join([f"{round(num, 2):.2f}" for num in accuracy_ratio[0]]))
    price_ratio = np.array(price)/price[0][index_of_seperate]
    running_time_ratio = np.array(running_time)/running_time[0][index_of_seperate]
    efficiency_ratio = 1/(price_ratio * running_time_ratio)
    efficiency_ratio = price_ratio/running_time_ratio

    print("efficiency ratio")
    maximum_index = list(efficiency_ratio[0]).index(max(efficiency_ratio[0]))
    print_eff = [f"{round(num, 2):.4f}" for num in efficiency_ratio[0]]
    print_eff[maximum_index] = '\\textbf{' + print_eff[maximum_index] + '}'
    print(args.dataset + ' ' + args.model + '&' + '&'.join(print_eff))
    lineplot(catogories, [efficiency_ratio], ['Efficiency Ratio'], file_name, ylabel = "Efficiency Ratio")
    
    # completeness + efficiency
    file_name = str(args.length_num) + '/cme:' + '_'.join([args.model, args.dataset, args.label, i_th_experiment + str(args.length_of_quries)])
    cme = completeness * efficiency_ratio
    maximum_index = list(cme[0]).index(max(cme[0][1:]))
    print_cme = [f"{round(num, 2):.2f}" for num in cme[0]]
    print_cme[maximum_index] = '\\textbf{' + print_cme[             maximum_index] + '}'
    print('completeness')
    print(args.dataset + ' ' + args.model + '&' + '&'.join(print_cme))

    # draw distance between methods, too slow so commented, recover in future JY
    similarity_res = compare_distance(answer_dicts)
    file_name = str(args.length_num) + '/heat_map:' + '_'.join([args.model, args.dataset, args.label, i_th_experiment + str(args.length_of_quries)])
    heatmap(similarity_res, file_name)
    
    
    return print_acc, print_eff, print_com

def main():
    import warnings
    import logging

    # Set the log level for the 'transformers' logger to WARNING
    
    warnings.filterwarnings('ignore')
    args = parse_arguments()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    i_th_experiment = args.i_th_experiment
    # pre-defines
    labels = {"trec": ["coarse_label", "fine_label"], "squad": ["coarse_label"]}
    labels = {"trec": ["coarse_label"], "squad": ["coarse_label"], "hotpot_qa": ["coarse_label"]}
    labels = {'trec': ['coarse_label'], 'squad': ['coarse_label'], 'hotpot_qa': ['coarse_label'], 'CSQA': ['coarse_label'], 'GSM8K': ['coarse_label'], 'MATH': ['coarse_label'], 'ANLI': ['coarse_label'], 'MMLU': ['coarse_label']}
    pre_defined_added_string = { "gpt-4" : "Return the answer of each question with their numerical itemize.\n " , "gpt-3.5-turbo": "Return the answer of each question with their numerical itemize. You must return with numerical itemize!! Remember to start the answer of each question with 1. xxx\n 2. xxx\n ... \n "}# new 
    pre_defined_added_string = { "gpt-4" : "Return the answer of each question with their numerical itemize. You must return with numerical itemize!! Remember to start the answer of each question with 1. xxx\n 2. xxx\n ... \n " , "gpt-3.5-turbo": "Return the answer of each question with their numerical itemize. You must return with numerical itemize!! Remember to start the answer of each question with 1. xxx\n 2. xxx\n ... \n "}# new 
    plt.set_loglevel("error")
    experiments = [f'{i}.query_length' for i in range(10, 13)]

    for i_th_experiment in experiments:
    # for dataset in ["trec", "squad", "hotpot_qa"]:
    # for dataset in ["hotpot_qa"]:
    # for dataset in ["CSQA"]:
        all_acc, all_eff, all_cme = {}, {}, {}
        for dataset in ["CSQA", "GSM8K","MATH", "ANLI"]:
            args.dataset = dataset
            args.length_of_quries = dic_length_of_quires[args.dataset]
            # for model in ["gpt-4"]:
            for model in ["gpt-4", "gpt-3.5-turbo"]:
                args.model = model
                for label in labels[args.dataset]:
                    print("\n")
                    print(args.dataset, model, label)
                    args.label = label
                    print_acc, print_eff, print_cme = analyze_res(args, i_th_experiment, added_string = pre_defined_added_string[model])
                    all_acc[dataset + ' ' + model] = print_acc
                    all_eff[dataset + ' ' + model] = print_eff
                    all_cme[dataset + ' ' + model] = print_cme
        csv.writer(open('acc' + i_th_experiment, 'w', newline='')).writerows([all_acc.keys()] + list(zip(*all_acc.values())))
        csv.writer(open('eff' + i_th_experiment, 'w', newline='')).writerows([all_eff.keys()] + list(zip(*all_eff.values())))
        csv.writer(open('cme' + i_th_experiment, 'w', newline='')).writerows([all_cme.keys()] + list(zip(*all_cme.values())))

                # total_time_dicts = analyze_res(args, i_th_experiment)
    #         plt.plot(list(total_time_dicts.values()), label = label + " under " + model)
    # plt.xlabel(list(total_time_dicts.keys()))
    # plt.ylabel('Running time')
    # plt.legend()
    # plt.savefig("Figure/" + str(args.length_num) + '/' + i_th_experiment + ".eps")
    # plt.close()
if __name__ == '__main__':
    main()