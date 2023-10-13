import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="group efficiency")
    parser.add_argument("--length_of_quries", type = int, default = 16)
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--label", type = str, default = "coarse_label") # coarse_label, fine_label for trec
    parser.add_argument("--length_num", type = int, default = 256)
    parser.add_argument("--dataset", default = "hotpot_qa", choices = ["trec","squad","hotpot_qa"])
    parser.add_argument("--model", default = "gpt-3.5-turbo", choices = ["gpt-3.5-turbo", "gpt-4"])
    parser.add_argument("--mode", default = "random_plus_avg_length", choices = ["group", "sequence", "random", "full_random", "semantic_sim", "concept_plus_semantic_sim", "avg_length", "seq_length", "maximum_diff", 'random_plus_avg_length'])
    parser.add_argument("--i_th_experiment", type = str, default = "1.query_length")
    parser.add_argument("--resume", type = bool, default = True)
    args = parser.parse_args()
    return args

datasets = ["hotpot_qa"]
models = ["gpt-3.5-turbo", "gpt-4"]
modes = ["group", "seperate", "sequence", "full_random", "semantic_sim", "concept_plus_semantic_sim", "avg_length", "seq_length", "maximum_diff", 'random_plus_avg_length']
for dataset in datasets:
    for model in models:
        for mode in modes:
            string = "python gpt_efficiency_main.py --dataset " + dataset + " --model " + model + " --mode " + mode + " --i_th_experiment \"${i}.query_length\""
            print(string)
        print("\n")
