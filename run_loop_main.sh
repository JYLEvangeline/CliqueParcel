for i in {10..10}; do 
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode group --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode seperate --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode random --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode concept_plus_semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode seq_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode maximum_diff --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode random_plus_avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-3.5-turbo --mode full_random --i_th_experiment "${i}.query_length"


/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode group --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode seperate --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode random --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode concept_plus_semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode seq_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode maximum_diff --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode random_plus_avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset trec --model gpt-4 --mode full_random --i_th_experiment "${i}.query_length"


/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode group --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode seperate --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode random --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode concept_plus_semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode seq_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode maximum_diff --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode random_plus_avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-3.5-turbo --mode full_random --i_th_experiment "${i}.query_length"


/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode group --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode seperate --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode random --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode concept_plus_semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode seq_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode maximum_diff --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode random_plus_avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset squad --model gpt-4 --mode full_random --i_th_experiment "${i}.query_length"


/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode group --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode seperate --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode random --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode concept_plus_semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode seq_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode maximum_diff --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode random_plus_avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode full_random --i_th_experiment "${i}.query_length"


/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode group --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode seperate --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode random --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode concept_plus_semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode seq_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode maximum_diff --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode random_plus_avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset hotpot_qa --model gpt-4 --mode full_random --i_th_experiment "${i}.query_length"

done

echo "Loop finished"