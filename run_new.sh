datasets=('trec' 'squad' 'hotpot_qa' 'CSQA' 'GSM8K' 'MATH' 'ANLI' 'MMLU')
# datasets=('GSM8K')

models=('gpt-3.5-turbo' 'gpt-4')
for dataset in "${datasets[@]}"; do
for model in "${models[@]}"; do
for i in {1..10}; do 
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset $dataset --model $model --mode group --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset $dataset --model $model --mode seperate --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset $dataset --model $model --mode semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset $dataset --model $model --mode concept_plus_semantic_sim --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset $dataset --model $model --mode avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset $dataset --model $model --mode seq_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset $dataset --model $model --mode maximum_diff --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset $dataset --model $model --mode random_plus_avg_length --i_th_experiment "${i}.query_length"
/home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_main.py --dataset $dataset --model $model --mode full_random --i_th_experiment "${i}.query_length"


done
done
done

echo "Loop finished"