for (( i = 3; i <= 10; i++ )); do
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode seperate --model gpt-3.5-turbo --label fine_label 
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode label --model gpt-3.5-turbo --label fine_label
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode random --model gpt-3.5-turbo --label fine_label
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode semantic_sim --model gpt-3.5-turbo --label fine_label


    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode seperate --model gpt-4 --label coarse_label
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode label --model gpt-4 --label coarse_label
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode random --model gpt-4 --label coarse_label
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode semantic_sim --model gpt-4 --label coarse_label

    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode seperate --model gpt-4 --label fine_label
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode label --model gpt-4 --label fine_label
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode random --model gpt-4 --label fine_label
    /home/eva/miniconda3/envs/prompt/bin/python /home/eva/code/promptwork/MetaPrompt/gpt_efficiency_new.py --mode semantic_sim --model gpt-4 --label fine_label
done

echo "Loop finished"