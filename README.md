### Overview

This code provides the implementation to conduct gradient-based task affinity estimation. The experiments involve multi-label classification on graphs with community detection labels and fine-tuning language mdoels. 

### Requirements

We list the package requirements under each folder. Install related packages based on the following: 

```
pip install -r requirements.txt
```

### Data Preparation

**Community detection.** We provide the datasets for conducting community detection named `data.zip` under the `./data/` folder used. Unzip the file under the folder, then one can directly load them in the code.

**Language model fine-tuning.** The code will automatically download the datasets. Please specify the name of the dataset while using it.  

### Usage

Our algorithm contains four steps:

1. **Training meta initializations**: This trains a multitask learning model on the combination of all tasks 

For graph datasets, use `train_node_pred_multitask.py` to train the model. Modify the `--dataset` to specify the dataset. This script will save the model checkpoints under a `./saved/` folder. Please create one before usage. 

```Python
python train_node_pred_multitask.py --dataset youtube \
     --model sign --num_layers 3 --hidden_channels 256 --lr 0.01 --dropout 0.1 --mlp_layers 2\
     --evaluator f1_score --sample_method decoupling --batch_size 1000 --epochs 100 --device 2 --runs 50\
     --save_name youtube_sign_mtl
```

For fine-tuning language models,  use `train_multi_instruction.py` to fine-tune a model on all instructions. Modify the `--task_name` to specify the dataset name. Specify the `--template_idx` from 1 to 100 in order to train on all the instructions. 

```
CUDA_VISIBLE_DEVICES=1 python train_multi_instruction.py \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --model_name_or_path t5-base \
    --max_source_length 512 \
    --max_target_length 128 \
    --pad_to_max_length True \
    --generation_max_length 128 \
    --data_dir data/splits/default \
    --task_dir data/tasks \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-05 \
    --lr_scheduler_type constant \
    --max_steps 5000 \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 1000 \
    --metric_for_best_model eval_exact_match\
    --greater_is_better True \
    --task_name rte \
    --template_idx 0 1 2 3 4 5 6 7 8 9 \
    --output_dir saved/ \
    --load_best_model_at_end \
    --disable_tqdm True \
    --save_name rte_mtl \
    --train_lora True --lora_rank 4 --lora_alpha 32 --runs 10
```

2. **Compute the projected gradients** on each training data sample of all tasks. 

For graph datasets, use `fast_estimate_collect_gradients.py`. Specify `--project_dim` as the number of projections. This file will save the projection matrix and all projected gradients under a `./gradients/` folder. Please create the folder before usage. 

```python
python fast_estimate_collect_gradients.py --dataset youtube \
    --model sign --num_layers 3 --hidden_channels 256 --dropout 0.1 --mlp_layers 2 --sample_method decoupling\
    --device 3 --run $run --create_projection\
    --project_dim 200 
```

For fine-tuning language models, use `fast_estimate_collect_gradients.py`. Use `--load_model_dir` to specify a saved checkpoint directory as the base model. The rest of parameters would be the same as above. 

```
CUDA_VISIBLE_DEVICES=0 python fast_estimate_collect_gradients.py \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --model_name_or_path t5-base \
    --max_source_length 512 \
    --max_target_length 128 \
    --pad_to_max_length True \
    --generation_max_length 128 \
    --data_dir data/splits/default \
    --task_dir data/tasks \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-05 \
    --lr_scheduler_type constant \
    --max_steps 5000 \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 1000 \
    --metric_for_best_model eval_exact_match\
    --greater_is_better True \
    --task_name rte \
    --template_idx 0 1 2 3 4 5 6 7 8 9 \
    --output_dir saved/ \
    --load_best_model_at_end \
    --disable_tqdm True \
    --save_name rte_mtl \
    --train_lora True --lora_rank 4 --lora_alpha 32 \
    --run 0 --create_projection True --project_dim 200 \
    --load_model_dir rte_t5-base_run_0
```

3. **Solve linear regression on gradients** to estimate the output of model fine-tuned on a subset of tasks. 

For graph datasets, use `fast_estimate_linear_model.py`. Specify `--save_name` for the file to save the evaluation results of estimated models. Inside the file, one can modify the subsets collection file under `./sampled_tasks/sample_{dataset}_128_10.txt` to specify the sampled subsets of tasks. Usually, it should be randomly sampled subsets. 

```
python fast_estimate_linear_model.py --dataset youtube \
        --model sign --num_layers 3 --hidden_channels 256 --dropout 0.1 --mlp_layers 2 --sample_method decoupling\
        --device 1 --run 0 --save_name sample_youtube_linear_approximate_0 --project_dim 200
```

For fine-tuning language models, use `fast_estimate_linear_model.py`. Similarly, specify `--save_name` for the file to save the evaluation results of estimated models.

```
CUDA_VISIBLE_DEVICES=0 python fast_estimate_linear_model.py \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --model_name_or_path t5-base \
    --max_source_length 512 \
    --max_target_length 128 \
    --pad_to_max_length True \
    --generation_max_length 128 \
    --data_dir data/splits/default \
    --task_dir data/tasks \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-05 \
    --lr_scheduler_type constant \
    --max_steps 5000 \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 1000 \
    --metric_for_best_model eval_exact_match\
    --greater_is_better True \
    --task_name rte \
    --template_idx 0 1 2 3 4 5 6 7 8 9 \
    --output_dir saved/ \
    --load_best_model_at_end \
    --disable_tqdm True \
    --train_lora True --lora_rank 4 --lora_alpha 32 \
    --run 0 --create_projection True --project_dim 200 \
    --load_model_dir rte_t5-base_run_0 \
    --save_name rte_linear_approximate_0 
```

4. Clustering tasks to generate task groups and train one model on each subset of tasks. 

We provide an implementation of our SDP-based clustering algorithm in `./exps_on_graph_datasets/run_clustering.py`. One can load a computed task affinity score matrix and apply the clustering on top of the matrix. See the file for details.