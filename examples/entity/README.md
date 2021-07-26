# Run experiment for pre-defined train/test splits

Prepare data: download lcquad folder from [box](https://ibm.box.com/s/n4nnh1i1efh7s30ew2mho8moq9k3qvou)
and put the folder inside the local `data` directory

1. Rule-EL

```bash
python run_rule_el.py --use_refcount
python run_rule_el.py
python run_rule_el.py --use_blink --use_only_blink_candidates
python run_rule_el.py --use_blink
```

2. LNN-EL

```bash
"""
export EXP_NAME=LNN_EL_without_refcount
export MODEL_NAME=complex_pure_ctx
python run_lnn_el.py \
--experiment_name exp_$EXP_NAME_$MODEL_NAME \
--model_name $MODEL_NAME \
--num_epoch 30 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name log_$EXP_NAME_$MODEL_NAME.log \
--checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME.pt


export EXP_NAME=LNN_EL_plus_refcount
export MODEL_NAME=complex_pure_ctx
python run_lnn_el.py \
--use_refcount \
--experiment_name exp_$EXP_NAME_$MODEL_NAME \
--model_name $MODEL_NAME \
--num_epoch 30 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name log_$EXP_NAME_$MODEL_NAME.log \
--checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME.pt


export EXP_NAME=LNN_EL_blink_candidates_only
export MODEL_NAME=complex_pure_ctx
python run_lnn_el.py \
--use_refcount --use_only_blink_candidates --use_blink \
--experiment_name exp_$EXP_NAME_$MODEL_NAME \
--model_name $MODEL_NAME \
--num_epoch 30 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name log_$EXP_NAME_$MODEL_NAME.log \
--checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME.pt


export EXP_NAME=LNN_EL_blink
export MODEL_NAME=complex_pure_ctx
python run_lnn_el.py \
--use_refcount --use_blink \
--experiment_name exp_$EXP_NAME_$MODEL_NAME \
--model_name $MODEL_NAME \
--num_epoch 30 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name log_$EXP_NAME_$MODEL_NAME.log \
--checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME.pt

"""
```