!/bin/bash

# tuning on the learning rate
for lrate in 0.001 0.0005 0.0001 0.00005 0.00001 0.005 0.01
do
    # Compute diachronic average pairwise distance
    export EXP_NAME=LNN_EL_blink_tuning_learning_rate
    export MODEL_NAME=complex_pure_ctx
    python run_lnn_el.py \
    --use_blink \
    --experiment_name exp_$EXP_NAME_$MODEL_NAME \
    --model_name $MODEL_NAME \
    --num_epoch 200 \
    --margin 0.601 \
    --alpha 0.9 \
    --learning_rate $lrate \
    --log_file_name log_$EXP_NAME_$MODEL_NAME.log \
    --checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME_$lrate.pt
done


# tuning on the alpha values (alpha has a constraint)
for alphavalue in 0.7 0.75 0.8 0.85 0.9 0.95
do
    # Compute diachronic average pairwise distance
    export EXP_NAME=LNN_EL_blink_tuning_alphavalue
    export MODEL_NAME=complex_pure_ctx
    python run_lnn_el.py \
    --use_blink \
    --experiment_name exp_$EXP_NAME_$MODEL_NAME \
    --model_name $MODEL_NAME \
    --num_epoch 200 \
    --margin 0.601 \
    --alpha $alphavalue \
    --learning_rate 0.001 \
    --log_file_name log_$EXP_NAME_$MODEL_NAME.log \
    --checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME_$alphavalue.pt
done