# How to run the code

## How to format the data

1. please refer to `create-splits-for-5-fold-cross-validation.ipynb` to see 
how to create 5 splits from `data/type_train.csv` and `data/type_test.csv`
2. These are the steps I did in the notebook
    - concatenate `type_train.csv` and `type_test.csv` to make `type_train_test_sorted.csv`
    - read the new csv as dataframe and create a mew `Mention` column from `Mention_label`
    - create `QuestionMention` column by concatenating `Question` and `Mention` columns with --
    - sort the dataframe by ['QuestionMention', 'Label']
    - create train and test csv files for each of the split under `data/type-qald-5-folds/`

## Run LNN-EL with learnable thresholds on QALD-9

- Run the scipt by specifying model type, use the following:
    - `purename`
    - `context`
    - `type`
    - `complex_pure_ctx`
    - `complex_pure_ctx_type`
    - `lr`
- You should also name the experiment (`experiment_name`),
 log_file_name (`exp_complex_pure_ctx_type.log`),
  and `checkpoint_name` as you like

```bash
python run_el_kfolds_type.py \
--experiment_name exp_complex_pure_ctx_type \
--model_name complex_pure_ctx_type \
--num_epoch 1 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name exp_complex_pure_ctx_type.log \
--checkpoint_name checkpoint/best_model_complex_pure_ctx_type.pt
```

- After runnning the script above, the script will automatically creates an output folder
named `output/{$experiment_name}` that contains the prediction files for five splits. However, these
predictions files do not contain missing sentences because of AMR parsing. 
- Run the following script to distribute missing sentences into each of the five splits. 
    - Specifically, there are 11 missing sentences, we divide them into five parts: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10]] 
    and add them to the end of each prediction file respectively 
- After running the script, a folder 
named `output/{$experiment_prediction_foldername}_with_missing_sentences` is created 
with the updated prediction files (adding the missing sentences)

```bash
# get top 1 and top k performance
python evaluate.py \
--experiment_prediction_foldername exp_complex_pure_ctx_type \
--missing_file_path data/type_missing.csv
```

## Run LNN-EL with fixed thresholds
- simply add `--use_fixed_threshold`
- change experiment name to save the predictions 

```bash
# run 
python run_el_kfolds_type.py \
--experiment_name exp_complex_pure_ctx_type_fixed \
--model_name complex_pure_ctx_type \
--num_epoch 1 \
--margin 0.601 \
--alpha 0.9 \
--use_fixed_threshold \
--learning_rate 0.001 \
--log_file_name exp_complex_pure_ctx_type_fixed.log \
--checkpoint_name checkpoint/best_model_complex_pure_ctx_type_fixed.pt
```

```bash
# get top 1 and top k performance
python evaluate.py \
--experiment_prediction_foldername exp_complex_pure_ctx_type_fixed \
--missing_file_path data/type_missing.csv
```


## Replicate Rule-EL performance on QALD-9

1. to specify model type (we only support three for ruleEL)
    - `purename`
    - `context`
    - `complex`: purename + context
    
```bash
python run_rule_kfolds.py \
--experiment_name exp_complex_ruleel \
--model_name complex \
--log_file_name exp_complex_ruleel.log \
--checkpoint_name checkpoint/best_model_exp_complex_ruleel.pt
```

```bash
# get top 1 and top k performance
python evaluate.py \
--experiment_prediction_foldername exp_complex_ruleel \
--missing_file_path data/type_missing.csv
```

## Analysis

1. check the notebook `weight_threshold_analysis.ipynb` 

## Prediction files we used for Hang's exit presentation
- Check `output/` [here](https://github.ibm.com/IBM-Research-AI/enhanced_amr/tree/hang-rankingloss/lnn/examples/entity/output)
