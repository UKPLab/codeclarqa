# Python Code Generation by Asking Clarification Questions

This repo include codes that we used for the experiments in our ACL 2023 paper:


```
Python Code Generation by Asking Clarification Questions
Haau-Sing Li, Mohsen Mesgar, André F. T. Martins, Iryna Gurevych
```

Contact person: [Haau-Sing Li](mailto:hli@ukp.tu-darmstadt.de)

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> ⚠️ **This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.** 

### Usage

1. Installing packages from `requirements.txt`. Note that for ranking model please refer to our [fork](https://github.com/lhaausing/transformer_rankers) of `transformer_rankers`. (We use `Python: 3.9.12` and `cuda 11.6`)

2. Download our [dataset](https://drive.google.com/file/d/1bM-b-L10vNpk7Onyft9BXK8GlMIGl52q/view?usp=sharing).

4. (Optional) If you want to generate the dataset files for training on different modules, you can use the following script.
```bash
python3 gen_dataset.py --/path/to/data/
```

5. Training

- Clarification Need Prediction

```bash
python3 classifier.py --model_name $MODEL \
                      --data_dir /path/to/data \
                      --model_dir /path/to/saved/models \
                      --seed $SEED
```

- CQ Ranking
```bash
python3 ranker.py --model_name $MODEL --seed $SEED --num_epochs $NUM_EPOCHS \
                        --negative_sampling_strategy $SAMPLING_STRATEGY \
                        --train_batch_size 32 --eval_batch_size 1024 \
                        --learning_rate 5e-5 --max_seq_len 192 \
                         --save_dir /path/to/dir
```
- Code Generation
```bash
python3 {t5|plbart|causal_lm}.py --data_dir /path/to/data 
                                 --model_dir /path/to/saved/models 
                                 --model_name $MODEL 
                                 --data_affix $DATA_AFF 
                                 --seed $SD 
                                 --num_train_epochs #I use 40 since it converges only after these many.
```
- You should run code from `./evaluate_module` to evaluate models, at least for rankers and code generator as causal LMs (since they take more time).

6. Inference on the whole pipeline. You should run files from `./evaluate_pipeline`. the order should be:
- `pred_ranker.py`
- `gen_data_preds.py`
- `pred_plbart/t5.py`
