# Introduction

We use 6 pre-trained transformer-based models (i.e., BERT, ALBERT, RoBERTa, XLNet, BERTOverflow, CostsentBert) for multi-class API review classification task with 10-fold cross-validation. In total, we fine-tune models more than 7k times.

Do remember to change your file name or location of the data into the scripts.

# Dataset
API review dataset have been used. The sources of the dataset are noted in the paper. Credit to the original authors. You can download the original dataset in the following sources.
+ [API review dataset](https://github.com/giasuddin/OpinionValueTSE):

We also provide the data preprocess scripts in the ```aspect_classifier/preprocess``` folder.

We also provide the test dataset in the ```aspect_classifier/data``` folder (in our experiment, we use 10-fold cross-validation thus we don't use the test dataset).

# Approaches
## PTMs(pre-trained transformer-based models)
For PTMs except CostSenstBert we use Huggingface [Transformer library](https://huggingface.co/transformers/). For CostSenstBert we use its [replicated package](https://github.com/H-TayyarMadabushi/Cost-Sensitive_Bert_and_Transformers). 
## baseline
We reproduced the Opiner in the ```Gias_approach``` folder.

To run the expriment, search for the optimal parameters, and output the evaluation metrics, please use the ```opiner_cvgridsearch.py``` script.

# Scripts
## Running PTMs
For PTMs, We used five python scripts (model name+'_train.py') in aspect_classification/classification folder to train BERT, RoBERTa, ALBERT, BERTOverflow and XLNet.

For example, to train the BERT model, you can directly run the ```bert_train.py```, the scripts would automatically search for the optimal hyper-parameters, run 10-fold cross-validation and record the evaluation result as the csv file in the same folder. 

The result would contain the optimal hyper-parameter and the evaluation metrics of avg F1, recall, precision, MCC, and AUC. 

If you wanna change the scope of parameter searching, please change them in the scripts.
## Running CostSensBert
For CostSensBert, you can run the [script](deepsummary/PTM-classification/aspect_classifier/classification/costsensitivebert/Cost-Sensitive_Bert_and_Transformers/examples/text-classification/run_glue_class-weighted.py) to train the model and get the result.

For more information, please refet to its [original package](https://github.com/H-TayyarMadabushi/Cost-Sensitive_Bert_and_Transformers)

