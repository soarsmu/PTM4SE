# A Cost Sensitive version of Transformers for Classification of Imbalanced Data

This version of [🤗 Transformers](https://github.com/huggingface/transformers) adds on the ability to train models with associated cost weights for different classes for sequence classification. This has been shown to be an effective strategy when dealing with imbalanced classes, especially if the training and test sets are dissimilar. 

All PyTorch models (Albert, BERT, BART, ...) have been modified to handle class weights. TensorFlow implementations of models do not yet include class weights.  

We are in the process of adding this functionality for sequence to sequence models. 

If you make use of this package, please [cite](https://github.com/H-TayyarMadabushi/Cost-Sensitive_Bert_and_Transformers#citation) our work

## Table of Contents

- [Installation](#Installation)
  * [With pip](#With-pip)
  * [From source](#From-source)
- [Examples](#Examples)
- [Propaganda detection](#Propaganda-detection)
- [Corpus Similarity testing](#Wilcoxon-Signed-rank-test-for-Corpus-Similarity)
- [Running a classifier with cost weights](#Running-a-classifier-with-cost-weights)
- [Model architectures](#Model-architectures)
- [More details on 🤗 Transformers](#More-details-on-🤗-Transformers)
- [What is cost weighting](#What-is-cost-weighting)
- [Citation](#Citation)
    
<!-- toc -->

## Installation

This repo is tested on Python 3.6+ and PyTorch 1.0.0+.

**![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) _IMPORTANT_: You must uninstall any previously installed version of transforms you might have. For this reason, it is best you use a virtual environment.**


Like Transformers, it is a good idea install this package in a [virtual environment](https://docs.python.org/3/library/venv.html). 

First you need to install PyTorch. *Please note that the Tensorflow versions do not yet have cost weighting included.*
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) page regarding the specific install command for your platform.

### With pip

The cost weighted version of transformers is not available using pip. Please install from source as described in the next section.

### From source

You can install from source by cloning the repository:

```bash
git clone https://github.com/H-TayyarMadabushi/Cost-Sensitive_Bert_and_Transformers.git
cd Cost-Sensitive_Bert_and_Transformers
pip install .
```

## Examples

We provide two example, one to run a generic classification task and the other a modified version of run_glue which ships with 🤗 Transformers. These files are: 

1. [examples/text-classification/run_classifier.py](https://github.com/H-TayyarMadabushi/Cost-Sensitive_Bert_and_Transformers/blob/master/examples/text-classification/run_classifier.py)
2. [examples/text-classification/run_glue_class-weighted.py](https://github.com/H-TayyarMadabushi/Cost-Sensitive_Bert_and_Transformers/blob/master/examples/text-classification/run_glue_class-weighted.py)

## Propaganda detection

We have used Cost Sensitive Transformers for Sentence-Level Propaganda Detection task and achieved **2nd rank** in [The Shared Task on Fine-Grained Propaganda Detection for the 2nd Workshop on NLP for Internet Freedom](https://propaganda.qcri.org/nlp4if-shared-task/leaderboard.php). 

The code can be found [in this Colab notebook](https://colab.research.google.com/drive/18t8WVBwYycyEbM02V-5LZLaqED8sdXBP?usp=sharing). 

Contact the [task organisers](https://propaganda.qcri.org/nlp4if-shared-task/index.html#organisation) to gain access tho the data. 

More details are available in [our paper](https://www.aclweb.org/anthology/D19-5018/) which describes our system (ProperGander). 

## Wilcoxon Signed-rank test for Corpus Similarity

We found that cost weighting is especially useful when the training, development and test sets are dissimilar. We show that when training and test sets are dissimilar, cost weighting can be effective. **Training and test data are often dissimilar in social media and similar user generated datasets.** You can use the [script provided in utils](utils/Wilcoxon_Signed-rank_test_for_Corpus_similarity.py) to check the similarity of corpora. [This colab notebook](https://colab.research.google.com/drive/1j-hHzpjuY98FY8470oyJH9eajYV18Qan?usp=sharing) describes its usage.

### Running a classifier with cost weights

You can fine-tune and evaluate a classification task as follows:

```shell

python Cost-Sensitive_Bert_and_Transformers/examples/text-classification/run_classifier.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --name MyTask \
  --do_train \
  --do_eval \
  --data_dir /path/to/data \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /path/to/output/ \
  --class_weights=20,1
```

The class_weights parameter is optional and will default to equal weights if not provided. As in the glue version described below, the evaluation on the development set will be written to the output directory.


### `run_glue_class-weighted.py`: Fine-tuning on GLUE tasks for sequence classification with class based cost weights

The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running any of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

You should also install the additional packages required by the examples:

```shell
pip install -r ./examples/requirements.txt
```

```shell
export GLUE_DIR=/path/to/glue
export TASK_NAME=CoLA

python ./examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/$TASK_NAME/ \
    --class_weights=20,1
```

where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file 'eval_results.txt' in the specified output_dir. In case of MNLI, since there are two separate dev sets, matched and mismatched, there will be a separate output folder called '/tmp/MNLI-MM/' in addition to '/tmp/MNLI/'.

**![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) The actual weights to use is a hyperparameter and will depend on your task and dataset**

### CoLA Results

Below are the results for CoLA with and without cost weighting (NOTE: Matthews Coefficient, MCC, is the official metric for this dataset): 


Weighting               |  MCC      | Accuracy  | F1 macro
----------------------- | --------- | --------- | --------
No Cost Weight          | 0.5676    | 0.6132    | 0.7700
With Cost Weight (20:1) | 0.5792    | 0.5845    | 0.7886


These results were obtained using the following parameters: 

```shell
per_gpu_train_batch_size=32, per_gpu_eval_batch_size=8, gradient_accumulation_steps=1, learning_rate=2e-05, weight_decay=0.0, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=3.0, max_steps=-1, warmup_steps=0, logging_dir=None, logging_first_step=False, logging_steps=500, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level='O1', local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False

bert-base-cased config: 
Model config BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 28996
}
```

**![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) The same weights are unlikely to work for other pre-trained models.**

## Model architectures

This package provides the following subset of architectures provided by 🤗 Transformers:

1. **[BERT](https://huggingface.co/transformers/model_doc/bert.html)** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. **[XLNet](https://huggingface.co/transformers/model_doc/xlnet.html)** (from Google/CMU) released with the paper [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
3. **[XLM](https://huggingface.co/transformers/model_doc/xlm.html)** (from Facebook) released together with the paper [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau.
4. **[RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)** (from Facebook), released together with the paper a [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
5. **[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)** (from HuggingFace), released together with the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into [DistilGPT2](https://github.com/huggingface/transformers/tree/master/examples/distillation), RoBERTa into [DistilRoBERTa](https://github.com/huggingface/transformers/tree/master/examples/distillation), Multilingual BERT into [DistilmBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation) and a German version of DistilBERT.
6. **[CamemBERT](https://huggingface.co/transformers/model_doc/camembert.html)** (from Inria/Facebook/Sorbonne) released with the paper [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
7. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
8. **[XLM-RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html)** (from Facebook AI), released together with the paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov.
9. **[FlauBERT](https://huggingface.co/transformers/model_doc/flaubert.html)** (from CNRS) released with the paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
10. **[BART](https://huggingface.co/transformers/model_doc/bart.html)** (from Facebook) released with the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.

## What is cost weighting

Cost weighting involves increasing the cost associated with getting a low frequency class label wrong. It is an important alternative to data augmentation for imbalanced classes. 

**![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) We found setting the weights to 1, 20 to be most effective, but also 1, 4 and so on. This is a hyperparameter.**

While the changes themselves are relatively straightforward, incorporating them into pre-trained models like BERT tends to be fairly complicated due to the high level of abstraction. 
In essence, class weights need to be passed to the cost function. Below is how this is typically done within the training loop: 

```python
OUTPUT_MODE = "classification"
NUM_TRAIN_EPOCHS = 10
ensemble_model.train()
# Init weights. WARNING: Make sure that 1 is the minority class
weights = torch.tensor([1, 20], dtype=torch.float, device=device )
for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")): 
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, ner = batch

        #  logits = ensemble_model(x1, ner) pytorch-pretrained-bert to pytorch-transformers
        x1 = (input_ids, segment_ids, input_mask, None)
        logits = ensemble_model(x1, ner)

        if OUTPUT_MODE == "classification":
            # Pass weights to cost function
            loss_fct = CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif OUTPUT_MODE == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()
        print("\r%f" % loss, end='')
        
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scheduler.step() # Add scheduler step.
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
```

## More details on 🤗 Transformers
For more details on 🤗 Transformers, see the [original package](https://github.com/huggingface/transformers).

## Citation

If you make use of this work, please cite us: 

```bibtex
@inproceedings{tayyar-madabushi-etal-2019-cost,
    title = "Cost-Sensitive {BERT} for Generalisable Sentence Classification on Imbalanced Data",
    author = "Tayyar Madabushi, Harish  and
      Kochkina, Elena  and
      Castelle, Michael",
    booktitle = "Proceedings of the Second Workshop on Natural Language Processing for Internet Freedom: Censorship, Disinformation, and Propaganda",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-5018",
    doi = "10.18653/v1/D19-5018",
    pages = "125--134",
    abstract = "The automatic identification of propaganda has gained significance in recent years due to technological and social changes in the way news is generated and consumed. That this task can be addressed effectively using BERT, a powerful new architecture which can be fine-tuned for text classification tasks, is not surprising. However, propaganda detection, like other tasks that deal with news documents and other forms of decontextualized social communication (e.g. sentiment analysis), inherently deals with data whose categories are simultaneously imbalanced and dissimilar. We show that BERT, while capable of handling imbalanced classes with no additional data augmentation, does not generalise well when the training and test data are sufficiently dissimilar (as is often the case with news sources, whose topics evolve over time). We show how to address this problem by providing a statistical measure of similarity between datasets and a method of incorporating cost-weighting into BERT when the training and test sets are dissimilar. We test these methods on the Propaganda Techniques Corpus (PTC) and achieve the second highest score on sentence-level propaganda classification.",
}

```
