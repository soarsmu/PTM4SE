# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
This code has been adapted from 'run_glue.py'

Finetuning the library models for generic sequence classification with cost-weighting (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa).
"""


import os
import sys
import csv
import time
import torch
import logging
import dataclasses

import numpy      as np

from typing                                import Dict, List, Optional
from filelock                              import FileLock
from dataclasses                           import dataclass, field
from sklearn.metrics                       import f1_score, recall_score, precision_score
from sklearn.metrics                       import matthews_corrcoef
from torch.utils.data.dataset              import Dataset


from transformers                          import AutoTokenizer, EvalPrediction
from transformers                          import AutoConfig, AutoModelForSequenceClassification
from transformers.tokenization_utils       import PreTrainedTokenizer
from transformers.tokenization_roberta     import RobertaTokenizer, RobertaTokenizerFast
from transformers.data.processors.glue     import glue_convert_examples_to_features
from transformers.data.processors.utils    import InputFeatures
from transformers.data.processors.utils    import DataProcessor, InputExample, InputFeatures
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
import six
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)

## From /src/transformers/data/metrics/__init__.py
def simple_accuracy(preds, labels):
    return (preds == labels).mean()
def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    mcc = matthews_corrcoef(labels, preds) # labels and preds instead of other way!
    return_dict = { 
        "acc" : acc,
        "mcc" : mcc, 
    }
    for average in [ "binary", "micro", "macro", "weighted"] : 
        f1        = f1_score       (y_true=labels, y_pred=preds, average=average)
        precision = precision_score(y_true=labels, y_pred=preds, average=average)
        recall    = recall_score   (y_true=labels, y_pred=preds, average=average)
        return_dict[ "f1_"        + average ] = f1
        return_dict[ "precision_" + average ] = precision
        return_dict[ "recall_"    + average ] = recall
    return return_dict

# ColaProcessor from src/transformers/data/processors/glue.py
class Processor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["data"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "val.tsv")), "dev")

    def get_labels(self):
        """See base class."""

        return [str(x) for x in range(2)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)

            text_a = line[3]
            # label = line[5]
            if 'Security' not in line[5]:
                label = str(0)
            else:
                label = str(1)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        print(self.get_labels())
        return examples


# GlueDataTrainingArguments from: src/transformers/data/datasets/glue.py
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    name: str = field(metadata={"help": "A name for this task (Cache folders will be based on this"})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    
# From: src/transformers/data/datasets/glue.py
class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: DataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        evaluate=False,
    ):
        self.args = args
        processor = Processor()
        self.output_mode = 'classification'
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train", tokenizer.__class__.__name__, str(args.max_seq_length), args.name,
            ),
        )
        print(cached_features_file)
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = processor.get_labels()
                
                # if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                #     RobertaTokenizer,
                #     RobertaTokenizerFast,
                #     XLMRobertaTokenizer,
                # ):
                #     # HACK(label indices are swapped in RoBERTa pretrained model)
                #     label_list[1], label_list[2] = label_list[2], label_list[1]

                    
                examples = (
                    processor.get_dev_examples(args.data_dir)
                    if evaluate
                    else processor.get_train_examples(args.data_dir)
                )
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode='classification', 
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
        

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    class_weights: Optional[str] = field(
        default=None, metadata={"help": "Comma seperated list of class weights. Weight for Class 0, Weight for Class 1"}
    )


class Modifiedargments:

    def __init__(self, datadir):
        self.name = 'CoLA'
        self.data_dir =  datadir
        self.max_seq_length = 128
        self.overwrite_cache = False

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    print(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels  = 2 ## Defined in processor. Change labels before changing this (also hardcoded elsewhere) .
    output_mode = 'classification'

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2, ## Defined in processor. Change labels before changing this (also hardcoded elsewhere) .
        finetuning_task='cola', ## Not sure why we need this!
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )


    #############################################################################
    #                            Set Class Weights                             ##
    #############################################################################
    class_weights = None
    if not model_args.class_weights is None :
        class_weights = model_args.class_weights.lstrip().rstrip().split( ',' )
        class_weights = [ float(i) for i in class_weights ]
        logger.info(
            "Using Class Weights {}".format( ','.join( [ str(i) for i in class_weights ] ) )
        )
        class_weights = torch.tensor( class_weights, dtype=torch.float, device=training_args.device )
    model.set_class_weights( class_weights ) 
    #############################################################################

    
    # Get datasets
    aspect = 'Security'
    print(data_args)
    train_path = '/workspace/RoBERTa_textclassification/aspect_classifier/costsensitivebert/Cost-Sensitive_Bert_and_Transformers/examples/glue_data/CoLA/'+aspect+'/0/'
    eval_path = '/workspace/RoBERTa_textclassification/aspect_classifier/costsensitivebert/Cost-Sensitive_Bert_and_Transformers/examples/glue_data/CoLA/'+aspect+'/0/'
    train_dataset = GlueDataset(Modifiedargments(datadir = train_path), tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = GlueDataset(Modifiedargments(datadir = eval_path), tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        # return glue_compute_metrics(data_args.task_name, preds, p.label_ids)
        with open( training_args.output_dir + 'predictions.csv', 'w' ) as fh:
            writer = csv.writer( fh )
            preds  = [ [i] for i in preds ]
            writer.writerows( preds ) 
        return acc_and_f1(preds, p.label_ids)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=True))

        for eval_dataset in eval_datasets:
            result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_classification.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results classification *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
