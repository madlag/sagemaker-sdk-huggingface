from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch
import copy

import transformers
import transformers.utils.logging

logger = logging.getLogger(__name__)

def main(param_dict):
    import nn_pruning.examples.question_answering.qa_sparse_xp as qa_sparse_xp

    qa = qa_sparse_xp.QASparseXP(param_dict)
    qa.run()

original_param_dict = {
  "model_name_or_path": "bert-base-uncased",
  "dataset_name": "squad",
  "do_train": 1,
  "do_eval": 1,
  "per_device_train_batch_size": 16,
  "max_seq_length": 384,
  "doc_stride": 128,
  "num_train_epochs": 20,
  "logging_steps": 100,
  "save_steps": 5000,
  "eval_steps": 250,
  "save_total_limit": 5,
  "seed": 17,
  "evaluation_strategy": "steps",
  "learning_rate": 3e-5,
  "mask_scores_learning_rate": 1e-2,
  "output_dir": "output/squad_test3",
  "logging_dir": "output/squad_test3",
  "overwrite_cache": 0,
  "overwrite_output_dir": 1,
  "warmup_steps": 5400,
  "initial_warmup": 1,
  "final_warmup": 10,
  "initial_threshold": 0,
  "final_threshold": 0.1,
  "dense_pruning_method": "sigmoied_threshold:1d_alt",
  "dense_block_rows":1,
  "dense_block_cols":1,
  "dense_lambda":0.25,
  "attention_pruning_method": "sigmoied_threshold",
  "attention_block_rows":32,
  "attention_block_cols":32,
  "attention_lambda":1.0,
  "ampere_pruning_method": "disabled",
  "mask_init": "constant",
  "mask_scale": 0.0,
  "regularization": "l1",
  "regularization_final_lambda": 10,
  "distil_teacher_name_or_path":"csarron/bert-base-uncased-squad-v1",
  "distil_alpha_ce": 0.1,
  "distil_alpha_teacher": 0.9,
  "attention_output_with_dense": 0
}


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--num-train-epochs", type=float, default=20)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    
    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    

    parser.add_argument("--final-warmup", type=int, default=10)
    parser.add_argument("--regularization-final-lambda", type=float, default=10)
    parser.add_argument("--dense-pruning-method", type=str, default="sigmoied_threshold:1d_alt")
    parser.add_argument("--dense-block-rows", type=int, default=1)
    parser.add_argument("--dense-block-cols", type=int, default=1)
    parser.add_argument("--dense-lambda", type=float, default=1.0)
    parser.add_argument("--attention-pruning-method", type=str, default="sigmoied_threshold")
    parser.add_argument("--attention-block-rows", type=int, default=1)
    parser.add_argument("--attention-block-cols", type=int, default=1)
    parser.add_argument("--attention-lambda", type=float, default=1.0)
    
    parser.add_argument("--attention-output-with-dense", type=int, default=0)

    
    #parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    #parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    #parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    #parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
        
    args, _ = parser.parse_known_args()
    
    
    param_dict = copy.deepcopy(original_param_dict)
    param_dict["num_train_epochs"] = args.num_train_epochs
    param_dict["per_device_train_batch_size"] = args.per_device_train_batch_size
    param_dict["output_dir"] = args.output_dir
    param_dict["logging_dir"] = args.output_dir
    param_dict["final_warmup"] = args.final_warmup
    
    param_dict["regularization_final_lambda"] = args.regularization_final_lambda
    param_dict["dense_pruning_method"] = args.dense_pruning_method
    param_dict["dense_block_rows"] = args.dense_block_rows
    param_dict["dense_block_cols"] = args.dense_block_cols
    param_dict["dense_lambda"] = args.dense_lambda
    param_dict["attention_pruning_method"] = args.attention_pruning_method
    param_dict["attention_block_rows"] = args.attention_block_rows
    param_dict["attention_block_cols"] = args.attention_block_cols
    param_dict["attention_lambda"] = args.attention_lambda
    param_dict["attention_output_with_dense"] = args.attention_output_with_dense
        
    import sys
    print(sys.argv)
    print(param_dict)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    #if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {param_dict}")

    main(param_dict)    

    if False:
        # writes eval result to file which can be accessed later in s3 ouput
        with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
            print(f"***** Eval results *****")
            for key, value in sorted(eval_result.items()):
                writer.write(f"{key} = {value}\n")

        # Saves the model to s3
        trainer.save_model(args.model_dir)

