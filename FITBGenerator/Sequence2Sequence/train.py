"""
Utility to train the model
Uses FairSeq commands
"""

import argparse
from utility import run_command
from config.data import FAIRSEQ_PREPROCESSED_DATASET


# run_command("")


def train(model_name):
    if model_name == "LSTM":
        command = "!CUDA_VISIBLE_DEVICES=0 fairseq-train {}  --clip-norm 5 --batch-size 64 \
                    --save-dir checkpoints/lstm --arch lstm --max-epoch 15 --encoder-hidden-size 258 \
                    --encoder-layers 2  --decoder-hidden-size 258 --decoder-layers 2 --optimizer adam --lr 0.001  \
                    --dropout 0.3 --encoder-embed-path glove.6B.300d.txt --encoder-bidirectional --encoder-embed-dim 300 \
                    --decoder-embed-dim 300 --no-epoch-checkpoints --decoder-embed-path glove.6B.300d.txt --decoder-out-embed-dim 300 \
                    --num-workers 3".format(
            FAIRSEQ_PREPROCESSED_DATASET
        )
    elif model_name == "CNN":
        command = "!CUDA_VISIBLE_DEVICES=0 fairseq-train {} --batch-size 64 \
                    --save-dir checkpoints/conv --arch fconv_iwslt_de_en --max-epoch 15 \
                    --optimizer adam --lr 0.001  \
                    --dropout 0.3 --encoder-embed-path glove.6B.300d.txt --encoder-embed-dim 300 \
                    --decoder-embed-dim 300 --no-epoch-checkpoints --decoder-embed-path glove.6B.300d.txt --decoder-out-embed-dim 300 \
                    --num-workers 3".format(
            FAIRSEQ_PREPROCESSED_DATASET
        )
    else:
        raise NotImplementedError

    run_command(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility to preprocess the dataset")

    parser.add_argument(
        "-m",
        "--model",
        default="LSTM",
        choices=["LSTM", "CNN"],
        help="Select the Seq2Seq Model to train",
    )

    args = parser.parse_args()

    train(args.model)
