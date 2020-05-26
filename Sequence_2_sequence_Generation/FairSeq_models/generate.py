"""
Utility to Generate Sentences from Test Set
"""

import argparse
from utility import run_command
from config.hyperparameters import BATCH_SIZE


def generate(model, batch_size):
    if model == "LSTM":
        command = "fairseq-generate data/fairseq_binaries \
                    --path checkpoints/lstm/checkpoint_last.pt \
                    --batch-size 64 --beam 3"
    elif model == "CNN":
        command = "fairseq-generate data/fairseq_binaries \
                    --path checkpoints/conv/checkpoint_last.pt \
                    --batch-size 64 --beam 3"
    else:
        raise NotImplementedError

    run_command(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to Generate Sentences from Test Set"
    )

    parser.add_argument(
        "-m",
        "--model",
        default="LSTM",
        choices=["LSTM", "CNN"],
        help="Select the Seq2Seq Model to train",
    )

    parser.add_argument(
        "-b", "--batch-size", default=BATCH_SIZE, help="Training Batch Size"
    )

    args = parser.parse_args()

    generate(args.model, args.batch_size)
