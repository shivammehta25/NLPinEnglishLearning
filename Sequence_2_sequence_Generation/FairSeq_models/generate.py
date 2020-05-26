"""
Utility to Generate Sentences from Test Set
"""

import argparse
from utility import run_command
from config.hyperparameters import BATCH_SIZE


def generate(model, batch_size, sub_model):

    if model == "LSTM":
        command = "fairseq-generate data/fairseq_binaries \
                    --path checkpoints/lstm/checkpoint_{}.pt \
                    --batch-size {} --beam 3".format(
            sub_model, batch_size
        )
    elif model == "CNN":
        command = "fairseq-generate data/fairseq_binaries \
                    --path checkpoints/conv/checkpoint_{}.pt \
                    --batch-size {} --beam 3".format(
            sub_model, batch_size
        )
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
        "-sm",
        "--sub-model",
        default="last",
        choices=["best", "last"],
        help="Select which model to generate with the one with best valid loss or the last epoch trained model",
    )

    parser.add_argument(
        "-b", "--batch-size", default=BATCH_SIZE, help="Training Batch Size"
    )

    args = parser.parse_args()

    generate(args.model, args.batch_size, args.sub_model)
