"""
Dataset Level Prprocessing Strategies will be written here
Make sure your dataset is present in the preprocessing configurations
"""

import argparse
import json
import logging
import os
import random

from tqdm import tqdm

from config.data import (
    DATA_FOLDER,
    DATA_FOLDER_PROCESSED,
    DATA_FOLDER_RAW,
    DATASETS,
    RAW_FILENAMES,
    SQUAD_NAME,
)
from config.root import LOGGING_FORMAT, LOGGING_LEVEL
from utils import nlp_sentence

# Initialize logger for this file
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)

INPUT_PATH = os.path.join(DATA_FOLDER, DATA_FOLDER_RAW)
OUTPUT_PATH = os.path.join(DATA_FOLDER, DATA_FOLDER_PROCESSED)


def convert_to_file_without_answers(
    dataset, dataset_type="train", get_impossible=False
):
    """
    Takes an input json and generates dataset_type.paragraphs and dataset_type.questions
    Input:
    dataset : string -> Name of json input
    dataset_type: string -> Type of dataset like (Train, test, valid)
    get_impossible: boolean -> Flag to get unanswerable questions
    """
    if not os.path.exists(os.path.join(OUTPUT_PATH, SQUAD_NAME)):
        os.makedirs(os.path.join(OUTPUT_PATH, SQUAD_NAME))

    para_output = open(
        os.path.join(OUTPUT_PATH, SQUAD_NAME, dataset_type + ".paragraphs"), "w"
    )
    question_output = open(
        os.path.join(OUTPUT_PATH, SQUAD_NAME, dataset_type + ".questions"), "w"
    )
    dataset = dataset["data"]
    dataset_size = []
    for paragraphs in tqdm(dataset):
        paragraphs = paragraphs["paragraphs"]
        for i, paragraph in enumerate(paragraphs):
            para = paragraph["context"]
            for questionanswers in paragraph["qas"]:
                if questionanswers["is_impossible"]:
                    continue
                question = questionanswers["question"]
                para = para.replace("\n", " ")
                para_output.write(para.strip().lower() + "\n")
                question_output.write(question.strip().lower() + "\n")
                dataset_size.append(i)
    logger.info("Size of the {} dataset: {}".format(dataset_type, len(dataset_size)))
    para_output.close()
    question_output.close()


def split_train_valid(dataset_name, split_ratio=0.9):
    """
    Splits the train set to a validation set
    creates files in the processed folder with 
    """
    logger.debug(
        "Splitting the {}'s train set into train and valid".format(dataset_name)
    )
    if not os.path.exists(os.path.join(OUTPUT_PATH, dataset_name)):
        raise NotImplementedError(
            "The Dataset has not been preprocessed yet please call the \
                 processing method before spliting the trainset"
        )

    filename_paragraph = os.path.join(OUTPUT_PATH, dataset_name, "train.paragraphs")
    filename_questions = os.path.join(OUTPUT_PATH, dataset_name, "train.questions")

    with open(filename_paragraph) as paragraphs_file, open(
        filename_questions
    ) as questions_file:
        data_paragraphs = paragraphs_file.readlines()
        data_questions = questions_file.readlines()

    logger.debug(
        "# of Paragraphs: {} # of Questions: {} ".format(
            len(data_paragraphs), len(data_questions)
        )
    )

    assert len(data_paragraphs) == len(
        data_questions
    ), "Number of Paragraphs and Questions mismatch"

    # Output files
    train_paragraphs_file = open(
        os.path.join(OUTPUT_PATH, dataset_name, "train.paragraphs"), "w"
    )
    valid_paragraphs_file = open(
        os.path.join(OUTPUT_PATH, dataset_name, "valid.paragraphs"), "w"
    )
    train_questions_file = open(
        os.path.join(OUTPUT_PATH, dataset_name, "train.questions"), "w"
    )
    valid_questions_file = open(
        os.path.join(OUTPUT_PATH, dataset_name, "valid.questions"), "w"
    )

    train_count, valid_count = 0, 0

    for i in tqdm(range(len(data_paragraphs))):
        if random.random() < split_ratio:
            train_paragraphs_file.write(data_paragraphs[i].strip() + "\n")
            train_questions_file.write(data_questions[i].strip() + "\n")
            train_count += 1
        else:
            valid_paragraphs_file.write(data_paragraphs[i].strip() + "\n")
            valid_questions_file.write(data_questions[i].strip() + "\n")
            valid_count += 1

    logger.info(
        "Total Trainset: {} | Total ValidSet: {}".format(train_count, valid_count)
    )


# def convert_to_file_on_answers(dataset, dataset_type="train", get_impossible=False):
#     """
#     Generates output file with paragraph at answer level answer level
#     TODO: Implement it when I will be filtering answer level repition
#     Incase this method is not implemented it was not used and will be removed
#     in the next version.
#     """
#     raise NotImplementedError


def load_json(filelocation):
    """
    Takes Filename as input and returns a Json object
    Input:
    filelocation: string
    Returns:
    json_data: json object
    """
    with open(filelocation) as file:
        json_data = json.load(file)

    return json_data


def preprocess_squad(name, mode, filter):
    """
    PreProcesses Squad
    Input:
    name: string -> Name of the dataset
    mode: string -> To replicate sentences based on number of answers or just questions
    """
    logger.debug("PreProcessing SQUAD")
    # TODO: remove from here
    # split_train_valid(name)
    logger.debug("Loading JSON")
    train_file = load_json(os.path.join(INPUT_PATH, RAW_FILENAMES[name]["train"]))
    test_file = load_json(os.path.join(INPUT_PATH, RAW_FILENAMES[name]["test"]))

    if mode.upper() == "QUESTION" and not filter:
        convert_to_file_without_answers(train_file, "train")
        convert_to_file_without_answers(test_file, "test")
    else:
        filter_sentences_on_answer(train_file, "train")
        filter_sentences_on_answer(test_file, "test")

    logger.debug("Now we will split train set to train and valid set")
    split_train_valid(name)

    logger.info("{} Preprocessed".format(name))


def extract_filtered_sentences(questionanswers, para):
    """
    Method returns filtered sentences from the answers and para for SQUAD
    """
    tokenized_paragraph = nlp_sentence(para)
    sentences = [sent.string for sent in tokenized_paragraph.sents]

    filtered_sentences = set()

    # This iterates over every answer in question
    for answer in questionanswers["answers"]:
        answer_index = answer["answer_start"]
        length = 0

        # find sentence that has answer and filter them
        for sentence in sentences:
            if answer_index <= length + len(sentence):
                filtered_sentences.add(sentence.replace("\n", " ").strip())
                break
            length += len(sentence)

        if not filtered_sentences:
            print("Length : {}".format(length))
            raise Exception("One of the Answers had no sentence please check the data")

    return " ".join(filtered_sentences)


def filter_sentences_on_answer(dataset, dataset_type="train", get_impossible=False):
    """
    Filter the paragraph with only sentences relevant to answer and generates files
    with sentences and questions instead of paragraphs and questions
    Input:
    dataset: string
    dataset_type: string
    get_impossible: boolean
    """
    if not os.path.exists(os.path.join(OUTPUT_PATH, SQUAD_NAME)):
        os.makedirs(os.path.join(OUTPUT_PATH, SQUAD_NAME))

    para_output = open(
        os.path.join(OUTPUT_PATH, SQUAD_NAME, dataset_type + ".paragraphs"), "w"
    )
    question_output = open(
        os.path.join(OUTPUT_PATH, SQUAD_NAME, dataset_type + ".questions"), "w"
    )
    dataset = dataset["data"]
    dataset_size = []

    logger.debug("Starting to filter sentences on answer")

    # This loops iterates over every paragraph
    for paragraphs in tqdm(dataset):
        paragraphs = paragraphs["paragraphs"]
        for i, paragraph in enumerate(paragraphs):
            para = paragraph["context"]
            # This loop iterates over every question in para
            for questionanswers in paragraph["qas"]:
                if questionanswers["is_impossible"]:
                    continue
                question = questionanswers["question"]

                filtered_sentences = extract_filtered_sentences(questionanswers, para)

                para_output.write(filtered_sentences.strip().lower() + "\n")
                question_output.write(question.strip().lower() + "\n")

                dataset_size.append(i)

    logger.info("Size of the {} dataset: {}".format(dataset_type, len(dataset_size)))
    para_output.close()
    question_output.close()

    logger.debug("Sentences Filtered on Answers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to Preprocess datasets currently available datasets: {}".format(
            ",".join(DATASETS.keys())
        )
    )
    parser.add_argument("-d", "--dataset", default="SQUAD", help="Name of Dataset")
    parser.add_argument(
        "-m", "--mode", default="QUESTION", help="Split on ANSWER or QUESTION"
    )
    parser.add_argument(
        "-f",
        "--filter",
        action="store_true",
        default=False,
        help="filter the sentences on answers",
    )

    args = parser.parse_args()

    assert args.dataset.upper() in DATASETS.keys(), "Invalid Dataset Selected"

    if args.dataset.upper() == "SQUAD":
        preprocess_squad(args.dataset.upper(), args.mode, args.filter)
