# data_prep.py

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import random_split


def tokenize_and_save_datasets():
    """
    Tokenizes the English Wikipedia dataset and saves the tokenized training and validation datasets to disk.
    """
    # Load the English Wikipedia dataset
    ds = load_dataset("lucadiliello/english_wikipedia")

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(ds["train"]))
    val_size = len(ds["train"]) - train_size
    train_dataset, val_dataset = random_split(ds["train"], [train_size, val_size])

    # Load the pre-trained tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )

    def tokenize_function(examples):
        """
        Tokenizes the input texts.

        Args:
            examples: A batch of examples containing 'maintext'.

        Returns:
            A dictionary with tokenized inputs.
        """
        return tokenizer(examples["maintext"], truncation=True, max_length=1024)

    # Convert subsets to datasets and tokenize
    train_dataset = Dataset.from_dict(train_dataset.dataset[train_dataset.indices])
    val_dataset = Dataset.from_dict(val_dataset.dataset[val_dataset.indices])

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["maintext"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["maintext"])

    # Save the tokenized datasets to disk
    train_dataset.save_to_disk("train_dataset")
    val_dataset.save_to_disk("val_dataset")
    print("Datasets saved to 'train_dataset' and 'val_dataset'")


if __name__ == "__main__":
    tokenize_and_save_datasets()
