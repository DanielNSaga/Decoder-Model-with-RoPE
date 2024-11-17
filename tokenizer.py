# tokenizer.py

from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


def get_training_corpus(dataset):
    """
    Generator function to yield batches of texts for tokenizer training.

    Args:
        dataset: The dataset containing the 'train' split with 'maintext' field.

    Yields:
        Iterator over batches of texts.
    """
    for i in range(0, len(dataset["train"]), 1000):
        yield dataset["train"][i: i + 1000]["maintext"]


def train_tokenizer():
    """
    Trains a Byte-Pair Encoding (BPE) tokenizer on the English Wikipedia dataset.
    """
    # Load the English Wikipedia dataset
    ds = load_dataset("lucadiliello/english_wikipedia")

    # Initialize the tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Define the normalization and pre-tokenization steps
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Define special tokens
    special_tokens = ["[UNK]", "[BOS]", "[EOS]", "[PAD]"]

    # Set up the trainer with vocabulary size and special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=32768,
        min_frequency=2,
        special_tokens=special_tokens
    )

    # Train the tokenizer on the dataset
    tokenizer.train_from_iterator(get_training_corpus(ds), trainer=trainer)

    # Set the post-processor and decoder
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    # Save the tokenizer to a file
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved to tokenizer.json")


if __name__ == "__main__":
    train_tokenizer()
