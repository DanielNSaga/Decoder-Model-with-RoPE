# train.py

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

from model import Model, ModelConfig


def collate_fn(batch):
    """
    Collate function for DataLoader to handle variable-length sequences.

    Args:
        batch: A list of examples from the dataset.

    Returns:
        A dictionary containing padded input_ids and attention_mask tensors.
    """
    # Extract input_ids and attention_mask from the batch
    input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
    attention_mask = [torch.tensor(example["attention_mask"], dtype=torch.long) for example in batch]

    # Pad sequences to the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, checkpoint_path="checkpoint.pth"):
    """
    Training loop for the model.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_epochs: Number of training epochs.
        device: Device to train on ('cuda' or 'cpu').
        checkpoint_path: Path to save the best model checkpoint.
    """
    # Compile the model for optimization (requires PyTorch 2.0 or higher)
    model = torch.compile(model)

    # Check if Automatic Mixed Precision (AMP) can be used
    if torch.cuda.is_available():
        use_amp = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        use_amp = False
        scaler = None

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Prepare inputs and targets
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass with AMP
                    logits, loss = model(inputs, attention_mask=attention_mask, targets=targets)
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                logits, loss = model(inputs, attention_mask=attention_mask, targets=targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training loss: {avg_train_loss}")

        # Step the scheduler
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Prepare inputs and targets
                targets = input_ids[:, 1:].contiguous()
                inputs = input_ids[:, :-1].contiguous()
                attention_mask = attention_mask[:, :-1].contiguous()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits, loss = model(inputs, attention_mask=attention_mask, targets=targets)
                else:
                    logits, loss = model(inputs, attention_mask=attention_mask, targets=targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation loss: {avg_val_loss}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("Saving checkpoint.")


if __name__ == "__main__":
    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )

    # Load the tokenized datasets
    train_dataset = load_from_disk("train_dataset")
    val_dataset = load_from_disk("val_dataset")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

    # Initialize the model with configuration
    config = ModelConfig(pad_token_id=tokenizer.pad_token_id)
    model = Model(config)

    # Configure the optimizer and scheduler
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=3e-4,
        betas=(0.9, 0.95),
        device_type=device.type
    )

    num_epochs = 3
    total_steps = num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Start training
    train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device=device)
