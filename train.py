from alive_progress import alive_bar
from dataset import LazyLoader, split_train_test_set
from network import PositionalEncoding, Encoder, MissingFinder
from tokenizer import tokenize
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.txt", help="the dataset to train on")
    parser.add_argument(
        "--create_datasets", action="store_true", help="creates the datasets"
    )

    parser.add_argument(
        "--create_tokenizer", action="store_true", help="creates the tokenizer"
    )
    parser.add_argument("--clear_model", action="store_true", help="clears the model")
    parser.add_argument(
        "--epochs", type=int, default=10, help="the number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="the learning rate",
    )

    args = parser.parse_args()
    out_dir = "."
    vocab_size = 35000
    batch_size = 32
    max_word_per_sentence = 10000
    max_batches = 1000
    models_dir = os.path.join(out_dir, "models")
    model_path = os.path.join(models_dir, "model.pt")
    data_dir = os.path.join(out_dir, "data")
    train_path = os.path.join(data_dir, "train.txt")
    eval_path = os.path.join(data_dir, "eval.txt")
    test_path = os.path.join(data_dir, "test.txt")
    tokenizer_path = os.path.join(out_dir, "tokenizer.json")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    tokenizer = None

    if args.create_datasets or not os.path.exists(train_path):
        train_count, test_count, eval_count, train_path, eval_path, test_path = (
            split_train_test_set(args.data, data_dir)
        )

    if args.create_tokenizer or not os.path.exists(tokenizer_path):
        tokenizer = tokenize(train_path, vocab_size)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    lazy_train_loader = LazyLoader(
        tokenizer=tokenizer,
        file_path=train_path,
        batch_size=batch_size,
        max_word_per_sentence=max_word_per_sentence,
    )

    model = MissingFinder(vocab_size=vocab_size)
    model.to(device)
    if os.path.exists(model_path) and not args.clear_model:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    pad_id = tokenizer.token_to_id("<pad>")
    mask_id = tokenizer.token_to_id("<mask>")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    print("start training")

    loss_history = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_count = 0
        with alive_bar(max_batches) as bar:
            train_loader = lazy_train_loader.loader()
            for idx, batch in enumerate(train_loader):
                if idx >= max_batches:
                    break
                batch_count += 1
                batch = lazy_train_loader.collate_fn(batch)
                src_batch, mask_batch, labels_batch = batch

                src_batch = src_batch.to(device)
                mask_batch = mask_batch.to(device)
                labels_batch = labels_batch.to(device)
                optimizer.zero_grad()

                outputs = model(src_batch, mask_batch)
                #loss = criterion(outputs, labels_batch)
                masked_indices = (src_batch == mask_id).nonzero(as_tuple=True)
                masked_logits = outputs[masked_indices]
                loss = criterion(masked_logits, labels_batch)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                bar.text("Loss: " + str(loss.item()))
                bar()
        avg_loss = epoch_loss / batch_count
        print(f"average loss: {avg_loss}")
        loss_history.append(avg_loss)
    print(loss_history)
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
