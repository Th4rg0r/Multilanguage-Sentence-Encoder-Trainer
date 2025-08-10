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
import json

def save_all_trials_callback(study, trial):
    # Create a list to hold all trial data
    trials_data = []

    # study.trials gives you a list of all completed trials so far
    for t in study.trials:
        trial_data = {
            "trial_number": t.number,
            "value": t.value,  # The loss or accuracy
            "state": t.state.name, # e.g., 'COMPLETE', 'PRUNED', 'FAIL'
            "params": t.params,
            "intermediate_values": t.intermediate_values
        }
        trials_data.append(trial_data)

    # Write the list of trial data to a JSON file
    # This will overwrite the file after each trial with the complete history
    with open("all_trials.json", "w") as f:
        json.dump(trials_data, f, indent=4)

    print(f"Trial {trial.number} finished. All trial data saved to all_trials.json")


def train_and_validate(
    epochs,
    vocab_size,
    tokenizer,
    optimizer_name,
    lr,
    lazy_train_loader,
    lazy_eval_loader,
    d_model,
    n_head,
    num_layers,
    dim_feed_forward,
    dropout,
    trial=None,
    max_batches=None,
    max_eval_batches=None,
    reload_model=False,
    save_model=False,
    model_path="./models/model.pt",
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MissingFinder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        dim_feed_forward=dim_feed_forward,
        dropout=dropout,
    )
    model.to(device)
    
    optimizer = None
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
   
    
    if reload_model and os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    pad_id = tokenizer.token_to_id("<pad>")
    mask_id = tokenizer.token_to_id("<mask>")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    print("start training")
    loss_history = []
    eval_loss_history = []
    min_eval_loss = 1000
    last_eval_loss = None
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        with alive_bar(max_batches) as bar:
            train_loader = lazy_train_loader.loader()
            for idx, batch in enumerate(train_loader):
                if max_batches and idx >= max_batches:
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
        model.eval();
        print("evaluation:")
        with torch.no_grad(), alive_bar(max_eval_batches) as bar:
            test_loader = lazy_eval_loader.loader()
            for idx, batch in enumerate(train_loader):
                if max_eval_batches and idx >= max_eval_batches:
                    break
                batch_count += 1
                batch = lazy_eval_loader.collate_fn(batch)
                src_batch, mask_batch, labels_batch = batch

                src_batch = src_batch.to(device)
                mask_batch = mask_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(src_batch, mask_batch)
                #loss = criterion(outputs, labels_batch)
                masked_indices = (src_batch == mask_id).nonzero(as_tuple=True)
                masked_logits = outputs[masked_indices]
                loss = criterion(masked_logits, labels_batch)
                epoch_loss += loss.item()
                bar.text("Eval Loss: " + str(loss.item()))
                bar()
        avg_eval_loss = epoch_loss / batch_count
        last_eval_loss = avg_eval_loss
                
        print(f"average loss: {avg_loss} | evaluation Loss:{avg_eval_loss}")
        loss_history.append(avg_loss)
        eval_loss_history.append(avg_eval_loss)
        if avg_eval_loss < min_eval_loss:
            min_eval_loss = avg_eval_loss
            if save_model:
                torch.save(model.state_dict(), model_path)
        if trial:
            trial.report(last_eval_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
 
    print(loss_history)
    print(eval_loss_history)
        
    return min_eval_loss
    
def objective(trial, vocab_size, tokenizer, lazy_train_loader, lazy_eval_loader, epochs):
    evaluation_loss =  train_and_validate(
        epochs=epochs,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        optimizer_name=trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        lr=trial.suggest_float("lr",1e-6, 1e-3, log=True),
        lazy_train_loader = lazy_train_loader,
        lazy_eval_loader = lazy_eval_loader,
        d_model=trial.suggest_int("d_model", 256, 512,step=32),
        n_head=trial.suggest_categorical("nhead",[4,8,16]),
        num_layers=trial.suggest_int("num_layers", 4, 8, step=2),
        dim_feed_forward=trial.suggest_int("dim_feed_forward", 512, 2048, step=512),
        dropout=trial.suggest_float("dropout", 0.2,0.5),
        trial=trial,
        max_batches=1000,
        max_eval_batches=250,
    )
    return evaluation_loss

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
    parser.add_argument("--optimize", action="store_true", help="uses optuna to optimize parameters")
    parser.add_argument("--optimizer_runs", type=int, default=100, help="the amount of optimzer runs if optmize is enablede is enabled")
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

    lazy_eval_loader = LazyLoader(
        tokenizer=tokenizer,
        file_path=eval_path,
        batch_size=batch_size,
        max_word_per_sentence=max_word_per_sentence)

    if args.optimize:
        storage_name = "sqlite:///my_study.db"
        study_name = "transformer-optimization-v1"  # Give your study a name

        study = optuna.create_study( 
            storage=storage_name,
            study_name = study_name,
            direction="minimize", 
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial,vocab_size, tokenizer, lazy_train_loader, lazy_eval_loader, args.epochs) , n_trials=args.optimizer_runs, callbacks=[save_all_trials_callback])
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        with open("best_trial.txt", "w") as f:
            f.write("Best trial:\n")
            f.write(f"  Value: {trial.value}\n")
            f.write("  Params: \n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
        # loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)
        # print(loaded_study.best_params)


    else:
        optimizer = "Adam"
        d_model=512
        n_head=8
        num_layers=6
        dim_feed_forward=2048
        dropout=0.15
        #max_batches = args.max_batches
        max_eval_batches = max_batches*0.2
        lr=1e-5

        best_loss = train_and_validate(
            epochs=args.epochs,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            optimizer_name=optimizer,
            lr=lr,
            lazy_train_loader=lazy_train_loader,
            lazy_eval_loader=lazy_eval_loader,
            d_model=d_model,
            n_head=n_head,
            num_layers=num_layers,
            dim_feed_forward=dim_feed_forward,
            dropout=dropout,
            max_batches=max_batches,
            max_eval_batches=max_eval_batches,
            reload_model=True,
            save_model=True,
        )
        print(f"best_loss:{best_loss}")
    

if __name__ == "__main__":
    main()
