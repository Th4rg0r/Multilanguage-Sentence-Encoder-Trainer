from alive_progress import alive_bar
from dataset import LazyLoader, split_train_test_set, get_file_line_cnt 
from info_nce_loss import InfoNCELoss
from network import PositionalEncoding, Encoder, MissingFinder, mean_pooling
from tokenizer import tokenize
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_
import argparse
import json
import optuna
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math

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

def fine_tune_and_validate(
        epochs,
        tokenizer,
        optimizer_name,
        lr,
        lazy_finetune_train_loader,
        lazy_finetune_eval_loader,
        model,
        output_path,
        max_batches=None,
        max_eval_batches=None,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = None
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    pad_id = tokenizer.token_to_id("<pad>")
    criterion = InfoNCELoss(temperature=0.07)
    model.train()
    print("start finetune training")
    loss_history = []
    eval_loss_history = []
    min_eval_loss = 1000
    last_eval_loss = None
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        with alive_bar(max_batches) as bar:
            train_loader = lazy_finetune_train_loader.loader()
            for idx, batch in enumerate(train_loader):
                if max_batches and idx >= max_batches:
                    break;
                batch_count += 1
                batch = lazy_finetune_train_loader.collate_fn(batch)
                src_batch, mask_batch = batch
                src_batch= src_batch.to(device)
                mask_batch = mask_batch.to(device)
                optimizer.zero_grad()
                
                z_i = model(src_batch, mask_batch)
                z_j = model(src_batch, mask_batch)
                z_i = mean_pooling(z_i, mask_batch)
                z_j = mean_pooling(z_j, mask_batch)
                loss  = criterion(z_i, z_j)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                bar.text("Loss: " + str(loss.item()))
                bar()
        if not max_batches:
            max_batches = batch_count
        avg_loss = epoch_loss /batch_count
        print(f"average loss: {avg_loss}")
        loss_history.append(avg_loss)
        model.eval()
        epoch_loss = 0
        batch_count = 0
        print("evaluation:")
        with torch.no_grad(), alive_bar(max_eval_batches) as bar:
            test_loader = lazy_finetune_eval_loader.loader()
            for idx, batch in enumerate(test_loader):
                if max_batches and idx >= max_eval_batches:
                    break;
                batch_count += 1
                batch = lazy_finetune_train_loader.collate_fn(batch)
                src_batch, mask_batch = batch
                src_batch= src_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                z_i = model(src_batch, mask_batch)
                z_j = model(src_batch, mask_batch)
                z_i = mean_pooling(z_i, mask_batch)
                z_j = mean_pooling(z_j, mask_batch)
                loss  = criterion(z_i, z_j)
                epoch_loss += loss.item()
                bar.text("Eval Loss: " + str(loss.item()))
                bar()
        if not max_eval_batches:
            max_eval_batches = batch_count
        avg_eval_loss = epoch_loss /batch_count
        last_eval_loss = avg_eval_loss

        if avg_eval_loss < min_eval_loss:
            min_eval_loss = avg_eval_loss;
            torch.save(model.state_dict(), output_path)

        print(f"average eval loss: {avg_eval_loss}")
        loss_history.append(avg_loss)
        return min_eval_loss

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
    accumulation_steps = 4
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
                epoch_loss += loss.item() 
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                if (idx + 1) % accumulation_steps  == 0:
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                bar.text("Loss: " + str(loss.item()))
                bar()
        if not max_batches:
            max_batches = batch_count
        avg_loss = epoch_loss / batch_count
        print(f"average loss: {avg_loss}")
        loss_history.append(avg_loss)
        model.eval();
        print("evaluation:")
        epoch_loss = 0
        batch_count = 0
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
    
def objective(trial, vocab_size, tokenizer, lazy_train_loader, lazy_eval_loader, epochs, max_batches=1000, max_eval_batches=250):
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
        max_batches=max_batches,
        max_eval_batches=max_eval_batches,
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
    parser.add_argument("--finetune", action="store_true", help="Finetune the saved model with Contrastive learning")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="the learning rate",
    )
    parser.add_argument("--max_batches", type=int, default=None, help="the maximum amount of batches for the train  loop per epoch")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="the maximum amount of batches for the evaluation  loop per epoch")

    args = parser.parse_args()
    project_dir = "."
    vocab_size = 35000
    batch_size = 32
    batch_size_factor = 4
    small_batch_size = 8
    max_word_per_sentence = 10000
    freeze_top_encoder_layer_ratio = 0.25
    max_batches = args.max_batches
    max_eval_batches = args.max_eval_batches
    out_dir = os.path.join(project_dir, "out")
    out_config_path = os.path.join(out_dir, "model.cfg")
    models_dir = os.path.join(project_dir, "models")
    model_path = os.path.join(models_dir, "model.pt")
    result_model_path = os.path.join(out_dir, "model.pt")
    data_dir = os.path.join(project_dir, "data")
    train_path = os.path.join(data_dir, "train.txt")
    eval_path = os.path.join(data_dir, "eval.txt")
    test_path = os.path.join(data_dir, "test.txt")
    tokenizer_path = os.path.join(project_dir, "tokenizer.json")

    os.makedirs(project_dir, exist_ok=True)
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

    if not max_batches:
        max_batches = int(math.ceil(get_file_line_cnt(train_path)/small_batch_size))
    if not max_eval_batches:
        max_eval_batches = int(math.ceil(get_file_line_cnt(eval_path)/small_batch_size))

    lazy_train_loader = LazyLoader(
        tokenizer=tokenizer,
        file_path=train_path,
        batch_size=small_batch_size,
        max_word_per_sentence=max_word_per_sentence,
    )

    lazy_eval_loader = LazyLoader(
        tokenizer=tokenizer,
        file_path=eval_path,
        batch_size=small_batch_size,
        max_word_per_sentence=max_word_per_sentence)

    storage_database_name = "my_study.db"
    storage_name = "sqlite:///my_study.db"
    study_name = "transformer-optimization-v1"  # Give your study a name
    if args.optimize:
        study = optuna.create_study( 
            storage=storage_name,
            study_name = study_name,
            direction="minimize", 
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial,vocab_size, tokenizer, lazy_train_loader, lazy_eval_loader, args.epochs,max_batches, max_eval_batches) , n_trials=args.optimizer_runs, callbacks=[save_all_trials_callback])
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


    else:
        optimizer = "Adam"
        d_model=512
        n_head=8
        num_layers=6
        dim_feed_forward=2048
        dropout=0.15
        lr=1e-5

        if os.path.exists(storage_database_name):
            loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)
            params = loaded_study.best_params
            optimizer = params["optimizer"]
            d_model = params["d_model"]
            n_head = params["nhead"]
            num_layers = params["num_layers"]
            dim_feed_forward=params["dim_feed_forward"]
            dropout = params["dropout"]
            lr = params["lr"]
            if args.finetune:
                # finetune model with a lesser learning rate
                lr = lr/10;
                with open(out_config_path, "w") as f:
                    json.dump(params, f, indent=4)

        if args.finetune:
            lazy_finetune_train_loader = LazyLoader(
                tokenizer=tokenizer,
                file_path=train_path,
                batch_size=batch_size,
                max_word_per_sentence=max_word_per_sentence,
                contrastive_learning=True
            )
            lazy_finetune_eval_loader = LazyLoader(
                tokenizer=tokenizer,
                file_path=eval_path,
                batch_size=batch_size,
                max_word_per_sentence=max_word_per_sentence,
                contrastive_learning=True,
            )
            if not os.path.exists(model_path) and not os.path.exists(result_model_path):
                print("please pretrain the model without the \"--finetune\" parameter before finetuning")
                return
            model = MissingFinder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_head=n_head,
                num_layers=num_layers,
                dim_feed_forward=dim_feed_forward,
                dropout=dropout,
            )

             
            if os.path.exists(result_model_path):
                print("reloading finetuned model")
                model = model.encoder
                state_dict = torch.load(result_model_path, weights_only=True)
                model.load_state_dict(state_dict)
            else:
                print("reloading pretrained model")
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                model = model.encoder
                
            model.to(device)
            # Freeze the entire embedding layer
            for param in model.embedding.parameters():
                param.requires_grad = False

            # You can tune this number.
            num_layers_to_freeze = int(math.ceil(gnum_layers*freeze_top_encoder_layer_ratio))
            for i in range(num_layers_to_freeze):
                for param in model.encoder.layers[i].parameters():
                    param.requires_grad = False

            best_loss = fine_tune_and_validate(
                epochs=args.epochs,
                tokenizer=tokenizer,
                optimizer_name = optimizer,
                lr=lr,
                lazy_finetune_train_loader=lazy_finetune_train_loader,
                lazy_finetune_eval_loader=lazy_finetune_eval_loader,
                model=model,
                output_path=result_model_path,
                max_batches=max_batches,
                max_eval_batches=max_eval_batches,
            )
            print(f"fine_tune best loss:{best_loss}")

        else:
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
