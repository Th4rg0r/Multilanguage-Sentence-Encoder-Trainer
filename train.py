import yaml
import argparse
import os
import torch
import torch.nn as nn
import optuna
import json
import math
from alive_progress import alive_bar
from torch.nn.utils import clip_grad_norm_

from tokenizers import Tokenizer
from tokenizer import tokenize
from dataset import LazyLoader, split_train_test_set
from network import MissingFinder, Encoder, mean_pooling
from info_nce_loss import InfoNCELoss

# -------------------
# Callbacks & Helpers
# -------------------

def save_all_trials_callback(study, trial):
    """Callback to save all Optuna trials to a JSON file after each trial."""
    trials_data = [{
        "trial_number": t.number,
        "value": t.value,
        "state": t.state.name,
        "params": t.params,
        "intermediate_values": t.intermediate_values
    } for t in study.trials]
    with open("all_trials.json", "w") as f:
        json.dump(trials_data, f, indent=4)
    print(f"\nTrial {trial.number} finished. All trial data saved to all_trials.json")

def get_optimizer(optimizer_name, parameters, lr):
    """Creates an optimizer instance from a name string."""
    if optimizer_name == "Adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(parameters, lr=lr)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# -------------------
# Core Training & Evaluation Logic
# -------------------

def run_pretraining_epoch(model, loader, criterion, optimizer, device, config):
    """Runs a single epoch of masked language model pre-training."""
    model.train()
    total_loss = 0
    max_batches = config.get('max_train_batches', -1)
    accumulation_steps = config.get('accumulation_steps', 1)
    
    with alive_bar(max_batches if max_batches != -1 else None) as bar:
        for i, batch in enumerate(loader):
            if max_batches != -1 and i >= max_batches:
                break
            
            src_batch, mask_batch, labels_batch = batch
            src_batch, mask_batch, labels_batch = src_batch.to(device), mask_batch.to(device), labels_batch.to(device)

            outputs = model(src_batch, mask_batch)
            masked_indices = (src_batch == model.tokenizer_mask_id).nonzero(as_tuple=True)
            masked_logits = outputs[masked_indices]
            
            unscaled_loss = criterion(masked_logits, labels_batch)
            total_loss += unscaled_loss.item()
            
            scaled_loss = unscaled_loss / accumulation_steps
            scaled_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            bar.text(f"Loss: {unscaled_loss.item():.4f}")
            bar()
    
    return total_loss / (i + 1)

def run_finetuning_epoch(model, loader, criterion, optimizer, device, config):
    """Runs a single epoch of contrastive learning fine-tuning."""
    model.train()
    total_loss = 0
    max_batches = config.get('max_train_batches', -1)
    
    with alive_bar(max_batches if max_batches != -1 else None) as bar:
        for i, batch in enumerate(loader):
            if max_batches != -1 and i >= max_batches:
                break
            
            src_batch, mask_batch = batch
            src_batch, mask_batch = src_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()

            z_i_tokens = model(src_batch, mask_batch)
            z_j_tokens = model(src_batch, mask_batch)

            z_i = mean_pooling(z_i_tokens, mask_batch)
            z_j = mean_pooling(z_j_tokens, mask_batch)

            loss = criterion(z_i, z_j)
            total_loss += loss.item()
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            bar.text(f"Loss: {loss.item():.4f}")
            bar()
            
    return total_loss / (i + 1)

def evaluate(model, loader, criterion, device, config, is_finetune_eval):
    """Evaluates the model on the evaluation set."""
    model.eval()
    total_loss = 0
    max_batches = config.get('max_eval_batches', -1)
    
    with torch.no_grad(), alive_bar(max_batches if max_batches != -1 else None) as bar:
        for i, batch in enumerate(loader):
            if max_batches != -1 and i >= max_batches:
                break
            
            src_batch, mask_batch, labels_batch = batch if not is_finetune_eval else (*batch, None)
            src_batch, mask_batch = src_batch.to(device), mask_batch.to(device)

            if is_finetune_eval:
                z_i_tokens = model(src_batch, mask_batch)
                z_j_tokens = model(src_batch, mask_batch)
                z_i = mean_pooling(z_i_tokens, mask_batch)
                z_j = mean_pooling(z_j_tokens, mask_batch)
                loss = criterion(z_i, z_j)
            else:
                labels_batch = labels_batch.to(device)
                outputs = model(src_batch, mask_batch)
                masked_indices = (src_batch == model.tokenizer_mask_id).nonzero(as_tuple=True)
                masked_logits = outputs[masked_indices]
                loss = criterion(masked_logits, labels_batch)
            
            total_loss += loss.item()
            bar.text(f"Eval Loss: {loss.item():.4f}")
            bar()
            
    return total_loss / (i + 1)

# -------------------
# Main Workflows
# -------------------

def get_model_params_from_config(config, study_name, storage_name):
    """Gets model parameters from config, either from custom or Optuna best."""
    if config['override_with_custom_params']['enabled']:
        print("Using custom parameters from config file.")
        return config['override_with_custom_params']
    else:
        print("Loading best parameters from Optuna study.")
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            return study.best_params
        except KeyError:
            print(f"Warning: Optuna study '{study_name}' not found. Falling back to custom parameters.")
            return config['override_with_custom_params']

def run_training(config, start_from_scratch=False):
    """Main workflow for pre-training the model."""
    print("--- Starting Model Pre-Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Configs ---
    data_cfg = config['data']
    tokenizer_cfg = config['tokenizer']
    train_cfg = config['train']
    optimize_cfg = config['optimize']
    
    # --- Get Model Parameters ---
    model_params = get_model_params_from_config(train_cfg, optimize_cfg['study_name'], optimize_cfg['storage_name'])

    # --- Load Tokenizer & Dataloaders ---
    tokenizer = Tokenizer.from_file(os.path.join(data_cfg['project_dir'], tokenizer_cfg['save_path']))
    train_loader = LazyLoader(tokenizer, os.path.join(data_cfg['project_dir'], data_cfg['train_path']), train_cfg['batch_size'], data_cfg['max_word_per_sentence'])
    eval_loader = LazyLoader(tokenizer, os.path.join(data_cfg['project_dir'], data_cfg['eval_path']), train_cfg['batch_size'], data_cfg['max_word_per_sentence'])

    # --- Model ---
    model = MissingFinder(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=model_params['embedding_dim'],
        num_attention_heads=model_params['num_attention_heads'],
        num_encoder_layers=model_params['num_encoder_layers'],
        feed_forward_dim=model_params['feed_forward_dim'],
        dropout=model_params['dropout']
    )
    model.tokenizer_mask_id = tokenizer.token_to_id("<mask>")
    model.to(device)

    model_save_path = os.path.join(data_cfg['project_dir'], train_cfg['model_save_path'])
    if not start_from_scratch and os.path.exists(model_save_path):
        print(f"Resuming training from saved model: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
    else:
        print("Starting training from scratch.")

    # --- Optimizer, Criterion & Scheduler ---
    optimizer = get_optimizer(model_params['optimizer_name'], model.parameters(), model_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        factor=train_cfg['scheduler_factor'], 
        patience=train_cfg['scheduler_patience'], 
        verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    # --- Training Loop ---
    best_eval_loss = float('inf')
    for epoch in range(train_cfg['epochs']):
        print(f"\n--- Epoch {epoch+1}/{train_cfg['epochs']} ---")
        avg_train_loss = run_pretraining_epoch(model, train_loader.loader(), criterion, optimizer, device, train_cfg)
        avg_eval_loss = evaluate(model, eval_loader.loader(), criterion, device, train_cfg, is_finetune_eval=False)
        
        print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Eval Loss: {avg_eval_loss:.4f}")

        scheduler.step(avg_eval_loss)

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"New best evaluation loss: {best_eval_loss:.4f}. Saving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
            # Save config used for this model
            config_save_path = os.path.join(data_cfg['project_dir'], train_cfg['config_save_path'])
            with open(config_save_path, 'w') as f:
                yaml.dump({'model_params': model_params}, f, indent=4)


def run_finetuning(config, start_from_scratch=False):
    """Main workflow for fine-tuning the model with contrastive loss."""
    print("--- Starting Model Fine-Tuning ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Configs ---
    data_cfg = config['data']
    tokenizer_cfg = config['tokenizer']
    train_cfg = config['train']
    finetune_cfg = config['finetune']
    optimize_cfg = config['optimize']

    # --- Get Model & Finetune Parameters ---
    base_model_params = get_model_params_from_config(train_cfg, optimize_cfg['study_name'], optimize_cfg['storage_name'])
    finetune_hyperparams = get_model_params_from_config(finetune_cfg, optimize_cfg['study_name'], optimize_cfg['storage_name'])

    # --- Load Tokenizer & Dataloaders ---
    tokenizer = Tokenizer.from_file(os.path.join(data_cfg['project_dir'], tokenizer_cfg['save_path']))
    train_loader = LazyLoader(tokenizer, os.path.join(data_cfg['project_dir'], data_cfg['train_path']), finetune_cfg['batch_size'], data_cfg['max_word_per_sentence'], contrastive_learning=True)
    eval_loader = LazyLoader(tokenizer, os.path.join(data_cfg['project_dir'], data_cfg['eval_path']), finetune_cfg['batch_size'], data_cfg['max_word_per_sentence'], contrastive_learning=True)

    # --- Model Instantiation ---
    model_to_load = MissingFinder(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=base_model_params['embedding_dim'],
        num_attention_heads=base_model_params['num_attention_heads'],
        num_encoder_layers=base_model_params['num_encoder_layers'],
        feed_forward_dim=base_model_params['feed_forward_dim'],
        dropout=base_model_params['dropout']
    )
    
    pre_trained_path = os.path.join(data_cfg['project_dir'], train_cfg['model_save_path'])
    finetuned_path = os.path.join(data_cfg['project_dir'], finetune_cfg['model_save_path'])

    # --- Model Loading Logic for --new flag ---
    if not start_from_scratch and os.path.exists(finetuned_path):
        print(f"Resuming fine-tuning from previously fine-tuned model: {finetuned_path}")
        model_to_load.load_state_dict(torch.load(finetuned_path))
    elif os.path.exists(pre_trained_path):
        print(f"Starting new fine-tuning session from pre-trained model: {pre_trained_path}")
        model_to_load.load_state_dict(torch.load(pre_trained_path))
    else:
        print(f"Error: Pre-trained model not found at {pre_trained_path}. Please run pre-training first.")
        return

    # We only fine-tune the encoder part
    model = model_to_load.encoder
    model.to(device)

    # --- Freeze Layers ---
    num_layers = len(model.encoder.layers)
    num_layers_to_freeze = int(num_layers * finetune_cfg['freeze_layer_ratio'])
    print(f"Freezing the bottom {num_layers_to_freeze} out of {num_layers} encoder layers.")
    
    for param in model.embedding.parameters():
        param.requires_grad = False
    for i in range(num_layers_to_freeze):
        for param in model.encoder.layers[i].parameters():
            param.requires_grad = False

    # --- Optimizer, Criterion & Scheduler ---
    optimizer = get_optimizer(
        finetune_hyperparams['optimizer_name'], 
        filter(lambda p: p.requires_grad, model.parameters()),
        finetune_hyperparams['learning_rate']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=finetune_cfg['scheduler_factor'],
        patience=finetune_cfg['scheduler_patience'],
        verbose=True
    )
    criterion = InfoNCELoss(temperature=0.07)

    # --- Fine-Tuning Loop ---
    best_eval_loss = float('inf')
    for epoch in range(finetune_cfg['epochs']):
        print(f"\n--- Finetune Epoch {epoch+1}/{finetune_cfg['epochs']} ---")
        avg_train_loss = run_finetuning_epoch(model, train_loader.loader(), criterion, optimizer, device, finetune_cfg)
        avg_eval_loss = evaluate(model, eval_loader.loader(), criterion, device, finetune_cfg, is_finetune_eval=True)
        
        print(f"Finetune Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Eval Loss: {avg_eval_loss:.4f}")

        scheduler.step(avg_eval_loss)

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"New best evaluation loss: {best_eval_loss:.4f}. Saving finetuned model to {finetuned_path}")
            torch.save(model.state_dict(), finetuned_path)




def run_optimization(config, start_new_study=False):
    """Main workflow for hyperparameter optimization with Optuna."""
    # ... (implementation from previous step) ...

# -------------------
# Main Entry Point
# -------------------

def main():
    """Main function to drive the script."""
    parser = argparse.ArgumentParser(description="Multilanguage Sentence Encoder Trainer")
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization with Optuna.')
    parser.add_argument('--finetune', action='store_true', help='Run contrastive learning fine-tuning on a pre-trained model.')
    parser.add_argument('--new', action='store_true', help='Start a new study or train from scratch, ignoring previous state.')
    parser.add_argument('--setup_data', action='store_true', help='Run the initial data setup (splitting and de-duplicating).')
    parser.add_argument('--setup_tokenizer', action='store_true', help='(Re)create the tokenizer.')
    args = parser.parse_args()

    # --- Load Config ---
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    data_cfg = config['data']
    tokenizer_cfg = config['tokenizer']
    project_dir = os.getcwd()
    data_cfg['project_dir'] = project_dir # Add project dir to config for easy path joining

    # --- Initial Setup ---
    if args.setup_data:
        print("--- Running Initial Data Setup ---")
        split_train_test_set(
            os.path.join(project_dir, data_cfg['input_path']),
            os.path.join(project_dir, 'data'),
            data_cfg['test_size']
        )
        return # Exit after setup

    if args.setup_tokenizer:
        print("--- (Re)creating Tokenizer ---")
        tokenizer = tokenize(os.path.join(project_dir, data_cfg['train_path']), tokenizer_cfg)
        tokenizer.save(os.path.join(project_dir, tokenizer_cfg['save_path']))
        print(f"Tokenizer saved to {tokenizer_cfg['save_path']}")
        return # Exit after setup

    # --- Main Action ---
    if args.optimize:
        run_optimization(config, start_new_study=args.new)
    elif args.finetune:
        run_finetuning(config, start_from_scratch=args.new) # Note: --new for finetuning is not implemented yet
    else:
        run_training(config, start_from_scratch=args.new)

if __name__ == "__main__":
    main()
