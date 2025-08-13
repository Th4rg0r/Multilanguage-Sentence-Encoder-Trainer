
import yaml
import argparse
import os
import torch
from tokenizers import Tokenizer
import torch.nn.functional as F
from alive_progress import alive_bar

def load_config():
    """Loads the main configuration file."""
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

EMBEDDING_CHUNK_SIZE = 10000 # Process and save embeddings in chunks of this size

def generate_and_save_embeddings_in_chunks(config, model, tokenizer):
    """
    Generates sentence embeddings in chunks and saves them to separate files.
    """
    data_cfg = config['data']
    input_file_path = os.path.join(data_cfg['project_dir'], data_cfg['input_path'])
    
    EMBEDDING_CHUNKS_DIR = os.path.join(data_cfg['project_dir'], 'data/embedding_chunks')
    os.makedirs(EMBEDDING_CHUNKS_DIR, exist_ok=True)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Generating and saving embeddings in chunks to {EMBEDDING_CHUNKS_DIR}...")
    
    batch_size = 64 # Batch size for processing sentences within a chunk
    all_chunk_paths = []
    
    num_sentences = len(sentences)
    num_chunks = (num_sentences + EMBEDDING_CHUNK_SIZE - 1) // EMBEDDING_CHUNK_SIZE

    with alive_bar(num_chunks, title="Processing embedding chunks") as bar:
        for chunk_idx in range(num_chunks):
            start_sentence_idx = chunk_idx * EMBEDDING_CHUNK_SIZE
            end_sentence_idx = min((chunk_idx + 1) * EMBEDDING_CHUNK_SIZE, num_sentences)
            
            chunk_sentences = sentences[start_sentence_idx:end_sentence_idx]
            
            chunk_embeddings_list = []
            for i in range(0, len(chunk_sentences), batch_size):
                batch_sentences = chunk_sentences[i:i+batch_size]
                embeddings = get_sentence_embedding(batch_sentences, model, tokenizer)
                chunk_embeddings_list.append(embeddings)
            
            chunk_embeddings_tensor = torch.cat(chunk_embeddings_list, dim=0)
            
            chunk_file_path = os.path.join(EMBEDDING_CHUNKS_DIR, f'chunk_{chunk_idx}.pt')
            torch.save(chunk_embeddings_tensor.cpu(), chunk_file_path) # Save to CPU to avoid GPU memory issues
            all_chunk_paths.append(chunk_file_path)
            bar()

            embeddings = get_sentence_embedding(batch_sentences, model, tokenizer)
            chunk_embeddings_list.append(embeddings)
            bar()

    
    
    return sentences, all_chunk_paths

def get_sentence_embedding(sentences, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Generates embeddings for a list of sentences."""
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
    encoded_input = tokenizer.encode_batch(sentences)

    input_ids = torch.tensor([e.ids for e in encoded_input], dtype=torch.long)
    attention_mask = torch.tensor([e.attention_mask for e in encoded_input], dtype=torch.bool)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        embeddings = model(input_ids, attention_mask)
    
    return embeddings

def semantic_search(query_sentence, model, tokenizer, all_sentences, embedding_chunk_paths, top_k=30):
    """
    Performs semantic search to find the most similar sentences to a query sentence
    by processing embeddings in chunks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_embedding = get_sentence_embedding([query_sentence], model, tokenizer)
    query_embedding_norm = F.normalize(query_embedding, p=2, dim=1) # Normalize query once

    # Initialize lists to store top results across all chunks
    all_top_k_scores = []
    all_top_k_indices = [] # These will be global indices relative to all_sentences

    current_sentence_offset = 0

    print(f"\n--- Searching for: \"{query_sentence}\" ---")
    with alive_bar(len(embedding_chunk_paths), title="Searching embedding chunks") as bar:
        for chunk_idx, chunk_path in enumerate(embedding_chunk_paths):
            chunk_embeddings = torch.load(chunk_path, map_location=device) # Load chunk to device
            chunk_embeddings_norm = F.normalize(chunk_embeddings, p=2, dim=1)

            # Calculate cosine similarity for the current chunk
            similarities = torch.mm(query_embedding_norm, chunk_embeddings_norm.transpose(0, 1))

            # Get top_k results for the current chunk
            # We need to get more than top_k from each chunk to ensure overall top_k
            # A heuristic: get top 2*top_k or min(top_k, chunk_size)
            k_in_chunk = min(top_k * 2, chunk_embeddings.size(0))
            if k_in_chunk == 0: # Handle empty chunks
                bar()
                continue

            chunk_top_k_scores, chunk_top_k_local_indices = torch.topk(similarities, k=k_in_chunk, dim=1)

            # Convert local indices to global indices
            chunk_top_k_global_indices = chunk_top_k_local_indices + current_sentence_offset

            all_top_k_scores.append(chunk_top_k_scores.cpu())
            all_top_k_indices.append(chunk_top_k_global_indices.cpu())
            
            current_sentence_offset += chunk_embeddings.size(0)
            bar()

    # Combine results from all chunks and get the overall top_k
    if not all_top_k_scores: # Handle case where no embeddings were processed
        print("No embeddings found to search.")
        return

    combined_scores = torch.cat(all_top_k_scores, dim=1)
    combined_indices = torch.cat(all_top_k_indices, dim=1)

    # Get the final overall top_k
    final_top_k_scores, final_top_k_combined_indices = torch.topk(combined_scores, k=min(top_k, combined_scores.size(1)), dim=1)

    print(f"\n--- Top {min(top_k, combined_scores.size(1))} results for: \"{query_sentence}\" ---")
    for i in range(min(top_k, combined_scores.size(1))):
        score = final_top_k_scores[0][i].item()
        sentence_idx = final_top_k_combined_indices[0][i].item()
        sentence = all_sentences[sentence_idx]
        print(f"  Score: {score:.4f} - \"{sentence}\"")


def load_existing_chunks(config):
    """
    Loads paths to existing embedding chunks and the full list of sentences.
    """
    data_cfg = config['data']
    input_file_path = os.path.join(data_cfg['project_dir'], data_cfg['input_path'])
    EMBEDDING_CHUNKS_DIR = os.path.join(data_cfg['project_dir'], 'data/embedding_chunks')

    with open(input_file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    embedding_chunk_paths = sorted([
        os.path.join(EMBEDDING_CHUNKS_DIR, f)
        for f in os.listdir(EMBEDDING_CHUNKS_DIR) if f.endswith('.pt')
    ])

    return sentences, embedding_chunk_paths

def main():
    """Main function to drive the script."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Perform semantic search using a trained sentence encoder.")
    parser.add_argument("query", type=str, help="The sentence to search for, enclosed in quotes.")
    parser.add_argument("--top_k", type=int, default=10, help="The number of top results to display.")
    parser.add_argument("--new", action="store_true", help="Regenerate embeddings even if they exist.")
    args = parser.parse_args()

    config = load_config()
    
    # --- Load Model and Tokenizer ---
    model_path = config['final_model_path']
    tokenizer_path = os.path.dirname(config["final_tokenizer_path"])

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print(f"Error: Model or tokenizer not found in '{model_dir}'.")
        print("Please run the full training and fine-tuning pipeline first using 'python train.py --finetune'.")
        return

    try:
        print("Loading model and tokenizer...")
        model = torch.load(model_path, map_location=device, weights_only=False)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        model.eval() # Set to evaluation mode
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model or tokenizer: {e}")
        return

    # --- Get Embeddings (chunked) ---
    data_cfg = config['data']
    EMBEDDING_CHUNKS_DIR = os.path.join(data_cfg['project_dir'], 'data/embedding_chunks')
    
    # Check if chunks exist and --new is not used
    if not args.new and os.path.exists(EMBEDDING_CHUNKS_DIR) and os.listdir(EMBEDDING_CHUNKS_DIR):
        print("Found existing embedding chunks. Loading them...")
        all_sentences, embedding_chunk_paths = load_existing_chunks(config)
    else:
        print("Generating new embedding chunks...")
        all_sentences, embedding_chunk_paths = generate_and_save_embeddings_in_chunks(config, model, tokenizer)


    # --- Perform Search (chunked) ---
    semantic_search(args.query, model, tokenizer, all_sentences, embedding_chunk_paths, top_k=args.top_k)

if __name__ == "__main__":
    main()
