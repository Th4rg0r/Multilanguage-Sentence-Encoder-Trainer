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

def semantic_search_rank(query_sentences, original_sentences, model, tokenizer, all_sentences, embedding_chunk_paths):
    """
    Performs semantic search to find the rank of the original sentence for each query.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get embeddings for all query sentences at once
    query_embeddings = get_sentence_embedding(query_sentences, model, tokenizer)
    query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=1)

    # Get embeddings for all original sentences at once
    original_embeddings = get_sentence_embedding(original_sentences, model, tokenizer)
    original_embeddings_norm = F.normalize(original_embeddings, p=2, dim=1)

    ranks = {query: 0 for query in query_sentences}

    print(f"\n--- Evaluating {len(query_sentences)} queries ---")
    with alive_bar(len(embedding_chunk_paths), title="Searching embedding chunks") as bar:
        for chunk_path in embedding_chunk_paths:
            chunk_embeddings = torch.load(chunk_path, map_location=device)
            chunk_embeddings_norm = F.normalize(chunk_embeddings, p=2, dim=1)

            # Calculate cosine similarity for all queries against the current chunk
            similarities = torch.mm(query_embeddings_norm, chunk_embeddings_norm.transpose(0, 1))

            # Calculate cosine similarity for all original sentences against the current chunk
            original_similarities = torch.mm(original_embeddings_norm, chunk_embeddings_norm.transpose(0, 1))

            for i, query in enumerate(query_sentences):
                # Get the similarity of the original sentence for this query
                original_similarity = original_similarities[i].max()

                # Get the similarities of all other sentences for this query
                query_similarity = similarities[i]

                # Calculate the rank of the original sentence
                ranks[query] += (query_similarity > original_similarity).sum().item()

            bar()

    for query, rank in ranks.items():
        print(f"Query: \"{query}\" - Original sentence rank: {rank + 1}")

def semantic_search(query_sentences, model, tokenizer, all_sentences, embedding_chunk_paths, top_k=30):
    """
    Performs semantic search to find the most similar sentences to a list of query sentences
    by processing embeddings in chunks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize dictionaries to store top results for each query
    all_top_k_scores = {query: [] for query in query_sentences}
    all_top_k_indices = {query: [] for query in query_sentences}

    # Get embeddings for all query sentences at once
    query_embeddings = get_sentence_embedding(query_sentences, model, tokenizer)
    query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=1)

    current_sentence_offset = 0

    print(f"\n--- Searching for {len(query_sentences)} queries ---")
    with alive_bar(len(embedding_chunk_paths), title="Searching embedding chunks") as bar:
        for chunk_path in embedding_chunk_paths:
            chunk_embeddings = torch.load(chunk_path, map_location=device)
            chunk_embeddings_norm = F.normalize(chunk_embeddings, p=2, dim=1)

            # Calculate cosine similarity for all queries against the current chunk
            similarities = torch.mm(query_embeddings_norm, chunk_embeddings_norm.transpose(0, 1))

            k_in_chunk = min(top_k * 2, chunk_embeddings.size(0))
            if k_in_chunk == 0:
                bar()
                continue

            chunk_top_k_scores, chunk_top_k_local_indices = torch.topk(similarities, k=k_in_chunk, dim=1)

            # Convert local indices to global indices
            chunk_top_k_global_indices = chunk_top_k_local_indices + current_sentence_offset

            for i, query in enumerate(query_sentences):
                all_top_k_scores[query].append(chunk_top_k_scores[i].cpu())
                all_top_k_indices[query].append(chunk_top_k_global_indices[i].cpu())
            
            current_sentence_offset += chunk_embeddings.size(0)
            bar()

    # Combine results from all chunks and get the overall top_k for each query
    for query in query_sentences:
        if not all_top_k_scores[query]:
            print(f"No embeddings found to search for query: \"{query}\"")
            continue

        combined_scores = torch.cat(all_top_k_scores[query])
        combined_indices = torch.cat(all_top_k_indices[query])

        final_top_k_scores, final_top_k_indices = torch.topk(combined_scores, k=min(top_k, combined_scores.size(0)))

        print(f"\n--- Top {min(top_k, combined_scores.size(0))} results for: \"{query}\" ---")
        for i in range(len(final_top_k_scores)):
            score = final_top_k_scores[i].item()
            sentence_idx = combined_indices[final_top_k_indices[i]].item()
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
    parser.add_argument("query", type=str, nargs='?', default=None, help="The sentence to search for, enclosed in quotes.")
    parser.add_argument("--file", type=str, default=None, help="Path to a text file with queries (one per line).")
    parser.add_argument("--original", type=str, default=None, help="Path to a text file with original sentences (one per line).")
    parser.add_argument("--top_k", type=int, default=10, help="The number of top results to display.")
    parser.add_argument("--new", action="store_true", help="Regenerate embeddings even if they exist.")
    args = parser.parse_args()

    if not args.query and not args.file:
        parser.error("Either a query or a --file must be provided.")

    if args.original and not args.file:
        parser.error("--original can only be used with --file.")

    if args.original and args.top_k != 10:
        parser.error("--top_k cannot be used with --original.")

    config = load_config()
    
    # --- Load Model and Tokenizer ---
    model_path = config['final_model_path']
    tokenizer_path = config["final_tokenizer_path"]

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print(f"Error: Model or tokenizer not found. Please run the full training and fine-tuning pipeline first using 'python train.py --finetune'.")
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


    # --- Perform Search ---
    if args.original:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            with open(args.original, 'r', encoding='utf-8') as f:
                originals = [line.strip() for line in f if line.strip()]
        except FileNotFoundError as e:
            print(f"Error: {e.strerror}: {e.filename}")
            return
        except Exception as e:
            print(f"An error occurred while reading the files: {e}")
            return

        if len(queries) != len(originals):
            print("Error: The number of lines in the query file and the original file must be the same.")
            return

        semantic_search_rank(queries, originals, model, tokenizer, all_sentences, embedding_chunk_paths)
    else:
        queries = []
        if args.query:
            queries.append(args.query)
        
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    queries.extend([line.strip() for line in f if line.strip()])
            except FileNotFoundError:
                print(f"Error: File not found at {args.file}")
                return
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")
                return

        semantic_search(queries, model, tokenizer, all_sentences, embedding_chunk_paths, top_k=args.top_k)

if __name__ == "__main__":
    main()
