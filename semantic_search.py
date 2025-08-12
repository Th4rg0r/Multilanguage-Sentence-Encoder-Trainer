
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

def get_or_create_embeddings(config, model, tokenizer):
    """
    Loads sentence embeddings from a file if it exists, otherwise creates them
    using the provided model and saves them to a file.
    """
    data_cfg = config['data']
    embeddings_path = os.path.join(data_cfg['project_dir'], 'data/embeddings.pt')
    input_file_path = os.path.join(data_cfg['project_dir'], data_cfg['input_path'])

    if os.path.exists(embeddings_path):
        print(f"Loading existing embeddings from {embeddings_path}...")
        data = torch.load(embeddings_path)
        print("Embeddings loaded.")
        return data['sentences'], data['embeddings']

    print("Embeddings file not found. Creating new embeddings...")
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Generate embeddings in batches to handle large files
    batch_size = 64
    all_embeddings = []
    with alive_bar(20000 // batch_size + 1) as bar:
    #with alive_bar(len(sentences) // batch_size + 1) as bar:
        #for i in range(0, len(sentences), batch_size):
        for i in range(0, 20000, batch_size):
            batch_sentences = sentences[i:i+batch_size]
            embeddings = get_sentence_embedding(batch_sentences, model, tokenizer)
            all_embeddings.append(embeddings)
            bar()

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)

    print(f"Saving embeddings to {embeddings_path}...")
    torch.save({'sentences': sentences, 'embeddings': all_embeddings_tensor}, embeddings_path)
    print("Embeddings saved.")
    
    return sentences, all_embeddings_tensor

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

def semantic_search(query_sentence, model, tokenizer, all_sentences, all_embeddings, top_k=30):
    """
    Performs semantic search to find the most similar sentences to a query sentence.
    """
    query_embedding = get_sentence_embedding([query_sentence], model, tokenizer)
    
    # Normalize embeddings for cosine similarity
    query_embedding_norm = F.normalize(query_embedding, p=2, dim=1)
    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
    
    # Calculate cosine similarity
    similarities = torch.mm(query_embedding_norm, all_embeddings_norm.transpose(0, 1))
    
    # Get the top_k most similar sentences
    top_k_scores, top_k_indices = torch.topk(similarities, k=top_k, dim=1)
    
    print(f"\n--- Top {top_k} results for: \"{query_sentence}\" ---")
    for i in range(top_k):
        score = top_k_scores[0][i].item()
        sentence = all_sentences[top_k_indices[0][i].item()]
        print(f"  Score: {score:.4f} - \"{sentence}\"")

def main():
    """Main function to drive the script."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Perform semantic search using a trained sentence encoder.")
    parser.add_argument("query", type=str, help="The sentence to search for, enclosed in quotes.")
    parser.add_argument("--top_k", type=int, default=30, help="The number of top results to display.")
    args = parser.parse_args()

    config = load_config()
    
    # --- Load Model and Tokenizer ---
    model_dir = os.path.dirname(config['final_model_path'])
    model_path = config['final_model_path']
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')

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

    # --- Get Embeddings ---
    all_sentences, all_embeddings = get_or_create_embeddings(config, model, tokenizer)

    # --- Perform Search ---
    semantic_search(args.query, model, tokenizer, all_sentences, all_embeddings, top_k=args.top_k)

if __name__ == "__main__":
    main()
