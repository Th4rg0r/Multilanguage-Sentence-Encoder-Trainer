from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers, Regex

def sentence_splitter(input_file, output_file):
    pattern = r"(?<=[。！？\.!?…！？؟।:;])\s+"
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = re.split(pattern, text)
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

def tokenize(text_file, config):
    """
    Creates a tokenizer from a text file and configuration.

    Args:
        text_file (str): The path to the text file for training the tokenizer.
        config (dict): A dictionary containing tokenizer settings.

    Returns:
        Tokenizer: The trained tokenizer.
    """
    keep_accents = config.get("keep_accents", True)
    vocab_size = config["vocab_size"]
    keep_punctuation = config.get("keep_punctuation", False)

    # Use Byte-Pair Encoding (BPE)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Build the normalizer sequence based on the configuration
    normalizer_list = []
    if not keep_punctuation:
        # If false, add a normalizer to replace punctuation with an empty string.
        # The \p{P} pattern matches any kind of punctuation character.
        normalizer_list.append(normalizers.Replace(Regex(r'\p{P}'), ''))
    
    # Always apply lowercasing
    normalizer_list.append(normalizers.Lowercase())

    if not keep_accents:
        # If false, add normalizers to decompose and strip accents (e.g., 'é' -> 'e').
        normalizer_list.extend([normalizers.NFD(), normalizers.StripAccents()])
    
    tokenizer.normalizer = normalizers.Sequence(normalizer_list)

    # The pre_tokenizer splits the text into initial words.
    if keep_punctuation:
        # If we keep punctuation, we treat it as its own token (e.g., "Hello," -> "Hello" , ",").
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
        ])
    else:
        # If punctuation is already stripped by the normalizer, we only need to split by whitespace.
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Define the trainer with the vocabulary size and special tokens
    special_tokens = ["<mask>", "<pad>", "<unk>", "<s>", "</s>"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    # Train the tokenizer on the input file
    tokenizer.train([text_file], trainer)
    
    return tokenizer
