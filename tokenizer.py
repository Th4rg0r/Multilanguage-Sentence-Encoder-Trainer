from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
import re


def sentence_splitter(input_file, output_file):
    pattern = r"(?<=[。！？\.!?…！？؟।:;])\s+"
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = re.split(pattern, text)
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")


def tokenize(text_file, vocab_size, keep_accents=True):
    """
    creates a tokenizer from a text_file and vocab_size as input

    does not normalize Accents by default
    if you want to normalize accents set "keep_accents" to False
    maybe useful for languages like english: makes "Ápplé" to "apple"

    Args:
        text_file (String): the text file name including all the text for the vocabulary
                            - ideal case is one sentence per line - makes better tokenizer but is not a neccessity
        vocab_size (int): the target vocabulary size

    Returns:
       Tokenizer: the tokenizer create by the vocab
    """
    # use byte-pair encoding to shrink vocab size
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    if keep_accents:
        # normalize everything to lower-case
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Lowercase(),
            ]
        )
    else:
        # normalizes to lowercase - also strip accents : é to e
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            # split by whytespace, also split and generate tokens for ".", "-", "@", even if they are adjacent to words
            pre_tokenizers.Whitespace(),
            # make punktuations their own tokens "?!" becomes "?","!"
            pre_tokenizers.Punctuation(),
        ]
    )

    # special tokens: padding, unknown, start of sequence, end of sequence
    special_tokens = ["<mask>", "<pad>", "<unk>", "<s>", "</s>"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    # train the tokenizer
    tokenizer.train([text_file], trainer)
    return tokenizer
