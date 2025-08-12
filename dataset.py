from alive_progress import alive_bar
from collections import deque
from random import shuffle
from tokenizers import Tokenizer
from torchdata.nodes import (
    IterableWrapper,
    ParallelMapper,
    Batcher,
    Loader,
    PinMemory,
    Prefetcher,
    Unbatcher,
)
import random
import torch
import os
import subprocess


def get_file_line_cnt(fp):
    line_count = 0
    with open(fp, "r") as f:
        for line in f:
            line_count += 1
    return line_count


def split_train_test_set(file_path, out_path, test_size=0.1):
    """Splits a text file into a training and a test set."""
    print("Splitting data into train and test sets...")
    
    line_count = get_file_line_cnt(file_path)
    if line_count == 0:
        print("Input file is empty. Nothing to split.")
        return 0, 0

    train_out_fp = os.path.join(out_path, "train.txt")
    test_out_fp = os.path.join(out_path, "test.txt")

    indices = list(range(line_count))
    random.shuffle(indices)
    
    split_idx = int(line_count * (1 - test_size))
    train_indices = set(indices[:split_idx])

    train_cnt = 0
    test_cnt = 0

    with open(file_path, "r", encoding='utf-8', errors='ignore') as f_input, \
         open(train_out_fp, "w", encoding='utf-8') as f_train, \
         open(test_out_fp, "w", encoding='utf-8') as f_test, \
         alive_bar(line_count) as bar:
        
        for i, line in enumerate(f_input):
            if i in train_indices:
                f_train.write(line)
                train_cnt += 1
            else:
                f_test.write(line)
                test_cnt += 1
            bar()
            
    print(f"Splitting complete. Train lines: {train_cnt}, Test lines: {test_cnt}")
    return train_cnt, test_cnt


class ShuffleBuffer:
    def __init__(self, iterable, buffer_size=1000):
        self.iterable = iterable
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = deque()
        iterator = iter(self.iterable)

        try:
            for _ in range(self.buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass  # fewer than buffer_size items

        while buffer:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            try:
                buffer[idx] = next(iterator)
            except StopIteration:
                buffer.remove(buffer[idx])


class LazyLoader:
    def __init__(self, tokenizer, file_path, batch_size, max_word_per_sentence, contrastive_learning = False, mask_ratio=0.15):
        self.file_path = file_path
        # self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer = tokenizer
        self.start_id = self.tokenizer.token_to_id("<s>")
        self.end_id = self.tokenizer.token_to_id("</s>")
        self.batch_size = batch_size
        self.max_word_per_sentence = max_word_per_sentence
        self.contrastive_learning = contrastive_learning
        self.mask_ratio = mask_ratio

    def stream_lines(self, fp):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if len(line.split()) > self.max_word_per_sentence:
                    continue
                yield line.strip()

    def get_one_hot_embedding(token_id: int) -> torch.Tensor:
        """
        Generates a one-hot encoded tensor for a given token ID.

        Args:
            tokenizer: The tokenizers.Tokenizer instance.
            token_id: The integer ID of the token to one-hot encode.

        Returns:
            A 1D torch.Tensor representing the one-hot encoding.
        """
        vocab_size = self.tokenizer.get_vocab_size()
        if not (0 <= token_id < vocab_size):
            raise ValueError(
                f"Token ID {token_id} is out of vocabulary range (0 to {vocab_size - 1})."
            )

        one_hot_vector = torch.zeros(vocab_size, dtype=torch.float)
        one_hot_vector[token_id] = 1.0
        return one_hot_vector

    def transform_input(self, line):
        enc = self.tokenizer.encode(line)
        token_ids = enc.ids
        mask_id = self.tokenizer.token_to_id("<mask>")
        pad_id = self.tokenizer.token_to_id("<pad>")

        n = len(token_ids)
        num_predicted = max(1, int(round(n * self.mask_ratio)))
        # Generate a random permutation of indices [0..n-1]
        perm = torch.randperm(n).tolist()
        # Take last num_predicted indices in the permutation as predicted positions
        pred_positions = set(perm[-num_predicted:])
        # Create labels: -100 for non-predicted tokens, original id for predicted
        labels = torch.full((n,), pad_id, dtype=torch.long)
        for idx in pred_positions:
            labels[idx] = token_ids[idx]
        # Create input with [MASK] tokens at predicted positions
        masked_ids = []
        for i, tok_id in enumerate(token_ids):
            if i in pred_positions:
                # 80% [MASK], 10% random token, 10% original (as in BERT)
                rand = torch.rand(1).item()
                if rand < 0.8:
                    masked_ids.append(mask_id)
                elif rand < 0.9:
                    masked_ids.append(torch.randint(self.tokenizer.get_vocab_size(), (1,)).item())
                else:
                    masked_ids.append(tok_id)
            else:
                masked_ids.append(tok_id)
        # Convert to tensors and add batch dim
        input_ids = torch.tensor(masked_ids)# [ seq_len]
        # Attention mask (1 for real tokens)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, labels

    def clr_transform_input(self, line):
        enc = self.tokenizer.encode(line)
        ids = enc.ids
        src_ids = [self.start_id] + ids + [self.end_id]
        return torch.tensor(src_ids, dtype=torch.long)

    def loader(self):
        # 1. Wrap line-stream
        buffered_stream = ShuffleBuffer(self.stream_lines(self.file_path), 2048)
        node = IterableWrapper(buffered_stream)

        # 2. apply paralell processing

        node = ParallelMapper(
            node,
            map_fn=lambda l: self.clr_transform_input(l) if self.contrastive_learning else self.transform_input(l),
            num_workers=4,
            method="thread",
        )
        # 3. group into batches
        node = Batcher(node, batch_size=self.batch_size, drop_last=False)

        node = PinMemory(node)
        node = Prefetcher(node, prefetch_factor=4)
        loader = Loader(node)
        return loader

    def collate_fn(self, batch):
        if not self.contrastive_learning:
            # batch is a list of (src_ids, tgt_ids) pairs (as torch Tensors of different lengths)
            src_batch, mask_batch, label_batch = zip(*batch)
            src_lens = [x.size(-1) for x in src_batch]
            max_src = max(src_lens)
            pad_id = self.tokenizer.token_to_id("<pad>")

            # Pad sequences and build masks
            padded_src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
            padded_label = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
            src_mask = torch.ones(len(batch), max_src, dtype=torch.bool)
            for i, (src_ids, mask, label) in enumerate(batch):
                padded_src[i, : src_ids.shape[0]] = src_ids
                src_mask[i, : src_ids.shape[0]] = mask
                padded_label[i, :src_ids.shape[0]] = label
                # Transformer expects key_padding_mask where True means **not** allowed
            src_mask = ~src_mask
            return padded_src, src_mask, padded_label
        else:
            # batch is a list of (src_ids, tgt_ids) pairs (as torch Tensors of different lengths)
            src_batch = batch
            src_lens = [len(x) for x in src_batch]
            max_src = max(src_lens)
            pad_id = self.tokenizer.token_to_id("<pad>")

            # Pad sequences and build masks
            padded_src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
            src_mask = torch.ones(len(batch), max_src, dtype=torch.bool)
            for i, src_ids in enumerate(batch):
                padded_src[i, : len(src_ids)] = src_ids
                src_mask[i, : len(src_ids)] = 0
                # Transformer expects key_padding_mask where True means **not** allowed
        return padded_src, src_mask
            
