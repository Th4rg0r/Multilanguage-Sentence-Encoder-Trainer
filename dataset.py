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
    def __init__(self, tokenizer, file_path, batch_size, max_word_per_sentence, contrastive_learning = False):
        self.file_path = file_path
        # self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer = tokenizer
        self.start_id = self.tokenizer.token_to_id("<s>")
        self.end_id = self.tokenizer.token_to_id("</s>")
        self.batch_size = batch_size
        self.max_word_per_sentence = max_word_per_sentence
        self.contrastive_learning = contrastive_learning

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
        ids = enc.ids
        mask_id = self.tokenizer.token_to_id("<mask>")

        ids_len = len(ids)
        masked_count = int(ids_len * 0.22)

        choosen_elements = random.sample(list(range(ids_len)), masked_count)
        choosen_elements.sort()

        random_elements = list(choosen_elements)
        random.shuffle(random_elements)

        result_batch = []

        for i, idx in enumerate(random_elements):
            cur_ids = list(ids)
            tgt_mask = torch.zeros(len(cur_ids), dtype=torch.bool)
            label_id = cur_ids[idx]
            cur_ids[idx] = mask_id
            tgt_mask[random_elements[i + 1 :]] = 1
            src_ids = [self.start_id] + cur_ids + [self.end_id]
            tgt_mask = torch.cat(
                (
                    torch.tensor([0], dtype=torch.bool),
                    tgt_mask,
                    torch.tensor([0], dtype=torch.bool),
                )
            )
            result_batch.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    tgt_mask,
                    torch.tensor(label_id, dtype=torch.long),
                )
            )
        return result_batch

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
        # 3. Unbatcher if not contrastive learning
        if not self.contrastive_learning:
            node = Unbatcher(node)
        # 4. group into batches
        node = Batcher(node, batch_size=self.batch_size, drop_last=False)

        node = PinMemory(node)
        node = Prefetcher(node, prefetch_factor=4)
        loader = Loader(node)
        return loader

    def collate_fn(self, batch):
        if not self.contrastive_learning:
            # batch is a list of (src_ids, tgt_ids) pairs (as torch Tensors of different lengths)
            src_batch, mask_batch, label_batch = zip(*batch)
            label_batch = torch.tensor(label_batch, dtype=torch.long)
            src_lens = [len(x) for x in src_batch]
            max_src = max(src_lens)
            pad_id = self.tokenizer.token_to_id("<pad>")

            # Pad sequences and build masks
            padded_src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
            src_mask = torch.ones(len(batch), max_src, dtype=torch.bool)
            for i, (src_ids, mask, label) in enumerate(batch):
                padded_src[i, : len(src_ids)] = src_ids
                src_mask[i, : len(src_ids)] = mask
                # Transformer expects key_padding_mask where True means **not** allowed
            return padded_src, src_mask, label_batch
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
            
