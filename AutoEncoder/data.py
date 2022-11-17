import os
from io import open
import torch
from Bio import SeqIO
import numpy as np
from typing import Any

class Sequence(object):
    def __init__(self, path):
        self.train = self.tokenize(os.path.join(path, 'virus-2014.fasta'))
        self.valid = self.tokenize(os.path.join(path, 'virus-2014.fasta'))
        self.test = self.tokenize(os.path.join(path, 'virus-2014.fasta'))

    def tokenize(self, path):
        """Tokenizes a fasta file."""
        assert os.path.exists(path)
        # read the fasta file 
        records = list(SeqIO.parse(path, "fasta"))
        # convert records to one-hot encoding
        sequence=[]
        def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGU',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
            """One-hot encode a sequence."""
            def to_uint8(string):
                return np.frombuffer(string.encode('ascii'), dtype=np.uint8)

            hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
            hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
            hash_table[to_uint8(neutral_alphabet)] = neutral_value
            hash_table = hash_table.astype(dtype)
            return hash_table[to_uint8(sequence)]

        for record in records:
            # convert the sequence to one-hot encoding
            one_hot = one_hot_encode(str(record.seq))
            sequence.append(torch.tensor(one_hot))
            

        return np.array(sequence)
