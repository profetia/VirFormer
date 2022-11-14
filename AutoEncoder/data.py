import os
from io import open
import torch
from Bio import SeqIO



class Sequence(object):
    def __init__(self, path):
        self.sequence = [] 
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a fasta file."""
        assert os.path.exists(path)
        # read the fasta file 
        records = list(SeqIO.parse(path, "fasta"))
        # convert records to one-hot encoding
        idxs = []
        for record in records:
            # convert the sequence to one-hot encoding
            one_hot = torch.zeros(len(record.seq), 4)
            for i, base in enumerate(record.seq):
                if base == 'A':
                    one_hot[i][0] = 1
                elif base == 'C':
                    one_hot[i][1] = 1
                elif base == 'G':
                    one_hot[i][2] = 1
                elif base == 'T':
                    one_hot[i][3] = 1
                else:
                    print("Error: invalid base")
            # add the sequence to the dictionary
            self.sequence.append(one_hot)
            idxs.append(len(self.sequence) - 1)
        return idxs
