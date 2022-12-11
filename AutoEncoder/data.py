import os
from io import open
import torch
from Bio import SeqIO
import numpy as np
from typing import Any
import glob
import numpy as np
import re
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# create a label encoder with 'acgtn' alphabet
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','t','z','c','g']))
def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(string_to_array(my_array))
    return integer_encoded
class Sequence(object):
    def __init__(self, path):
        # train_list=[]
        # train_list.append(glob.glob(r'../dataset/virus/-2014/*.py'))
        self.train = self.tokenize(os.path.join(path, 'virus/-2014/virus-2014.fasta'))
        self.valid = self.tokenize(os.path.join(path, 'virus/2014-2015/virus2014-2015.fasta'))
        self.test = self.tokenize(os.path.join(path, 'virus/2015-/virus2015-.fasta'))
        # self.train = self.tokenize(os.path.join(path, 'virus/2014-2015/virus2014-2015.fasta'))
        # self.valid = self.train
        # self.test = self.train

    
    def tokenize(self, path):
        """Tokenizes a fasta file."""
        print(path)
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
            enc = ordinal_encoder(str(record.seq))
class ClassifierDataset(data.Dataset):

    def __init__(self, data_list, seq_len, sample_num = 20) -> None:
        super().__init__()
        self.data_list = data_list
        self.data = []
        self.seq_len = seq_len
        self.sample_num = sample_num
        self.regenerate_data()
            
    def regenerate_data(self):
        self.data = []
        for seq, label in self.data_list:
            # Randomly sample a slice of the sequence
            if len(seq) > self.seq_len:
                for i in range(self.sample_num):
                    start = np.random.randint(0, len(seq) - self.seq_len)
                    end = start + self.seq_len
                    self.data.append((seq[start:end], label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]