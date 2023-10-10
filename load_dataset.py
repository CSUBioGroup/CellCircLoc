import numpy as np
import os, pickle, torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def seq_process(seq, max_len):
    """ padding and truncate sequence """

    if len(seq) < max_len:
        return seq + '*' * (max_len - len(seq))
    else:
        return seq[:max_len]


def seqItem2id(item):
    """ Convert seq item to token """
    if item not in 'ATCG*':
        print("unexcept")
    items = 'ATCG'
    seqItem2id = {}
    seqItem2id.update(dict(zip(items, range(1, len(items) + 1))))
    seqItem2id.update({"*": 0})
    return seqItem2id[item]


def id2seqItem(i):
    """ Convert token to seq item  """
    items = 'ATCG'
    id2seqItem = ["*"] + list(items)
    return id2seqItem[i]


def vectorize(emb_type, window=13, sg=1, workers=0):
    """ Get embedding of 'onehot' or 'word2vec-[dim] """
    items = 'ATCG'
    emb_path = os.path.join(r'../embeds/', emb_type)
    emb_file = os.path.join(emb_path, emb_type + '.pkl')
    with open(r"data.pkl", "rb") as f:
        seq = pickle.load(f)

    if os.path.exists(emb_file):
        with open(emb_file, 'rb') as f:
            embedding = pickle.load(f)
        return embedding

    if emb_type == 'onehot':
        embedding = np.concatenate(
            (
                [np.zeros(len(items))], np.eye(len(items))
            )
        ).astype('float32')


    if os.path.exists(emb_path) == False:
        os.makedirs(emb_path)
    with open(emb_file, 'wb') as f:
        pickle.dump(embedding, f, protocol=4)
    return embedding


class CelllineDataset(Dataset):
    def __init__(self, seqs, labels, emb_type,max_len):

        self.labels = labels
        self.num_ess = np.sum(self.labels == 1)
        self.num_non = np.sum(self.labels == 0)
        self.raw_seqs = seqs
        self.processed_seqs = [seq_process(seq, max_len) for seq in self.raw_seqs]
        self.tokenized_seqs = [[seqItem2id(i) for i in seq] for seq in self.processed_seqs]
        embedding = nn.Embedding.from_pretrained(torch.tensor(vectorize(emb_type)))
        self.emb_dim = embedding.embedding_dim
        self.features = embedding(torch.LongTensor(self.tokenized_seqs))

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

    def __len__(self):
        return len(self.labels)




def load_dataset(emb_type,  max_len, seed):
    """ Load train & test dataset """

    with open(r"data.pkl", "rb") as f:
        seq = pickle.load(f)
        subcell = pickle.load(f)

    seq_data=seq
    temp_label_data=subcell
    temp_sub_cell=[]
    label_data=[]
    for s in temp_label_data:
        temp_sub_cell.append(s[0])
    temp_dict={'n':0,'c':1}  #nucleus 0  cytoplasm 1
    for s in temp_sub_cell:
        label_data.append(temp_dict[s])

    ess_indexes = [i for i, e in enumerate(label_data) if int(e) == 1]
    non_indexes = [i for i, e in enumerate(label_data) if int(e) == 0]
    num_ess = len(ess_indexes)
    num_non = len(non_indexes)
    print(f' dataset   label 1 :{num_ess}   label 0 :{num_non}')
    label_data=np.array(label_data)
    train_seqs, test_seqs, train_labels, test_labels = train_test_split(seq_data, label_data, test_size=0.1,
                                                                          random_state=seed, stratify=label_data)

    train_dataset = CelllineDataset(train_seqs, train_labels, emb_type, max_len)
    test_dataset = CelllineDataset(test_seqs, test_labels,emb_type, max_len)
    return train_dataset, test_dataset


def kmer_X(k,RNA):  #A T C G 0,1,2,3 RNA:list
    w_n={"A":0,"T":1,"C":2,"G":3}
    X=np.zeros([len(RNA),4**k])
    for i in range(len(RNA)):
        for j in range(len(RNA[i])-k+1):
            num=0
            for h in range(k):
                temp=w_n[RNA[i][j+h]]
                num=num+temp*(4**h)
            X[i,num]=X[i,num]+1
    return X
