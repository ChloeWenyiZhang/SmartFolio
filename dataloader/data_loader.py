import os
import sys
from torch.utils import data
import pickle

class AllGraphDataSampler(data.Dataset):
    def __init__(self, base_dir, gname_list=None,
                 data_start=None, data_middle=None, data_end=None,
                 train_start_date=None, train_end_date=None,
                 val_start_date=None, val_end_date=None,
                 test_start_date=None, test_end_date=None,
                 idx=False, date=True,
                 mode="train"):
        self.data_dir = os.path.join(base_dir)
        self.mode = mode
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        if gname_list is None:
            self.gnames_all = os.listdir(self.data_dir)
            self.gnames_all.sort()
        if idx:
            if mode == "train":
                self.gnames_all = self.gnames_all[self.data_start:self.data_middle]
            elif mode == "val":
                self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
            elif mode == "test":
                self.gnames_all = self.gnames_all[self.data_end:]
        if date:
            if mode == "train":
                self.gnames_all = self.gnames_all[self.date_to_idx(train_start_date):self.date_to_idx(train_end_date) + 1]
            elif mode == "val":
                self.gnames_all = self.gnames_all[self.date_to_idx(val_start_date):self.date_to_idx(val_end_date) + 1]
            elif mode == "test":
                self.gnames_all = self.gnames_all[self.date_to_idx(test_start_date):self.date_to_idx(test_end_date) + 1]
        self.data_all = self.load_state()

    def __len__(self):
        return len(self.data_all)

    def load_state(self):
        data_all = []
        length = len(self.gnames_all)
        for i in range(length):
            sys.stdout.flush()
            sys.stdout.write('{} data loading: {:.2f}%{}'.format(self.mode, i*100/length, '\r'))
            data_all.append(pickle.load(open(os.path.join(self.data_dir, self.gnames_all[i]), "rb")))
        print('{} data loaded!'.format(self.mode))
        return data_all

    def __getitem__(self, idx):
        return self.data_all[idx]

    def date_to_idx(self, date):
        result = None
        for i in range(len(self.gnames_all)):
            if date == self.gnames_all[i][:10]:
                result = i
        return result
