import os.path as osp
import numpy as np
import pandas as pd
import torch
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

from dataset.dataset_split import train_test_split

current_path = osp.dirname(osp.realpath(__file__))

class DMIDataset():
    def __init__(self, seed):
        self.seed = seed
        self.db_num = 2210
        self.tax_num = 436
        self.uni_num = 1894

        self.ft_dict = {}
        
        self.ft_dict['db_whole'] = np.load(osp.join(current_path, 'db_2210_1024.npy'), allow_pickle=True)
        self.ft_dict['tax_whole'] = np.load(osp.join(current_path, 'tax_436_416.npy'), allow_pickle=True)
        self.ft_dict['uni_whole'] = np.load(osp.join(current_path, 'uni_1893_1295.npy'), allow_pickle=True)
                
        self.known_db_idx = np.load(osp.join(current_path, 'db.npy'), allow_pickle=True)
        self.known_uni_idx = np.load(osp.join(current_path, 'known_uni.npy'), allow_pickle=True)
        self.unknown_uni_idx_70 = np.load(osp.join(current_path, 'unknown_uni_70.npy'), allow_pickle=True)
        self.unknown_uni_idx_80 = np.load(osp.join(current_path, 'unknown_uni_80.npy'), allow_pickle=True)
        self.unknown_uni_idx_90 = np.load(osp.join(current_path, 'unknown_uni_90.npy'), allow_pickle=True)
        self.padding_uni_idx = np.load(osp.join(current_path, 'padding_uni.npy'), allow_pickle=True)
        self.known_tax_idx = np.load(osp.join(current_path, 'known_tax.npy'), allow_pickle=True)
        self.unknown_tax_idx = np.load(osp.join(current_path, 'unknown_tax.npy'), allow_pickle=True)
        
        self.test_db = np.load(osp.join(current_path, 'test_db.npy'), allow_pickle=True)
        self.test_tax = np.load(osp.join(current_path, 'test_tax.npy'), allow_pickle=True)
        self.test_label = np.load(osp.join(current_path, 'test_label.npy'), allow_pickle=True)
        self.test_uni = np.load(osp.join(current_path, 'test_uni_padding.npy'), allow_pickle=True)
        self.test_uni_label = np.load(osp.join(current_path, 'test_uni_label_padding.npy'), allow_pickle=True)
        self.test_uni_label_mask = np.load(osp.join(current_path, 'test_uni_label_mask.npy'), allow_pickle=True)
        
        self.valid_db = np.load(osp.join(current_path, 'valid_db.npy'), allow_pickle=True)
        self.valid_tax = np.load(osp.join(current_path, 'valid_tax.npy'), allow_pickle=True)
        self.valid_label = np.load(osp.join(current_path, 'valid_label.npy'), allow_pickle=True)
        self.valid_uni = np.load(osp.join(current_path, 'valid_uni_padding.npy'), allow_pickle=True)
        self.valid_uni_label = np.load(osp.join(current_path, 'valid_uni_label_padding.npy'), allow_pickle=True)
        self.valid_uni_label_mask = np.load(osp.join(current_path, 'valid_uni_label_mask.npy'), allow_pickle=True)
        
        self.unseen_db = np.load(osp.join(current_path, 'unseen_db.npy'), allow_pickle=True)
        self.unseen_tax = np.load(osp.join(current_path, 'unseen_tax.npy'), allow_pickle=True)
        self.unseen_label = np.load(osp.join(current_path, 'unseen_label.npy'), allow_pickle=True)
        self.unseen_uni_70 = np.load(osp.join(current_path, 'unseen_uni_70_padding.npy'), allow_pickle=True)
        self.unseen_uni_label_70 = np.load(osp.join(current_path, 'unseen_uni_70_label_padding.npy'), allow_pickle=True)
        self.unseen_uni_label_mask_70 = np.load(osp.join(current_path, 'unseen_uni_70_label_mask.npy'), allow_pickle=True)
        self.unseen_uni_80 = np.load(osp.join(current_path, 'unseen_uni_80_padding.npy'), allow_pickle=True)
        self.unseen_uni_label_80 = np.load(osp.join(current_path, 'unseen_uni_80_label_padding.npy'), allow_pickle=True)
        self.unseen_uni_label_mask_80 = np.load(osp.join(current_path, 'unseen_uni_80_label_mask.npy'), allow_pickle=True)
        self.unseen_uni_90 = np.load(osp.join(current_path, 'unseen_uni_90_padding.npy'), allow_pickle=True)
        self.unseen_uni_label_90 = np.load(osp.join(current_path, 'unseen_uni_90_label_padding.npy'), allow_pickle=True)
        self.unseen_uni_label_mask_90 = np.load(osp.join(current_path, 'unseen_uni_90_label_mask.npy'), allow_pickle=True)
        
        self.train_pos_db = np.load(osp.join(current_path, 'train_pos_db.npy'), allow_pickle=True)
        self.train_pos_tax = np.load(osp.join(current_path, 'train_pos_tax.npy'), allow_pickle=True)
        self.train_pos_label = np.load(osp.join(current_path, 'train_pos_label.npy'), allow_pickle=True)
        self.train_pos_uni = np.load(osp.join(current_path, 'train_pos_uni_padding.npy'), allow_pickle=True)
        self.train_pos_uni_label = np.load(osp.join(current_path, 'train_pos_uni_label_padding.npy'), allow_pickle=True)
        self.train_pos_uni_label_mask = np.load(osp.join(current_path, 'train_pos_uni_label_mask.npy'), allow_pickle=True)
        
        self.train_neg_db = np.load(osp.join(current_path, 'train_neg_db.npy'), allow_pickle=True)
        self.train_neg_tax = np.load(osp.join(current_path, 'train_neg_tax.npy'), allow_pickle=True)
        self.train_neg_label = np.load(osp.join(current_path, 'train_neg_label.npy'), allow_pickle=True)
        self.train_neg_uni = np.load(osp.join(current_path, 'train_neg_uni_padding.npy'), allow_pickle=True)
        self.train_neg_uni_label = np.load(osp.join(current_path, 'train_neg_uni_label_padding.npy'), allow_pickle=True)
        self.train_neg_uni_label_mask = np.load(osp.join(current_path, 'train_neg_uni_label_mask.npy'), allow_pickle=True)
        
        self.train_db = np.load(osp.join(current_path, 'train_db.npy'), allow_pickle=True)
        self.train_tax = np.load(osp.join(current_path, 'train_tax.npy'), allow_pickle=True)
        self.train_label = np.load(osp.join(current_path, 'train_label.npy'), allow_pickle=True)
        self.train_uni = np.load(osp.join(current_path, 'train_uni_padding.npy'), allow_pickle=True)
        self.train_uni_label = np.load(osp.join(current_path, 'train_uni_label_padding.npy'), allow_pickle=True)
        self.train_uni_label_mask = np.load(osp.join(current_path, 'train_uni_label_mask.npy'), allow_pickle=True)
        
        self.train_neg_sample_db = None
        self.train_neg_sample_tax = None
        self.train_neg_sample_label = None
        self.train_neg_sample_uni = None
        self.train_neg_sample_uni_label = None
        self.train_neg_sample_uni_mask_label = None

    def to_tensor(self, device='cpu'):
        print('DMI ft to tensor', device)
        for ft_name in self.ft_dict:
            self.ft_dict[ft_name] = torch.FloatTensor(self.ft_dict[ft_name]).to(device)



