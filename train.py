from models.deepdmi import DeepDMI
from dataset.dmi_dataset import DMIDataset
from metrics.evaluate import evaluate_mrr, evaluate_mrr_p
import random
import numpy as np
import torch
import math
import os
import os.path as osp
import sys
from utils.index_map import get_map_index_for_sub_arr
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

current_path = osp.dirname(osp.realpath(__file__))
save_model_data_dir = 'save_model_data'

class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.setup_seed(self.param_dict['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = DMIDataset(seed=self.param_dict['seed'])
        self.dataset.to_tensor(self.device)
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        self.trainer_info = '{}_seed={}_batch={}'.format(self.file_name, self.param_dict['seed'], self.param_dict['batch_size'])
        self.loss_op = nn.BCELoss()
        self.build_model()

    def build_model(self):
        self.model = DeepDMI(**self.param_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'])
        self.best_res = None
        self.min_dif = 1e-10
                                       
    def setup_seed(self, seed):
        seed = int(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        # torch.use_deterministic_algorithms(True)
                                       
    def iteration(self, epoch, db_pair_idx, tax_pair_idx, label_pair_idx, uni_pair_idx, uni_label_pair_idx, uni_label_mask_pair_idx, db_graph_node_idx, tax_graph_node_idx, uni_graph_node_idx, is_training=True, shuffle=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        pair_num = db_pair_idx.shape[0]
        if shuffle is True:
            range_idx = np.random.permutation(pair_num)
        else:
            range_idx = np.arange(0, pair_num)

        all_pred_1 = []
        all_pred_2 = []

        all_label_1 = []
        all_label_2 = []
        
        all_db_1 = []
        all_db_2 = []
        all_tax_1 = []
        all_tax_2 = []
        all_uni = []
        
        all_mask_1 = []
        all_mask_2 = []

        db_graph_node_ft = self.dataset.ft_dict['db_whole'][db_graph_node_idx]
        tax_graph_node_ft = self.dataset.ft_dict['tax_whole'][tax_graph_node_idx]
        uni_graph_node_ft = self.dataset.ft_dict['uni_whole'][uni_graph_node_idx[:-1]]              
        db_graph_map_arr = get_map_index_for_sub_arr(
            db_graph_node_idx, np.arange(0, self.dataset.db_num))
        tax_graph_map_arr = get_map_index_for_sub_arr(
            tax_graph_node_idx, np.arange(0, self.dataset.tax_num))
        uni_graph_map_arr = get_map_index_for_sub_arr(
            uni_graph_node_idx, np.arange(0, self.dataset.uni_num))

        for i in range(math.ceil(pair_num/self.param_dict['batch_size'])):
            right_bound = min((i + 1)*self.param_dict['batch_size'], pair_num + 1)
            batch_idx = range_idx[i * self.param_dict['batch_size']: right_bound]
            

            b_db_idx = db_pair_idx[batch_idx]
            b_tax_idx = tax_pair_idx[batch_idx]
            b_label= torch.FloatTensor(label_pair_idx[batch_idx]).to(self.device)
            b_uni_idx = uni_pair_idx[batch_idx]
            b_label_uni = torch.FloatTensor(uni_label_pair_idx[batch_idx]).to(self.device)
            b_label_uni_mask1 = torch.FloatTensor(uni_label_mask_pair_idx[batch_idx]).to(self.device)
            b_label_2 = torch.unsqueeze(b_label, 1)
            b_label_3 = torch.repeat_interleave(b_label_2, b_label_uni.size()[1], 1)
            b_label_uni_mask2 = b_label_3

            batch_db_node_idx_in_graph = torch.LongTensor(db_graph_map_arr[b_db_idx.astype('int64')]).to(self.device)
            batch_tax_node_idx_in_graph = torch.LongTensor(tax_graph_map_arr[b_tax_idx.astype('int64')]).to(self.device)
            batch_uni_node_idx_in_graph = torch.LongTensor(uni_graph_map_arr[b_uni_idx.astype('int64')]).to(self.device)

            ft_dict = {
                'db_graph_node_ft': db_graph_node_ft,
                'tax_graph_node_ft': tax_graph_node_ft,
                'uni_graph_node_ft': uni_graph_node_ft,
                'db_idx': batch_db_node_idx_in_graph,
                'tax_idx': batch_tax_node_idx_in_graph,
                'uni_idx': batch_uni_node_idx_in_graph,
                'uni_mask': b_label_uni_mask1
            }

            pred1, pred2, db_adj, tax_adj, uni_adj = self.model(**ft_dict)
            pred1 = pred1.view(-1)
            pred2 = torch.squeeze(pred2, -1)
            if is_training:
                c_loss = self.loss_op(pred1, b_label) + self.loss_op(pred2 * b_label_uni_mask1 * b_label_uni_mask2, b_label_uni)
                param_l2_loss = 0
                param_l1_loss = 0
                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_l2_loss += torch.norm(param, p=2)
                        param_l1_loss += torch.norm(param, p=1)
                param_l2_loss = self.param_dict['param_l2_coef'] * param_l2_loss
                #param_l1_loss = self.param_dict['param_l1_coef'] * param_l1_loss
                adj_l1_loss = self.param_dict['adj_loss_coef'] * torch.norm(db_adj) + \
                                  self.param_dict['adj_loss_coef'] * torch.norm(tax_adj) + \
                                  self.param_dict['adj_loss_coef'] * torch.norm(uni_adj)
                loss = c_loss + adj_l1_loss + param_l2_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            pred1_ = pred1.detach().to('cpu').numpy()
            b_label_ = b_label.detach().to('cpu').numpy()
            all_pred_1 = np.hstack([all_pred_1, pred1_])
            all_label_1 = np.hstack([all_label_1, b_label_])

            all_db_1 = np.hstack([all_db_1, b_db_idx])
            all_tax_1 = np.hstack([all_tax_1, b_tax_idx])
            
            pred2_ = pred2.view(-1).detach().to('cpu').numpy()
            b_label_uni_ = b_label_uni.view(-1).detach().to('cpu').numpy()
            all_pred_2 = np.hstack([all_pred_2, pred2_])
            all_label_2 = np.hstack([all_label_2, b_label_uni_])  
            
            all_db_2 = np.hstack([all_db_2, b_db_idx.repeat(b_label_uni.size()[1], 0)])
            all_tax_2 = np.hstack([all_tax_2, b_tax_idx.repeat(b_label_uni.size()[1], 0)])
            all_uni = np.hstack([all_uni, b_uni_idx.flatten()])
            
            all_mask_1 = np.hstack([all_mask_1, b_label_uni_mask1.view(-1).detach().to('cpu').numpy()])
            all_mask_2 = np.hstack([all_mask_2, b_label_uni_mask2.view(-1).detach().to('cpu').numpy()])
                                       
        return all_pred_1, all_label_1, all_db_1, all_tax_1, all_pred_2, all_label_2, all_db_2, all_tax_2, all_uni, all_mask_1, all_mask_2

    def print_res(self, res_list, epoch):
        train_mrr, valid_mrr, test_mrr, unseen_mrr_70, unseen_mrr_80, unseen_mrr_90  = res_list

        msg_log = 'Epoch: {:03d}, MRR: Train {:.6f}, Val: {:.6f}, Test: {:.6f}, Unseen_70: {:.6f}, Unseen_80: {:.6f}, Unseen_90: {:.6f}, '.format(epoch, train_mrr, valid_mrr, test_mrr, unseen_mrr_70, unseen_mrr_80, unseen_mrr_90)
        print(msg_log, end='\n')
        
    def print_res_p(self, res_list, epoch):
        train_mrr, valid_mrr, test_mrr, unseen_mrr_70, unseen_mrr_80, unseen_mrr_90 = res_list

        msg_log = 'Epoch: {:03d}, MRR_p: Train {:.6f}, Val: {:.6f}, Test: {:.6f}, Unseen_70: {:.6f}, Unseen_80: {:.6f}, Unseen_90: {:.6f}, '.format(epoch, train_mrr, valid_mrr, test_mrr, unseen_mrr_70, unseen_mrr_80, unseen_mrr_90)
        print(msg_log, end='\n')

    def train(self, display=True):

        for epoch in range(1, self.param_dict['epoch_num'] + 1):
            # train
            idx_sample = np.random.choice([_ for _ in range(self.dataset.train_neg_db.shape[0])], size=self.dataset.train_pos_db.shape[0], replace=False)
                                                
            self.dataset.train_neg_sample_db = self.dataset.train_neg_db[idx_sample]
            self.dataset.train_db = np.concatenate((self.dataset.train_pos_db, self.dataset.train_neg_sample_db))
            self.dataset.train_neg_sample_tax = self.dataset.train_neg_tax[idx_sample]
            self.dataset.train_tax = np.concatenate((self.dataset.train_pos_tax, self.dataset.train_neg_sample_tax))
            self.dataset.train_neg_sample_label = self.dataset.train_neg_label[idx_sample]
            self.dataset.train_label = np.concatenate((self.dataset.train_pos_label, self.dataset.train_neg_sample_label))
            self.dataset.train_neg_sample_uni = self.dataset.train_neg_uni[idx_sample]
            self.dataset.train_uni = np.concatenate((self.dataset.train_pos_uni, self.dataset.train_neg_sample_uni))
            self.dataset.train_neg_sample_uni_label = self.dataset.train_neg_uni_label[idx_sample]
            self.dataset.train_uni_label = np.concatenate((self.dataset.train_pos_uni_label, self.dataset.train_neg_sample_uni_label))
            self.dataset.train_neg_sample_uni_label_mask = self.dataset.train_neg_uni_label_mask[idx_sample]
            self.dataset.train_uni_label_mask = np.concatenate((self.dataset.train_pos_uni_label_mask, self.dataset.train_neg_sample_uni_label_mask))
            
            train_pred_1, train_label_1, train_db_1, train_tax_1, train_pred_2, train_label_2, train_db_2, train_tax_2, train_uni, train_mask_1, train_mask_2 = \
                self.iteration(epoch, self.dataset.train_db,
                               self.dataset.train_tax,
                               self.dataset.train_label,
                               self.dataset.train_uni,
                               self.dataset.train_uni_label,
                               self.dataset.train_uni_label_mask,
                               db_graph_node_idx=self.dataset.known_db_idx,
                               tax_graph_node_idx=self.dataset.known_tax_idx,
                               uni_graph_node_idx=np.hstack((self.dataset.known_uni_idx, self.dataset.padding_uni_idx)),
                               is_training=True, shuffle=True)
            train_mrr = evaluate_mrr(train_db_1, train_tax_1, train_label_1, train_pred_1, 2210)
            # train_mrr_p = evaluate_mrr_p(train_db_2, train_tax_2, train_uni, train_label_2, train_pred_2, train_mask_1, train_mask_2, 2210)

            # valid
            valid_pred_1, valid_label_1, valid_db_1, valid_tax_1, valid_pred_2, valid_label_2, valid_db_2, valid_tax_2, valid_uni, valid_mask_1, valid_mask_2 = \
                self.iteration(epoch, self.dataset.valid_db,
                               self.dataset.valid_tax,
                               self.dataset.valid_label,
                               self.dataset.valid_uni,
                               self.dataset.valid_uni_label,
                               self.dataset.valid_uni_label_mask,
                               db_graph_node_idx=self.dataset.known_db_idx,
                               tax_graph_node_idx=self.dataset.known_tax_idx,
                               uni_graph_node_idx=np.hstack((self.dataset.known_uni_idx, self.dataset.padding_uni_idx)),
                               is_training=False, shuffle=False)
            valid_mrr = evaluate_mrr(valid_db_1, valid_tax_1, valid_label_1, valid_pred_1, 2210)
            # valid_mrr_p = evaluate_mrr_p(valid_db_2, valid_tax_2, valid_uni, valid_label_2, valid_pred_2, valid_mask_1, valid_mask_2, 2210)

            # test
            test_pred_1, test_label_1, test_db_1, test_tax_1, test_pred_2, test_label_2, test_db_2, test_tax_2, test_uni, test_mask_1, test_mask_2 = \
                self.iteration(epoch, self.dataset.test_db,
                               self.dataset.test_tax,
                               self.dataset.test_label,
                               self.dataset.test_uni,
                               self.dataset.test_uni_label,
                               self.dataset.test_uni_label_mask,
                               db_graph_node_idx=self.dataset.known_db_idx,
                               tax_graph_node_idx=self.dataset.known_tax_idx,
                               uni_graph_node_idx=np.hstack((self.dataset.known_uni_idx, self.dataset.padding_uni_idx)),
                               is_training=False, shuffle=False)
            test_mrr = evaluate_mrr(test_db_1, test_tax_1, test_label_1, test_pred_1, 2210)
            # test_mrr_p = evaluate_mrr_p(test_db_2, test_tax_2, test_uni, test_label_2, test_pred_2, test_mask_1, test_mask_2, 2210)

            # unseen_70
            unseen_pred_1, unseen_label_1, unseen_db_1, unseen_tax_1, unseen_pred_2, unseen_label_2, unseen_db_2, unseen_tax_2, unseen_uni, unseen_mask_1, unseen_mask_2 = \
                self.iteration(epoch, self.dataset.unseen_db,
                               self.dataset.unseen_tax,
                               self.dataset.unseen_label,
                               self.dataset.unseen_uni_70,
                               self.dataset.unseen_uni_label_70,
                               self.dataset.unseen_uni_label_mask_70,
                               db_graph_node_idx=self.dataset.known_db_idx,
                               tax_graph_node_idx=np.hstack((self.dataset.known_tax_idx, self.dataset.unknown_tax_idx)),
                               uni_graph_node_idx=np.hstack((self.dataset.known_uni_idx, self.dataset.unknown_uni_idx_70, self.dataset.padding_uni_idx)),
                               is_training=False, shuffle=False)
            unseen_mrr_70 = evaluate_mrr(unseen_db_1, unseen_tax_1, unseen_label_1, unseen_pred_1, 2210)
            # unseen_mrr_p_70 = evaluate_mrr_p(unseen_db_2, unseen_tax_2, unseen_uni, unseen_label_2, unseen_pred_2, unseen_mask_1, unseen_mask_2, 2210)

            # unseen_80
            unseen_pred_1, unseen_label_1, unseen_db_1, unseen_tax_1, unseen_pred_2, unseen_label_2, unseen_db_2, unseen_tax_2, unseen_uni, unseen_mask_1, unseen_mask_2 = \
                self.iteration(epoch, self.dataset.unseen_db,
                               self.dataset.unseen_tax,
                               self.dataset.unseen_label,
                               self.dataset.unseen_uni_80,
                               self.dataset.unseen_uni_label_80,
                               self.dataset.unseen_uni_label_mask_80,
                               db_graph_node_idx=self.dataset.known_db_idx,
                               tax_graph_node_idx=np.hstack((self.dataset.known_tax_idx, self.dataset.unknown_tax_idx)),
                               uni_graph_node_idx=np.hstack((self.dataset.known_uni_idx, self.dataset.unknown_uni_idx_80, self.dataset.padding_uni_idx)),
                               is_training=False, shuffle=False)
            unseen_mrr_80 = evaluate_mrr(unseen_db_1, unseen_tax_1, unseen_label_1, unseen_pred_1, 2210)
            # unseen_mrr_p_80 = evaluate_mrr_p(unseen_db_2, unseen_tax_2, unseen_uni, unseen_label_2, unseen_pred_2, unseen_mask_1, unseen_mask_2, 2210)

            # unseen_90
            unseen_pred_1, unseen_label_1, unseen_db_1, unseen_tax_1, unseen_pred_2, unseen_label_2, unseen_db_2, unseen_tax_2, unseen_uni, unseen_mask_1, unseen_mask_2 = \
                self.iteration(epoch, self.dataset.unseen_db,
                               self.dataset.unseen_tax,
                               self.dataset.unseen_label,
                               self.dataset.unseen_uni_90,
                               self.dataset.unseen_uni_label_90,
                               self.dataset.unseen_uni_label_mask_90,
                               db_graph_node_idx=self.dataset.known_db_idx,
                               tax_graph_node_idx=np.hstack((self.dataset.known_tax_idx, self.dataset.unknown_tax_idx)),
                               uni_graph_node_idx=np.hstack((self.dataset.known_uni_idx, self.dataset.unknown_uni_idx_90, self.dataset.padding_uni_idx)),
                               is_training=False, shuffle=False)
            unseen_mrr_90 = evaluate_mrr(unseen_db_1, unseen_tax_1, unseen_label_1, unseen_pred_1, 2210)
            # unseen_mrr_p_90 = evaluate_mrr_p(unseen_db_2, unseen_tax_2, unseen_uni, unseen_label_2, unseen_pred_2, unseen_mask_1, unseen_mask_2, 2210)
            
            res_list = [
                train_mrr,
                valid_mrr,
                test_mrr,
                unseen_mrr_70,
                unseen_mrr_80,
                unseen_mrr_90
            ]
            res_list_p = [
                0,
                0,
                0,
                0,
                0,
                0
            ]

            if valid_mrr > self.min_dif:
                self.min_dif = valid_mrr
                self.best_res = res_list
                self.best_res_p = res_list_p
                self.best_epoch = epoch
                # save model
                # save_complete_model_path = osp.join(current_path, save_model_data_dir, self.trainer_info + '_complete.pkl')
                # torch.save(self.model, save_complete_model_path)
                save_model_param_path = osp.join(current_path, save_model_data_dir, self.trainer_info + '_param.pkl')
                torch.save(self.model.state_dict(), save_model_param_path)

            if display:
                self.print_res(res_list, epoch)
                self.print_res_p(res_list_p, epoch)

            if epoch % 50 == 0 and epoch > 0:
                print('Best res', end='\n')
                self.print_res(self.best_res, self.best_epoch)
                self.print_res_p(self.best_res_p, self.best_epoch)
                
    def evaluate_model(self):
        return 0


if __name__ == '__main__':
    for seed in range(10):
        print('seed = ', seed)
        param_dict = {
            'seed': seed,
            'kmer_min_df': 0.1,
            'batch_size': 256,
            'epoch_num': 10000,
            'h_dim': 512,
            'dropout_num': 0.4,
            'lr': 5e-5,
            'adj_loss_coef': 5e-4,
            'param_l2_coef': 5e-4,
            'db_dim': 1024,
            'tax_dim': 416,
            'uni_dim': 1295,
            'uni_max': 345
        }
        trainer = Trainer(**param_dict)
        f = open(r'./'+ trainer.trainer_info +'_log.txt','a+')
        sys.stdout=f
        trainer.train()
        #trainer.evaluate_model()
        f.close()
