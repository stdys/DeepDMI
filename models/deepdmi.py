import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from gcn_conv_input_mat import GCNConv
import sys
import time

class DeepDMI(nn.Module):
    def __init__(self, **param_dict):
        super(DeepDMI, self).__init__()
        self.param_dict = param_dict
        self.db_dim = param_dict['db_dim']
        self.tax_dim = param_dict['tax_dim']
        self.uni_dim = param_dict['uni_dim']
        self.h_dim = param_dict['h_dim']
        self.dropout = param_dict['dropout_num']

        self.db_linear_global = nn.Linear(param_dict['db_dim'], self.h_dim)
        self.tax_linear_global = nn.Linear(param_dict['tax_dim'], self.h_dim)
        self.uni_linear_global = nn.Linear(param_dict['uni_dim'], self.h_dim)

        self.db_linear_local = nn.Linear(param_dict['db_dim'], self.h_dim)
        self.tax_linear_local = nn.Linear(param_dict['tax_dim'], self.h_dim)
        self.uni_linear_local = nn.Linear(param_dict['uni_dim'], self.h_dim)
        
        self.share_linear = nn.Linear(self.h_dim, self.h_dim)
        self.share_gcn1 = GCNConv(self.h_dim, self.h_dim)

        self.db_adj_trans = nn.Linear(self.h_dim, self.h_dim)
        self.tax_adj_trans = nn.Linear(self.h_dim, self.h_dim)
        self.uni_adj_trans = nn.Linear(self.h_dim, self.h_dim)
        
        self.db_tax_uni_linear_global = nn.Linear(self.h_dim*3, self.h_dim)
        self.db_uni_linear_global = nn.Linear(self.h_dim*2, self.h_dim)
        
        self.db_tax_uni_linear_local = nn.Linear(self.h_dim*3, self.h_dim)
        self.db_uni_linear_local = nn.Linear(self.h_dim*2, self.h_dim)
        
        
        self.db_tax_pred_linear = nn.Linear(self.h_dim, 1)
        self.db_uni_pred_linear = nn.Linear(self.h_dim, 1)

        self.activation = nn.ReLU()
        
        self.cross_scale_merge_1 = nn.Parameter(
            torch.ones(1)
        )
        
        self.cross_scale_merge_2 = nn.Parameter(
            torch.ones(1)
        )
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)


    def forward(self, **ft_dict):
        device = ft_dict['db_graph_node_ft'].device
        db_graph_node_num = ft_dict['db_graph_node_ft'].size()[0]
        tax_graph_node_num = ft_dict['tax_graph_node_ft'].size()[0]
        uni_graph_node_num = ft_dict['uni_graph_node_ft'].size()[0]
        db_res_mat_global = torch.zeros(db_graph_node_num, self.h_dim).to(device)
        tax_res_mat_global = torch.zeros(tax_graph_node_num, self.h_dim).to(device)
        uni_res_mat_global = torch.zeros(uni_graph_node_num, self.h_dim).to(device)
        
        # global db
        db_node_ft_global = self.db_linear_global(ft_dict['db_graph_node_ft'])
        db_node_ft_global = self.activation(db_node_ft_global)

        db_node_ft_global = self.share_linear(db_node_ft_global)
        db_node_ft_global = self.activation(db_node_ft_global)
        db_res_mat_global = db_res_mat_global + db_node_ft_global

        db_trans_ft_global = self.db_adj_trans(db_node_ft_global)
        db_trans_ft_global = torch.tanh(db_trans_ft_global)
        w_global = torch.norm(db_trans_ft_global, p=2, dim=-1).view(-1, 1)
        w_mat_global = w_global * w_global.t()
        db_adj_global = torch.mm(db_trans_ft_global, db_trans_ft_global.t()) / w_mat_global

        db_node_ft_global = self.share_gcn1(db_node_ft_global, db_adj_global)
        db_res_mat_global = db_res_mat_global + db_node_ft_global
        
        # global tax
        tax_node_ft_global = self.tax_linear_global(ft_dict['tax_graph_node_ft'])
        tax_node_ft_global = self.activation(tax_node_ft_global)

        tax_node_ft_global = self.share_linear(tax_node_ft_global)
        tax_node_ft_global = self.activation(tax_node_ft_global)
        tax_res_mat_global = tax_res_mat_global + tax_node_ft_global

        tax_trans_ft_global = self.tax_adj_trans(tax_node_ft_global)
        tax_trans_ft_global = torch.tanh(tax_trans_ft_global)
        w_global = torch.norm(tax_trans_ft_global, p=2, dim=-1).view(-1, 1)
        w_mat_global = w_global * w_global.t()
        tax_adj_global = torch.mm(tax_trans_ft_global, tax_trans_ft_global.t()) / w_mat_global

        tax_node_ft_global = self.share_gcn1(tax_node_ft_global, tax_adj_global)
        tax_res_mat_global = tax_res_mat_global + tax_node_ft_global
        
        # global uni
        uni_node_ft_global = self.uni_linear_global(ft_dict['uni_graph_node_ft'])
        uni_node_ft_global = self.activation(uni_node_ft_global)

        uni_node_ft_global = self.share_linear(uni_node_ft_global)
        uni_node_ft_global = self.activation(uni_node_ft_global)
        uni_res_mat_global = uni_res_mat_global + uni_node_ft_global

        uni_trans_ft_global = self.uni_adj_trans(uni_node_ft_global)
        uni_trans_ft_global = torch.tanh(uni_trans_ft_global)
        w_global = torch.norm(uni_trans_ft_global, p=2, dim=-1).view(-1, 1)
        w_mat_global = w_global * w_global.t()
        uni_adj_global = torch.mm(uni_trans_ft_global, uni_trans_ft_global.t()) / w_mat_global

        uni_node_ft_global = self.share_gcn1(uni_node_ft_global, uni_adj_global)
        uni_res_mat_global = uni_res_mat_global + uni_node_ft_global

        # global
        db_res_mat_global = self.activation(db_res_mat_global)
        tax_res_mat_global = self.activation(tax_res_mat_global)
        uni_res_mat_global = self.activation(uni_node_ft_global)
        uni_padding_global = torch.zeros(1,self.h_dim).to(device)
        uni_res_mat_global = torch.cat([uni_res_mat_global, uni_padding_global], dim=0)

        # global 
        db_ft_global = db_res_mat_global[ft_dict['db_idx']]
        tax_ft_global = tax_res_mat_global[ft_dict['tax_idx']]
        uni_ft_global = uni_res_mat_global[ft_dict['uni_idx']]

                                       
        # global_db_uni
        db_ft_global_2 = torch.unsqueeze(db_ft_global, 1)
        db_ft_global_3 = torch.repeat_interleave(db_ft_global_2, uni_ft_global.size()[1], 1)
        db_uni_ft_global = torch.cat([db_ft_global_3, uni_ft_global], dim=-1)
        db_uni_ft_global = self.db_uni_linear_global(db_uni_ft_global)
        
        # local_db
        db_node_ft_local = self.db_linear_local(ft_dict['db_graph_node_ft'])
        
        # local_tax
        tax_node_ft_local = self.tax_linear_local(ft_dict['tax_graph_node_ft'])
        
        # local_uni
        uni_node_ft_local = self.uni_linear_local(ft_dict['uni_graph_node_ft'])

        # local
        db_node_ft_local = self.activation(db_node_ft_local)
        tax_node_ft_local = self.activation(tax_node_ft_local)
        uni_node_ft_local = self.activation(uni_node_ft_local)
        uni_padding_local = torch.zeros(1,self.h_dim).to(device)
        uni_node_ft_local = torch.cat([uni_node_ft_local, uni_padding_local], dim=0)

        # local
        db_ft_local = db_node_ft_local[ft_dict['db_idx']]
        tax_ft_local = tax_node_ft_local[ft_dict['tax_idx']]
        uni_ft_local = uni_node_ft_local[ft_dict['uni_idx']]
                                       
        # local_db_uni
        db_ft_local_2 = torch.unsqueeze(db_ft_local, 1)
        db_ft_local_3 = torch.repeat_interleave(db_ft_local_2, uni_ft_local.size()[1], 1)
        db_uni_ft_local = torch.cat([db_ft_local_3, uni_ft_local], dim=-1)
        db_uni_ft_local = self.db_uni_linear_local(db_uni_ft_local)
        
        # cross
        db_uni_ft_cross = db_uni_ft_global + db_uni_ft_local + (db_uni_ft_global * db_uni_ft_local) * self.cross_scale_merge_2
  
        # pred_db_uni
        db_uni_ft_cross = self.activation(db_uni_ft_cross)
        db_uni_ft_cross = F.dropout(db_uni_ft_cross, p=self.dropout, training=self.training)
        db_uni_pred = self.db_uni_pred_linear(db_uni_ft_cross)
        db_uni_pred = torch.sigmoid(db_uni_pred)
        
        uni_mask = torch.unsqueeze(ft_dict['uni_mask'], -1)
        
        db_uni_pred_sum = torch.unsqueeze(torch.sum(db_uni_pred * uni_mask, -2), -1)

        # global_db_tax
        uni_ft_1_global = uni_ft_global * db_uni_pred / db_uni_pred_sum
        uni_ft_2_global = uni_ft_1_global.transpose(1,2)
        uni_ft_3_global = torch.sum(uni_ft_2_global, -1)
        db_tax_ft_global = torch.cat([db_ft_global, tax_ft_global, uni_ft_3_global], dim=1)
        db_tax_ft_global = self.db_tax_uni_linear_global(db_tax_ft_global)    

        # local_db_tax
        uni_ft_1_local = uni_ft_local * db_uni_pred / db_uni_pred_sum
        uni_ft_2_local = uni_ft_1_local.transpose(1,2)
        uni_ft_3_local = torch.sum(uni_ft_2_local, -1)
        db_tax_ft_local = torch.cat([db_ft_local, tax_ft_local, uni_ft_3_local], dim=1)
        db_tax_ft_local = self.db_tax_uni_linear_local(db_tax_ft_local)
    
        # cross
        db_tax_ft_cross = db_tax_ft_global + db_tax_ft_local + (db_tax_ft_global * db_tax_ft_local) * self.cross_scale_merge_1

        # pred_db_tax
        db_tax_ft_cross = self.activation(db_tax_ft_cross)
        db_tax_ft_cross = F.dropout(db_tax_ft_cross, p=self.dropout, training=self.training)
        db_tax_pred = self.db_tax_pred_linear(db_tax_ft_cross)
        db_tax_pred = torch.sigmoid(db_tax_pred)
                                       
        return db_tax_pred, db_uni_pred, db_adj_global, tax_adj_global, uni_adj_global