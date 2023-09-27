from torchfm.layer import CrossNetwork, MultiLayerPerceptron
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np



class SelectionNetwork(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(SelectionNetwork, self).__init__()

    
        self.mlp =  MultiLayerPerceptron(input_dim=input_dims,
                                        embed_dims=[output_dims], output_layer=False, dropout=0.0)
        self.weight_init(self.mlp)
                                        
    def forward(self, input_mlp):
        output_layer = self.mlp(input_mlp)
        return torch.softmax(output_layer, dim=1)
      

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            

class DeepCrossNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = x
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        return p


class EMB(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x).transpose(1,2)

    
    
  
class MvFS_Controller(nn.Module):
    def __init__(self, input_dim, embed_dims, num_selections):
        super().__init__()
        self.inputdim = input_dim
        self.num_selections = num_selections

        self.T = 1

        self.gate = nn.Sequential(nn.Linear(embed_dims * num_selections , num_selections))
        
        self.SelectionNetworks = nn.ModuleList(
            [SelectionNetwork(input_dim, embed_dims) for i in range(num_selections)]
        )
    def forward(self, emb_fields):

        input_mlp = emb_fields.flatten(start_dim=1).float()
        importance_list= []
        for i in range(self.num_selections):
            importance_vector = self.SelectionNetworks[i](input_mlp)
            importance_list.append(importance_vector)

            
        gate_input = torch.cat(importance_list, 1)
        selection_influence = self.gate(gate_input)
        selection_influence = torch.sigmoid(selection_influence)
        
        scores = None
        for i in range(self.num_selections):
            score = torch.mul(importance_list[i], selection_influence[:,i].unsqueeze(1))
            if i == 0 :
                scores = score
            else:
                scores = torch.add(scores, score)
                
                
                
        scores =0.5 * (1+ torch.tanh(self.T*(scores-0.1)))
        
        if self.T < 5:
            self.T += 0.001
        return scores
    
    
class MvFS_DCN(nn.Module): 
    def __init__(self,field_dims, embed_dim, num_selections):
        super().__init__()
        self.num = len(field_dims)
        self.embed_dim = embed_dim
        self.emb = EMB(field_dims[:self.num],self.embed_dim)
        self.dcn = DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=[16,8], dropout=0.2)
        
        self.controller = MvFS_Controller(input_dim=len(field_dims)*self.embed_dim, 
                                         embed_dims=len(field_dims), num_selections= num_selections)
  
        self.weight = 0
        self.stage = -1
       

    def forward(self, field):
        field = self.emb(field)

        
        if self.stage == 1: # use controller
 
            self.weight = self.controller(field)
            selected_field = field * torch.unsqueeze(self.weight,1)
    
            input_mlp = selected_field
        else: # only backbone 
            input_mlp = field
        input_mlp = input_mlp.flatten(start_dim=1).float()
        res = self.dcn(input_mlp)

        return torch.sigmoid(res.squeeze(1))