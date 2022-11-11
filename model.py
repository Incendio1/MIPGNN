from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Linear
import torch
import torch.nn as nn


class AE(nn.Module):

    def __init__(self, n_hidden, n_input, n_z, dropout):
        super(AE, self).__init__()
        self.dropout = dropout
        self.enc_1 = Linear(n_input, n_hidden)
        #         self.enc_2 = Linear(n_hidden, n_hidden)
        self.z_layer = Linear(n_hidden, n_z)

        self.dec_1 = Linear(n_z, n_hidden)
        #         self.dec_2 = Linear(n_hidden, n_hidden)
        self.x_bar_layer = Linear(n_hidden, n_input)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.enc_1.weight)
        #         nn.init.xavier_uniform_(self.enc_2.weight)
        nn.init.xavier_uniform_(self.z_layer.weight)
        nn.init.xavier_uniform_(self.dec_1.weight)
        #         nn.init.xavier_uniform_(self.dec_2.weight)
        nn.init.xavier_uniform_(self.x_bar_layer.weight)
        nn.init.normal_(self.enc_1.bias, std=1e-6)
        #         nn.init.normal_(self.enc_2.bias, std=1e-6)
        nn.init.normal_(self.z_layer.bias, std=1e-6)
        nn.init.normal_(self.dec_1.bias, std=1e-6)
        #         nn.init.normal_(self.dec_2.bias, std=1e-6)
        nn.init.normal_(self.x_bar_layer.bias, std=1e-6)

    def reset_parameters(self):
        self.enc_1.reset_parameters()
        #         self.enc_2.reset_parameters()
        self.z_layer.reset_parameters()
        self.dec_1.reset_parameters()
        #         self.dec_2.reset_parameters()
        self.x_bar_layer.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        enc_h1 = F.relu(self.enc_1(x))
        enc_h1 = F.dropout(enc_h1, p=self.dropout, training=self.training)
        #         enc_h2 = F.relu(self.enc_2(enc_h1))
        #         enc_h2 = F.dropout(enc_h2, p=dropout, training=self.training)

        z = self.z_layer(enc_h1)
        z_drop = F.dropout(z, p=self.dropout, training=self.training)

        dec_h1 = F.relu(self.dec_1(z_drop))
        dec_h1 = F.dropout(dec_h1, p=self.dropout, training=self.training)
        #         dec_h2 = F.relu(self.dec_2(dec_h1))
        #         dec_h2 = F.dropout(dec_h2, p=dropout, training=self.training)
        x_bar = self.x_bar_layer(dec_h1)

        return x_bar, z


class prop_sum(MessagePassing):
    def __init__(self, num_classes, layers, alpha, **kwargs):
        super(prop_sum, self).__init__(aggr='add', **kwargs)
        self.layers = layers
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight):
        z = x
        embed_layer = []
        embed_layer.append(x)

        if(self.layers != [0]):
            for layer in self.layers:
                # edge_weight[layer - 1] = edge_weight[layer - 1]/torch.sum(edge_weight[layer - 1])
                h = (1 - self.alpha) * self.propagate(edge_index[layer - 1], x=x, norm=edge_weight[layer - 1]) + self.alpha * z
                embed_layer.append(h)

        embed_layer = torch.stack(embed_layer, dim=1)

        return embed_layer

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        pass


class prop_weight(MessagePassing):
    def __init__(self, num_classes, layers, alpha, **kwargs):
        super(prop_weight, self).__init__(aggr='add', **kwargs)

        self.weight = torch.nn.Parameter(torch.ones(len(layers) + 1), requires_grad=True)

        self.layers = layers
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight):
        z = x
        embed_layer = []
        embed_layer.append(self.weight[0] * x)

        for i in range(len(self.layers)):
            h = (1 - self.alpha) * self.propagate(edge_index[self.layers[i] - 1], x=x, norm=edge_weight[self.layers[i] - 1]) + self.alpha * z
            embed_layer.append(self.weight[i + 1] * h)


        embed_layer = torch.stack(embed_layer, dim = 1)
        return embed_layer

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        self.weight = torch.Parameter(torch.ones(len(self.layers) + 1), requires_grad=True)


class MIPGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MIPGNN, self).__init__()
        self.args = args
        self.agg = args.agg

        # self.AE = AE(args.hidden1, dataset.num_features, dataset.num_classes, dropout=args.dropout)

        self.lin1 = Linear(dataset.num_features, self.args.hidden)
        self.lin2 = Linear(self.args.hidden, dataset.num_classes)

        if (self.agg == 'sum'):
            self.prop = prop_sum(dataset.num_classes, self.args.layers, self.args.alpha)
        if (self.agg =='weighted_sum'):
            self.prop = prop_weight(dataset.num_classes, self.args.layers, self.args.alpha)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        # self.AE.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.args.dropout1, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.args.dropout2, training=self.training)
        x = self.lin2(x)

        # x_bar, x = self.AE(x)

        x = self.prop(x, self.args.hop_edge_index, self.args.hop_edge_att)
        x = torch.sum(x, dim=1)

        # return x_bar, F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)
