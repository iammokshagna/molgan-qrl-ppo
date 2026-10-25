import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDenseLayer(nn.Module):
    def __init__(self, aux_unit, linear_units, activation=None, dropout_rate=0.):
        super(MultiDenseLayer, self).__init__()
        layers = []
        for c0, c1 in zip([aux_unit] + linear_units[:-1], linear_units):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout_rate))
            if activation is not None:
                layers.append(activation)
        self.linear_layer = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.linear_layer(inputs)


class Generator(nn.Module):
    """Generator/Decoder network for molecular graphs."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate=0.0):
        super(Generator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, self.activation_f)

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class GraphEncoder(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) Encoder
    Encodes molecular graph (Adjacency A and Node features X) into latent distribution parameters (mu, logvar).
    """
    def __init__(self, vertexes=35, edges=5, nodes=10, conv_dims=[128, 64], z_dim=32):
        super(GraphEncoder, self).__init__()
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.z_dim = z_dim

        # Relational Graph Convolution Layer 1
        self.w_self1 = nn.Linear(nodes, conv_dims[0])
        self.w_edges1 = nn.ModuleList([nn.Linear(nodes, conv_dims[0]) for _ in range(edges)])

        # Relational Graph Convolution Layer 2
        self.w_self2 = nn.Linear(conv_dims[0], conv_dims[1])
        self.w_edges2 = nn.ModuleList([nn.Linear(conv_dims[0], conv_dims[1]) for _ in range(edges)])

        # Attention-based graph pooling
        self.gate = nn.Linear(conv_dims[1], 1)
        self.pool_proj = nn.Linear(conv_dims[1], 128)

        # Latent heads
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_logvar = nn.Linear(128, z_dim)

    def forward(self, a_tensor, x_tensor):
        # a_tensor: [B, N, N, E] (one-hot), x_tensor: [B, N, M] (one-hot)
        B, N, _, E = a_tensor.shape

        # Layer 1
        h1 = self.w_self1(x_tensor)
        for b in range(1, E):
            adj_b = a_tensor[:, :, :, b]
            h_b = torch.matmul(adj_b, x_tensor)
            h1 = h1 + self.w_edges1[b](h_b)
        h1 = F.relu(h1)

        # Layer 2
        h2 = self.w_self2(h1)
        for b in range(1, E):
            adj_b = a_tensor[:, :, :, b]
            h_b = torch.matmul(adj_b, h1)
            h2 = h2 + self.w_edges2[b](h_b)
        h2 = F.relu(h2)

        # Readout: Attention pooling across the 35 atoms
        gate_weights = F.softmax(self.gate(h2), dim=1)
        graph_emb = torch.sum(gate_weights * h2, dim=1)
        graph_proj = F.relu(self.pool_proj(graph_emb))

        mu = self.fc_mu(graph_proj)
        logvar = self.fc_logvar(graph_proj)
        return mu, logvar


class GraphVAE(nn.Module):
    """
    Graph Variational Autoencoder (VAE)
    Encoder: R-GCN + Attention Pooling -> mu, logvar -> z (32-D)
    Decoder: Dense MLP (identical architecture to MolGAN Generator) -> Edges logits & Nodes logits
    """
    def __init__(self, vertexes=35, edges=5, nodes=10, z_dim=32, g_conv_dims=[256, 512, 1024]):
        super(GraphVAE, self).__init__()
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.z_dim = z_dim

        self.encoder = GraphEncoder(vertexes=vertexes, edges=edges, nodes=nodes, z_dim=z_dim)
        self.decoder = Generator(conv_dims=g_conv_dims, z_dim=z_dim, vertexes=vertexes, edges=edges, nodes=nodes, dropout_rate=0.0)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, a_tensor, x_tensor):
        mu, logvar = self.encoder(a_tensor, x_tensor)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, a_tensor, x_tensor):
        mu, logvar = self.encode(a_tensor, x_tensor)
        z = self.reparameterize(mu, logvar)
        edges_logits, nodes_logits = self.decode(z)
        return edges_logits, nodes_logits, mu, logvar, z
