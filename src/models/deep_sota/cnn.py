import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5,
                 embedding_class='Word2Vec',
                 embedding_training=False):
        super(CNN, self).__init__()

        self.embedding_class = embedding_class

        if self.embedding_class != 'BERT':
            self.embedding = nn.Embedding(pretrained_embedding.shape[0], pretrained_embedding.shape[1])
            embedding_tensor = torch.from_numpy(pretrained_embedding).float()
            self.embedding.weight = nn.Parameter(embedding_tensor, requires_grad=embedding_training)

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.embedding_class != 'BERT':
            x = self.embedding(x).float()
        x_reshaped = x.permute(0, 2, 1)

        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        logits = self.fc(self.dropout(x_fc))

        return logits
