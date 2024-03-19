import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramCNN(nn.Module):
    def __init__(self, embedding_dim, embedding, num_classes, num_filters, window_size, embedding_training=False,
                 embedding_class='Word2Vec'):
        super(BigramCNN, self).__init__()
        self.embedding_class = embedding_class

        # Since no need for embedding layer when bert embeddings are directly passed as inputs
        if self.embedding_class != 'BERT':
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            embedding_tensor = torch.from_numpy(embedding).float()
            self.embedding.weight = nn.Parameter(embedding_tensor, requires_grad=embedding_training)

        self.conv = nn.Conv2d(1, num_filters, (window_size, embedding_dim))
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        if self.embedding_class != 'BERT':
            x = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, sequence_length, embedding_dim]
        x = F.relu(self.conv(x))  # [batch_size, num_filters, sequence_length - window_size + 1, 1]
        x = F.max_pool2d(x, (x.size(2), 1)).squeeze()  # [batch_size, num_filters]
        x = self.fc(x)  # [batch_size, num_classes]
        return x
