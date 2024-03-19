import torch
from torch import nn
from torch.autograd import Variable


class LstmClassification(nn.Module):
    def __init__(self, embed_size, embedding, hidden_size=128, out_features=2, embedding_training=False, batch_size=4,
                 embedding_class='Word2Vec'):
        super(LstmClassification, self).__init__()
        self.embedding_class = embedding_class

        if self.embedding_class != 'BERT':
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            embedding_tensor = torch.from_numpy(embedding).float()
            self.embedding.weight = nn.Parameter(embedding_tensor, requires_grad=embedding_training)

        self.bilstm = nn.LSTM(embedding.shape[1], hidden_size)
        self.hidden2label = nn.Linear(hidden_size, out_features)

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = self.init_hidden()
        self.lstm_reduce_by_mean = self.__dict__.get("lstm_mean", True)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return h0, c0

    def forward(self, inputs):
        x = inputs

        if self.embedding_class != 'BERT':
            x = self.embedding(x)
        x = x.permute(1, 0, 2)  # we do this because the default parameter of lstm is False
        self.hidden = self.init_hidden(inputs.size()[0])  # 2x64x64
        lstm_out, self.hidden = self.bilstm(x, self.hidden)  # lstm_out:200x64x128
        if self.lstm_reduce_by_mean == "mean":
            out = lstm_out.permute(1, 0, 2)
            final = torch.mean(out, 1)
        else:
            final = lstm_out[-1]
        y = self.hidden2label(final)  # 64x3  #lstm_out[-1]
        return y


class BiLSTM(nn.Module):
    def __init__(self, embed_size, embedding, hidden_size=128, dropout=0.1, out_features=2, embedding_training=False,
                 lstm_layers=1, keep_dropout=False, bidirectional=True, batch_size=4, embedding_class='Word2Vec'):
        super(BiLSTM, self).__init__()
        self.embedding_class = embedding_class

        if self.embedding_class != 'BERT':
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            embedding_tensor = torch.from_numpy(embedding).float()
            self.embedding.weight = nn.Parameter(embedding_tensor, requires_grad=embedding_training)

        self.bilstm = nn.LSTM(embedding.shape[1], hidden_size // 2, num_layers=lstm_layers, dropout=dropout,
                              bidirectional=bidirectional)
        self.hidden2label = nn.Linear(hidden_size, out_features)

        self.batch_size = batch_size

        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size

        self.hidden = self.init_hidden()
        self.lstm_reduce_by_mean = self.__dict__.get("lstm_mean", True)

        self.lstm_reduce_by_mean = 0.0  # TODO: add default value

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.hidden_size // 2).cuda())
            c0 = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.hidden_size // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.hidden_size // 2))
            c0 = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.hidden_size // 2))
        return h0, c0

    def forward(self, inputs):
        x = inputs
        if self.embedding_class != 'BERT':
            x = self.embedding(x)

        x = x.permute(1, 0, 2)  # we do this because the default parameter of lstm is False
        self.hidden = self.init_hidden(inputs.size()[0])  # 2x64x64
        lstm_out, self.hidden = self.bilstm(x, self.hidden)  # lstm_out:200x64x128
        if self.lstm_reduce_by_mean == "mean":
            out = lstm_out.permute(1, 0, 2)
            final = torch.mean(out, 1)
        else:
            final = lstm_out[-1]
        y = self.hidden2label(final)  # 64x3  #lstm_out[-1]
        return y
