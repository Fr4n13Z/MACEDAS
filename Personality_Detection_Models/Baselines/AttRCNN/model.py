import argparse

import torch
import torch.nn as nn


class AttRCNN(nn.Module):
    def __init__(self, args: argparse.Namespace, device):
        super(AttRCNN, self).__init__()
        self.args = args
        self.S = args.max_posts
        self.device = device
        # self.refuse_linear = nn.Linear(768, 100)
        self.W = args.max_length // 2
        self.embedding_dim = args.embedding_dim

        # self.embed = nn.Embedding(args.vocab_size, args.embedding_dim)
        # self.embed.weight.data.copy_(args.embedding_weight)

        # hidden_dim 50 n_layers 1
        self.gru1 = nn.GRU(input_size=args.embedding_dim, hidden_size=args.hidden_dim, num_layers=args.n_layers,
                           batch_first=True)
        self.mult_attention1 = nn.MultiheadAttention(embed_dim=args.hidden_dim, num_heads=1, dropout=0.5)

        self.gru1_bn = nn.Sequential(nn.BatchNorm2d(num_features=args.hidden_dim),
                                     nn.ReLU()
                                     )

        self.gru2 = nn.GRU(input_size=args.embedding_dim, hidden_size=args.hidden_dim, num_layers=args.n_layers,
                           batch_first=True)
        self.mult_attention2 = nn.MultiheadAttention(embed_dim=args.hidden_dim, num_heads=1, dropout=0.5)

        self.gru2_bn = nn.Sequential(nn.BatchNorm2d(num_features=args.hidden_dim),
                                     nn.ReLU(),
                                     )

        self.seqEncoder = nn.Sequential(
            nn.Linear(in_features=args.embedding_dim + self.args.hidden_dim * 2, out_features=100),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(50, 1))
        )

        self.conv1 = nn.Conv1d(in_channels=100, out_channels=50, kernel_size=1)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=1),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=3, padding=(1,))
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, padding=(1,)),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=5, padding=(2,))
        )

        self.conv4 = nn.Conv1d(in_channels=100, out_channels=50, kernel_size=3, padding=(1,))

        self.line1 = nn.Sequential(

            nn.Linear(in_features=self.S * 200, out_features=100),
            nn.Dropout(0.5),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU()
        )

        self.line2 = nn.Sequential(

            nn.Linear(in_features=100, out_features=50),
            nn.Dropout(0.5),
            nn.BatchNorm1d(num_features=50),
            nn.ReLU()
        )

        self.line3 = nn.ModuleList(nn.Linear(in_features=50, out_features=2) for _ in range(self.args.num_labels))

    def forward(self, x):
        x1 = x[:, :, 0:int(x.shape[2] / 2)]
        x2 = x[:, :, int(x.shape[2] / 2): x.shape[2]]

        # x1 = self.embed(x1)
        # x2 = self.embed(x2)

        BATCH_SIZE_TMP = x1.shape[0]
        embedding_dim = self.embedding_dim

        x1 = x1.view(BATCH_SIZE_TMP * self.S, self.W, embedding_dim)
        x2 = x2.view(BATCH_SIZE_TMP * self.S, self.W, embedding_dim)
        x3 = x1.view(BATCH_SIZE_TMP * self.S, self.W, embedding_dim)

        x1, states_1 = self.gru1(x1)
        x2, states_2 = self.gru1(x2)

        x1 = x1.view(BATCH_SIZE_TMP, self.S, self.W, self.args.hidden_dim)
        x2 = x2.view(BATCH_SIZE_TMP, self.S, self.W, self.args.hidden_dim)

        x1 = self.gru1_bn(x1)
        x2 = self.gru2_bn(x2)

        x1 = x1.view(BATCH_SIZE_TMP * self.S, self.W, self.args.hidden_dim)
        x2 = x2.view(BATCH_SIZE_TMP * self.S, self.W, self.args.hidden_dim)

        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)

        weight1 = torch.zeros((self.W, BATCH_SIZE_TMP * self.S, self.args.hidden_dim)).to(self.device)
        weight2 = torch.zeros((self.W, BATCH_SIZE_TMP * self.S, self.args.hidden_dim)).to(self.device)

        x1, attention_out = self.mult_attention1(key=x1, value=x1, query=weight1)
        x1 = x1.permute(1, 0, 2)

        x2, attention_out2 = self.mult_attention2(key=x2, value=x2, query=weight2)
        x2 = x2.permute(1, 0, 2)

        x = torch.cat((x1, x3, x2), dim=2)

        x = self.seqEncoder(x)

        x = x.squeeze(1)
        x = x.view(BATCH_SIZE_TMP, self.S, 100)

        x = x.permute(0, 2, 1)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)
        x = self.line1(x)
        x = self.line2(x)
        x = torch.stack([classifier(x) for classifier in self.line3],
                        dim=1).view(-1, 2)

        return x
