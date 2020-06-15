from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, use_bn=True, input_dropout=0., hidden_dropout=0.):
        super().__init__()

        layers = []
        for index, dim in enumerate(hidden_dims[:-1]):
            layers.append(nn.Linear(input_dim, dim, bias=True))
            nn.init.normal_(layers[-1].weight, std=0.01)
            nn.init.constant_(layers[-1].bias, 0.)

            if index < len(hidden_dims) - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.LeakyReLU(negative_slope=0.2))
            if input_dropout and index == 0:
                layers.append(nn.Dropout(p=input_dropout))
            elif hidden_dropout and index < len(hidden_dims) - 1:
                layers.append(nn.Dropout(p=hidden_dropout))

            input_dim = dim

        layers.append(nn.Linear(input_dim, hidden_dims[-1]))
        nn.init.normal_(layers[-1].weight, std=0.01)
        nn.init.constant_(layers[-1].bias, 0.)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
