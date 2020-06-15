import logging

import torch
from torch import nn

from .mlp import MLP
from .word import get_embeddings

logger = logging.getLogger(__name__)


class AdvAutoEncoder(nn.Module):

    def __init__(
        self,
        dataset,
        embeddings=None,
        encoder_config=None,
        decoder_config=None,
        discriminator_config=None,
        noise_dimension=50,
        noise_type="normal",
        device=None
    ):
        super().__init__()

        self.noise_dimension = noise_dimension
        self.noise_type = noise_type

        self.emb, _ = get_embeddings(dataset, embeddings, True)
        semantic_dim = self.emb.weight.shape[1]
        logger.info(f"Semantic dimension: {semantic_dim}.")

        if encoder_config is None:
            self.encoder = identity
        else:
            assert encoder_config["hidden_dims"][-1] == noise_dimension
            self.encoder = MLP(**encoder_config)

        decoder_config["input_dim"] = noise_dimension + semantic_dim
        assert decoder_config["hidden_dims"][-1] == encoder_config["input_dim"]
        self.decoder = MLP(**decoder_config)

        discriminator_config["input_dim"] = noise_dimension
        assert discriminator_config["hidden_dims"][-1] == 1
        self.discriminator = MLP(**discriminator_config)

        self.to(device)
        self.device = device

    def forward(self, words, real_features):
        attributes = self.emb(words)

        pred_noise = self.encoder(real_features)
        noise = self.get_noise(len(real_features))

        pred_x = self.decoder(torch.cat(attributes, pred_noise, dim=-1))
        # decode = self.decoder(torch.cat(attributes, noise, dim=-1))

        outputs = {"reconstruction": pred_x}
        outputs["fake_dis"] = self.discriminator(pred_noise)
        outputs["true_dis"] = self.discriminator(noise)

        return outputs

    def generate(self, words):
        attributes = self.emb(words)
        noise = self.get_noise(len(words))
        return self.decoder(torch.cat((attributes, noise), dim=-1))

    def get_noise(self, amount):
        if self.noise_type == "normal":
            return torch.randn(amount, self.noise_dimension).to(self.device)
        elif self.noise_type == "uniform":
            return torch.rand(amount, self.noise_dimension).to(self.device)
        else:
            raise ValueError(f"Unknown noise type {self.noise_type}.")


def identity(x):
    return x
