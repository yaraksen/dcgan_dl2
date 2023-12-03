import torch
from torch import nn
from torch import Tensor
import math
from numpy import prod

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MurkyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, max_len: int, dropout: float, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True, norm_first=True)
        layer_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers, norm=layer_norm)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, vocab_size)
        self.device = device
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(self.device)
        output = self.transformer_encoder(src, mask=src_mask, is_causal=True)
        output = self.linear(output)
        return output

    def __str__(self):
        """
        Model prints with number of trainable parameters
        Taken from our DLA homework template
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)