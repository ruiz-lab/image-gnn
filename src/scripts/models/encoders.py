import torch
import torch.nn as nn

from torch.nn import functional as F

from data.datasets import TimeSeries

from typing import List, Dict


class PosEncoding(nn.Module):
    """
    Positional encoding class for time encoder.
    """

    def __init__(self, hidden_size, dropout=0.0, max_len=3000):
        """
        Initialize the positional encoding module.

        Args:
            hidden_size (int): The dimensionality of the hidden state (i.e., embedding size).
            dropout (float, optional): Dropout probability applied to the positional encoding.
            max_len (int, optional): The maximum length of the input sequence.
        """

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        ii = torch.arange(0, hidden_size, 2)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.pow(10000.0, -(ii / hidden_size))

        pe = torch.zeros((1, max_len, hidden_size))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div) if hidden_size % 2 == 0\
                                              else torch.cos(pos * div[:-1])

        self.pe = pe

    def forward(self, em):
        """
        Apply positional encoding to the input embeddings.

        Args:
            em (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: The input embeddings after adding positional encoding.
        """

        pe = self.pe.to(em.device)

        return self.dropout(em + pe[:, :em.shape[1], :])

class Time2Vec(torch.nn.Module):

    def __init__(
            self, 
            in_features, 
            embedding_dim, 
            activation_fn=torch.sin
        ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.w0 = nn.parameter.Parameter(torch.randn((in_features, 1)))
        self.b0 = nn.parameter.Parameter(torch.randn((in_features, 1)))
        self.w = nn.parameter.Parameter(torch.randn((in_features, embedding_dim-1)))
        self.b = nn.parameter.Parameter(torch.randn((in_features, embedding_dim-1)))
        self.norm = nn.BatchNorm1d(1)
        self.activation_fn = activation_fn

    def forward(self, x):
        v_0 = (torch.matmul(self.w0, x[:, [0]]) + self.b0)
        v = self.activation_fn(torch.matmul(self.w, x[:, 1:]) + self.b)

        out = torch.cat([v_0, v], dim=1)

        return out

class EncoderLSTM(nn.Module):
    """
    Encodes graph format time-series sequence
    """

    def __init__(
        self,
        input_size_past,
        hidden_size,
        num_layers=1,
        dropout=0.1,
        batch_first=False,
        **kwargs,
    ):

        """
        Initialize the EncoderLSTM module.

        Args:
            input_size_past (int): The number of features in the input X.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int, optional): Number of recurrent layers (i.e., stacked LSTM layers).
            dropout (float, optional): Dropout rate between LSTM layers.
            batch_first (bool, optional): If True, the input and output tensors are provided as (batch, seq, feature).
            **kwargs: Additional keyword arguments.
        """

        super().__init__()

        self.input_size_past = input_size_past
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm_past = nn.LayerNorm(input_size_past)
        self.lstm = nn.LSTM(
            input_size=self.input_size_past,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first,
        )

    def forward(
        self,
        x_input: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Forward pass of the EncoderLSTM module.

        Args:
            x_input (List[torch.Tensor]): Input time-series sequences.

        Returns:
            List[torch.Tensor]: Encoded representations of the input sequences.
        """

        out = []
        full_past_features = []

        largest_past_window = max([len(x) for x in x_input])
        for i, x in enumerate(x_input):
            full_past_features.append(
                F.pad(
                    input=x,
                    pad=(0, 0, largest_past_window - x.shape[0], 0),
                    mode="constant",
                    value=0,
                )
            )
        full_past_features = torch.stack(full_past_features, dim=0)

        outputs_past, hidden_past = self.lstm(
            self.layer_norm_past(full_past_features)
        )
        for i, x in enumerate(x_input):
            out.append(
                outputs_past[i, -1]
            )

        return out

class TemporalEncoder(nn.Module):
    """
    Temporal encoder module that combines time encoding and LSTM encoders for time-series sequences.
    """

    def __init__(
        self,
        time_enc_size,
        rnn_config,
        embedding_size,
        num_layers,
        dropout,
        **kwargs
    ):
        """
        Initialize the TemporalEncoder module.

        Args:
            time_enc_size (int): The hidden size of the time encoding module.
            rnn_config (dict): A dictionary specifying the configuration for different types of time-series sequences.
                               The keys are the types of time-series sequences, and the values are dictionaries with the following keys:
                               - "input_size": The input size of the LSTM encoder for the corresponding time-series type.
            embedding_size (int): The size of the output embeddings.
            num_layers (int): The number of layers in the LSTM encoder.
            dropout (float): The dropout rate between LSTM layers.
            **kwargs: Additional keyword arguments.
        """

        super().__init__()

        self.embedding_size = embedding_size
        self.time_encoder = PosEncoding(
            hidden_size=time_enc_size, 
            dropout=dropout
        )

        self.rnn_encoders = nn.ModuleDict()
        for ts_type, config in rnn_config.items():
            self.rnn_encoders[ts_type] = EncoderLSTM(
                input_size_past=config["input_size"] + time_enc_size,
                hidden_size=embedding_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            )

    def forward(self, batch):
        """
        Forward pass of the TemporalEncoder module.

        Args:
            batch (Batch): A batch object containing the time-series sequences.

        Returns:
            torch.Tensor: Encoded representations of the time-series sequences in a graph-format.
        """

        ts_dict : Dict[str, List[TimeSeries]] = batch.ts
        batch_size = batch.y.shape[0]
        device = batch.y.device
        num_series = len(list(ts_dict.keys()))
        embedding_size = self.embedding_size

        x = {}
        for ts_type, ts_list in ts_dict.items():

            x_encoded = []
            for _, ts in enumerate(ts_list):
                encoded_timestamps : torch.Tensor = self.time_encoder(ts.timestamps)
                x_encoded.append(
                    torch.cat((ts.x, encoded_timestamps.squeeze(0)), dim=-1).to(device)
                )

            x[ts_type] = self.rnn_encoders[ts_type](x_encoded)            

        X = torch.zeros(batch_size, num_series, embedding_size)
        for b in range(batch_size):
            for i, key in enumerate(x.keys()):
                X[b, i] = x[key][b]

        return X.to(device)