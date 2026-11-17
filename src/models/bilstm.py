"""Bidirectional LSTM on the raw ECG (E2).

Follows the recurrent model of the PTB-XL benchmark (RNN1d in Strodthoff et al., 2021):
the 12 leads at every time step are the input vector of a stacked LSTM, here
bidirectional; the outputs over time are pooled by average, maximum and the last
state of each direction (final forward state, first backward state) and classified by
the same MLP head as the ResNets (heads.py). There are no convolutions.

    hidden   LSTM hidden size per direction (benchmark: 256)
    layers   stacked LSTM layers (benchmark: 2)
    fp32_lstm  run the recurrence in float32 even under autocast. Under bf16 autocast,
             PyTorch runs the cuDNN LSTM in float16 (not bf16), without loss scaling; over
             250 time steps this distorts the gradients considerably (E2). The rest of
             the model still uses bf16, as every other model does.
"""

import torch
from torch import nn

from src.models.heads import mlp_head


class BiLSTM(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden: int = 256, layers: int = 2,
                 fp32_lstm: bool = True, ps_head: float = 0.5, lin_ftrs_head: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.fp32_lstm = fp32_lstm
        self.lstm = nn.LSTM(in_channels, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = mlp_head(3 * 2 * hidden, num_classes, lin_ftrs_head, ps_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp32_lstm:
            with torch.autocast(x.device.type, enabled=False):
                out, _ = self.lstm(x.float().transpose(1, 2))     # (batch, time, 2 * hidden)
        else:
            out, _ = self.lstm(x.transpose(1, 2))
        h = self.lstm.hidden_size
        last = torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=1)  # end of each direction
        return self.head(torch.cat([out.mean(1), out.amax(1), last], dim=1))


def bilstm(in_channels: int, num_classes: int, **kwargs) -> BiLSTM:
    return BiLSTM(in_channels, num_classes, **kwargs)
