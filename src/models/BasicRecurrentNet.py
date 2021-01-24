import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.weights_initalizer import weights_init

input_size = 20
num_layers = 3
hidden_size = 6
num_classes = 20


class BasicRNN(nn.Module):
    def __init__(self):
        super(BasicRNN, self).__init__()
        self.activation = nn.PReLU()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size * 34, out_features=num_classes)

    def forward(self, x: t.Tensor):
        h0 = Variable(t.zeros(
            num_layers,
            x.size(0),
            hidden_size
        ))
        h0 = h0.cuda()

        c0 = Variable(t.zeros(
            num_layers,
            x.size(0),
            hidden_size
        ))
        c0 = c0.cuda()

        x = x.squeeze(1).permute(0, 2, 1)

        output, (hn, cn) = self.lstm(x, (h0, c0))
        # out = hn[-1, :, :].view(-1, hidden_size)
        out = output.contiguous().view(output.size(0), -1)
        # out = self.activation(out)
        out = self.fc(out)

        return F.log_softmax(out, dim=1)
