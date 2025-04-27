from torch import nn, zeros
from device import device

class EncoderRNN(nn.Module):
    """
    RNN encoder module for sequence processing.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initializes the encoder.
        Args:
            input_size (int): Size of input layer
            hidden_size (int): Size of hidden layers
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        Forward pass of the encoder.
        Args:
            input: Input tensor
            hidden: Hidden state tensor
        Returns:
            tuple: Output and hidden state
        """
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        """
        Initializes hidden state with zeros.
        Returns:
            tensor: Zero-initialized hidden state
        """
        return zeros(1, 1, self.hidden_size, device=device)