from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttnDecoderRNN(nn.Module):
    """
    Attention-based decoder RNN module for sequence generation.
    """
    def __init__(self, hidden_size, output_size, args):
        """
        Initializes the attention decoder.
        Args:
            hidden_size (int): Size of hidden layers
            output_size (int): Size of output layer
            args: Additional configuration arguments
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = args.dropout

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, args.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        Forward pass of the attention decoder.
        Args:
            input: Input tensor
            hidden: Hidden state tensor
            encoder_outputs: Outputs from encoder
        Returns:
            tuple: Output and hidden state
        """
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        """
        Initializes hidden state with zeros.
        Returns:
            tensor: Zero-initialized hidden state
        """
        return torch.zeros(1, 1, self.hidden_size, device=device)