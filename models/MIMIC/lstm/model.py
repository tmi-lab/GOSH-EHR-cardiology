import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # Output layer with 2 classes (for binary classification)

    def forward(self, input_ids, lengths):
        embeds = self.embedding(input_ids)

        # Pack the embedded sequences to handle variable lengths
        packed_embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, _) = self.lstm(packed_embeds)

        # Use the last hidden state for classification (classification based on the last token)
        last_hidden_state = hidden[-1]  # Use the last layer's hidden state

        # Pass through the fully connected layer to get class logits (2 logits for binary classification)
        logits = self.fc(last_hidden_state)

        return logits