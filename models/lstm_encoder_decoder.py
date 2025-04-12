import torch
import torch.nn as nn

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, bidirectional=True, teacher_forcing=True):
        super(LSTMEncoderDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.teacher_forcing = teacher_forcing

        # LSTM Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)

        # Latent space projection
        self.fc_mu = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, latent_dim)

        # LSTM Decoder
        self.decoder = nn.LSTM(latent_dim + input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        _, (h, _) = self.encoder(x)
        h = h[-1] if not self.bidirectional else torch.cat((h[-2], h[-1]), dim=-1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # Decode
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)  # Repeat z for each time step
        x = torch.cat((x, z), dim=-1)
        output, _ = self.decoder(x)
        output = self.fc_out(output)
        return output, mu, logvar