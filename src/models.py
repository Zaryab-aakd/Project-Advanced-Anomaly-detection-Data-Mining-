import torch
import torch.nn as nn

class TransformerGenerator(nn.Module):
    def __init__(self, feat_dim, d_model=64, n_heads=4, num_layers=2):
        super(TransformerGenerator, self).__init__()
        self.embedding = nn.Linear(feat_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, feat_dim)

    def forward(self, src):
        x = self.embedding(src) + self.pos_encoder
        memory = self.transformer_encoder(x)
        output = self.transformer_decoder(tgt=x, memory=memory)
        return self.output_layer(output)

class Discriminator(nn.Module):
    def __init__(self, feat_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.net(x)
