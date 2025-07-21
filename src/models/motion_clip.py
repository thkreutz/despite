import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

### Credit: Copy from MotionClip https://github.com/GuyTevet/MotionCLIP/src/models

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 activation="gelu", **kargs):
        super().__init__()
        
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        
        self.input_feats = self.njoints * self.nfeats

        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, batch):


        ### => Need this format for the batch
        # Need to some preprocessing here
        seq_length = batch.shape[1]
        # The mask (to specify which frames are valid)
        mask = torch.ones((batch.shape[0], seq_length), dtype=torch.long).long().to("cuda")
        # Example output (mu and sigma) for the encoder
        batch = {"x": batch.permute(0, 2, 3, 1), "y": torch.zeros(batch.shape[0]).long().to("cuda"), "mask": mask}
        #with torch.no_grad():
        #    out = encoder(batch)
        #    emb = out["mu"]
        #    emb /= emb.norm(dim=-1, keepdim=True)

        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # Blank Y to 0's , no classes in our model, only learned token
        y = y - y
        xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)

        maskseq = torch.cat((muandsigmaMask, mask), axis=1).bool()

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        mu = final[0]
        logvar = final[1] ## vae not used in the paper??
        return {"mu": mu}


class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
    
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
                
        self.input_feats = self.njoints * self.nfeats


        self.actionBiases = nn.Parameter(torch.randn(1, self.latent_dim))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, batch, use_text_emb=False):
        z, y, mask = batch["z"], batch["y"], batch["mask"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        z = z[None]  # sequence of size 1  #

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)

        return output
