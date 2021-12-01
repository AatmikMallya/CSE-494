import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, embedding, args):
        super(Net, self).__init__()

        # Embedding Layer, you can leave this alone
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino-1)
        if not (args.blosum is None or args.blosum.lower() == 'none'):
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

        # Sizes for concatenating
        self.net_pep_dim = args.max_len_pep * self.embedding_dim
        self.net_tcr_dim = args.max_len_tcr * self.embedding_dim
        self.net_cat_dim = self.net_pep_dim + self.net_tcr_dim

        # Neural Network Layer
        self.net = nn.Sequential(
            nn.Linear(self.net_cat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # Necessary to receive value between 0 and 1
        )

    def forward(self, pep, tcr):
        
        # The sequence length is set to 25 by default
        # The batch size is set to 32 by default
        # Your starting parameters are both the same size
        # (batch_size * sequence_len)
        
        # Embedding Layer
        # Learns latent features from the input and transforms into 3-dimensional tensor
        # (batch_size * sequence_len * embedding_dim)
        pep = self.embedding(pep) 
        tcr = self.embedding(tcr)
        
        # Flatten the 3-dimensional tensor to create 2-dimensional matrix
        # (batch_size * 1 * (sequence_len * embedding_dim))
        pep = pep.reshape(-1, 1, pep.size(-2) * pep.size(-1))
        tcr = tcr.reshape(-1, 1, tcr.size(-2) * tcr.size(-1))
        
        # Concatenate both inputs into a larger matrix
        # Squeeze removes any dimensions which are of size 1
        # (batch_size * (2 * sequence_len * embedding_dim))
        peptcr = torch.cat((pep, tcr), -1).squeeze(-2)

        # Neural network on the concatenated matrix
        peptcr = self.net(peptcr)

        # Your shape before returning should be (batch_size * 1)
        return peptcr
