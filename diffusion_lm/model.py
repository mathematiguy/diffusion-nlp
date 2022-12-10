import torch
import torch.nn as nn

from .utils import timestep_embedding
from transformers import BertTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


class DiffusionLM(nn.Module):
    def __init__(
        self,
        base_model="bert-base-uncased",
        T=2000,  # diffusion steps
        d=16,  # embedding dimensions
        lr=1e-4,
        dropout=0.1,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, d)
        self.bert_config = BertConfig()
        self.encoder = BertEncoder(self.bert_config)
        self.hidden_dim = d
        self.diffusion_steps = T
        self.time_embed_dim = 4 * d
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.bert_config.hidden_size
        self.LayerNorm = nn.LayerNorm(
            self.hidden_size, eps=self.bert_config.layer_norm_eps
        )

        # Add position embeddings
        self.register_buffer(
            "position_ids",
            torch.arange(self.bert_config.max_position_embeddings).expand((1, -1)),
        )
        self.position_embeddings = nn.Embedding(
                self.bert_config.max_position_embeddings, self.hidden_size
        )

        # Add time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(d, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.hidden_size),
        )

        # Downsample input vector
        self.input_projection = nn.Sequential(
            nn.Linear(d, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Downsample output vector
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, d),
        )

    def get_embedding(self, text):
        '''
        Returns the downsampled token embedding sequence for the given `words`
        '''
        # Convert text to tokens
        tokens = model.tokenizer(text, return_tensors="pt")["input_ids"]
        seq_length = tokens.size(1)

        # Get d-dimensional token embeddings
        embeddings = self.embedding(tokens)
        return embeddings

    def forward_diffusion(self, x, T):
        pass

    def forward(self, text, timestep):

        # Convert text to tokens
        embeddings = self.get_embedding(text)
        seq_length = embeddings.size(1)

        # Upsample to `hidden_size` dimensional embeddings
        upsampled = self.input_projection(embeddings)
        print(f"upsampled.shape: {upsampled.shape}")

        # Add timestep embedding + unroll across each sequence
        timesteps = self.time_embedding(timestep_embedding(timestep, self.hidden_dim))
        timesteps = timesteps.unsqueeze(1).expand(-1, seq_length, -1)
        print(f"timestep.shape: {timesteps.shape}")

        # Calculate positional embedding
        position_embeddings = self.position_embeddings(
            self.position_ids[:, :seq_length]
        )
        print(f"position_embeddings.shape: {position_embeddings.shape}")

        # Apply dropout + layernorm
        encoder_inputs = self.dropout(
            self.LayerNorm(upsampled + timesteps + position_embeddings)
        )

        # Get `hidden_size`-dimensional bert representation
        representations = self.encoder(encoder_inputs).last_hidden_state
        print(f"representations.shape: {representations.shape}")

        # Downsample to d-representation
        downsampled = self.output_projection(representations)

        return downsampled

    def fit(self, train_dataset, epochs, batch_size):

        for epoch in range(epochs):

            for batch in train_dataset:

                for text in batch:

                    # Embed the provided text
                    embedding = self.get_embedding(text)

                    # Calculate the forward diffusion steps
                    diffused_embeddings = self.forward_diffusion(embedding, T=self.diffusion_steps)
