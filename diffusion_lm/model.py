import pdb
import logging

from tqdm import tqdm

import einops
import numpy as np

import torch
import torch.nn as nn

from diffusion_lm.utils import timestep_embedding, diffusion_noise_schedule
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
        device='cpu'
    ):
        super().__init__()
        self.device = device
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

        # Add time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(d, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.hidden_size),
        )

        # Calculate timestep embeddings
        self.timestep_embeddings = self.get_timestep_embeddings()

        # Add position embeddings
        self.register_buffer(
            "position_ids",
            torch.arange(self.bert_config.max_position_embeddings).expand((1, -1)),
        )
        self.position_embeddings = nn.Embedding(
            self.bert_config.max_position_embeddings, self.hidden_size
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

        self.to(self.device)

    def get_timestep_embeddings(self):
        timesteps = torch.arange(self.diffusion_steps)
        timesteps = timestep_embedding(timesteps, self.hidden_dim)
        timesteps = self.time_embedding(timesteps)
        return timesteps.to(self.device)

    def q_sample(self, x, timesteps):
        """
        Otherwise known as q
        """

        batch_size, seq_length, embed_dim = x.shape

        # Calculate and propagate noise schedule
        beta_t = torch.Tensor(diffusion_noise_schedule(timesteps)).to(self.device)

        beta_t = einops.repeat(
            beta_t, "b -> b w x", b=batch_size, w=seq_length, x=embed_dim
        )

        q_t = torch.normal((1 - torch.sqrt(beta_t)) * x, std=1 - torch.sqrt(1 - beta_t))

        return q_t

    def forward(self, embeddings, timesteps):
        """
        Otherwise known as p
        """

        # Convert text to tokens
        batch_size, seq_length, embed_dim = embeddings.shape

        # Upsample to `hidden_size` dimensional embeddings
        upsampled = self.input_projection(embeddings)
        logging.debug(f"upsampled.shape: {upsampled.shape}")

        # Add timestep embedding + unroll across each sequence
        timestep_embeddings = self.timestep_embeddings[timesteps]
        timestep_embeddings = einops.repeat(
            timestep_embeddings, "b e -> b s e", s=seq_length
        )
        logging.debug(f"timestep.shape: {timesteps.shape}")

        # Calculate positional embedding
        position_embeddings = self.position_embeddings(
            self.position_ids[:, :seq_length]
        )
        position_embeddings = einops.repeat(
            position_embeddings, "1 s x -> b s x", b=batch_size
        )
        logging.debug(f"position_embeddings.shape: {position_embeddings.shape}")

        # Apply dropout + layernorm
        encoder_inputs = self.dropout(
            self.LayerNorm(upsampled + timestep_embeddings + position_embeddings)
        )

        encoded = self.encoder(encoder_inputs).last_hidden_state
        logging.debug(f"encoded.shape: {encoded.shape}")

        # Downsample to d-representation
        downsampled = self.output_projection(encoded)

        return downsampled

    def fit(self, train_dataset, epochs):

        self.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08
        )

        self.loss = nn.MSELoss()

        for epoch in range(epochs):

            progress = []

            for step, batch in tqdm(enumerate(train_dataset), total=len(train_dataset)):

                batch = batch.to(self.device)

                batch_size = batch.shape[0]

                # Zero the optimizer
                self.optimizer.zero_grad()

                # Embed the provided text
                embeddings = self.embedding(batch)

                # Sample timesteps randomly
                timesteps = torch.randint(self.diffusion_steps, (batch_size,)).to(self.device)

                # Estimate the denoised parameters
                predictions = self.forward(embeddings, timesteps)

                # Construct the target
                targets = self.q_sample(embeddings, timesteps)

                # Calculate the loss
                loss = self.loss(predictions, targets)

                # Track the loss
                progress.append(float(loss.cpu()))

                # Backpropagate the loss
                loss.backward(retain_graph=step==0)

                # Update the optimizer
                self.optimizer.step()

            print(f"Progress: {np.sum(progress)}")
