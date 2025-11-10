"""
Diffusion model for code refactoring localization.

This module implements the discrete diffusion model for predicting
refactoring locations in source code.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class DiffusionModel(nn.Module):
    """
    Diffusion model for code refactoring localization.
    
    This model uses a discrete diffusion process to iteratively denoise
    predictions of refactoring locations.
    
    Args:
        hidden_dim: Dimension of hidden representations
        num_timesteps: Number of diffusion timesteps
        num_classes: Number of classes for discrete diffusion (typically 2 for binary)
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_timesteps: int = 1000,
        num_classes: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        
        # Denoising network
        self.denoising_network = DenoisingNetwork(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
        
        # Timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Transition matrices for discrete diffusion
        self.register_buffer("betas", self._get_beta_schedule())
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        
    def _get_beta_schedule(self) -> torch.Tensor:
        """
        Generate beta schedule for diffusion process.
        
        Returns:
            Tensor of beta values for each timestep
        """
        # Linear schedule
        return torch.linspace(0.0001, 0.02, self.num_timesteps)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        
        Args:
            x_t: Noisy input at timestep t [batch, seq_len, num_classes]
            t: Timestep [batch]
            condition: Conditioning information (code embeddings) [batch, seq_len, hidden_dim]
            
        Returns:
            Predicted distribution over clean labels [batch, seq_len, num_classes]
        """
        # Embed timestep
        t_emb = self.time_embedding(t.unsqueeze(-1))  # [batch, hidden_dim]
        
        # Denoise
        output = self.denoising_network(x_t, t_emb, condition)
        
        return output
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            x_0: Clean labels [batch, seq_len]
            condition: Code embeddings [batch, seq_len, hidden_dim]
            
        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]
        
        # Sample timestep
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)
        
        # Add noise to get x_t
        x_t = self._add_noise(x_0, t)
        
        # Predict x_0 from x_t
        pred = self.forward(x_t, t, condition)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            pred.reshape(-1, self.num_classes),
            x_0.reshape(-1),
        )
        
        return loss
    
    def _add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Add noise to clean labels according to diffusion schedule.
        
        Args:
            x_0: Clean labels [batch, seq_len]
            t: Timesteps [batch]
            
        Returns:
            Noisy labels [batch, seq_len, num_classes]
        """
        # Convert to one-hot
        x_0_onehot = nn.functional.one_hot(x_0, self.num_classes).float()
        
        # Get alpha for timestep
        alpha_t = self.alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        
        # Sample noise
        noise = torch.randn_like(x_0_onehot)
        
        # Add noise: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        x_t = torch.sqrt(alpha_t) * x_0_onehot + torch.sqrt(1 - alpha_t) * noise
        
        return x_t
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample predictions using the diffusion process.
        
        Args:
            condition: Code embeddings [batch, seq_len, hidden_dim]
            num_steps: Number of sampling steps (defaults to num_timesteps)
            
        Returns:
            Predicted labels [batch, seq_len]
        """
        if num_steps is None:
            num_steps = self.num_timesteps
            
        batch_size, seq_len, _ = condition.shape
        
        # Start from random noise
        x_t = torch.randn(batch_size, seq_len, self.num_classes, device=condition.device)
        
        # Iteratively denoise
        for t in reversed(range(num_steps)):
            t_batch = torch.full((batch_size,), t, device=condition.device)
            
            # Predict
            pred = self.forward(x_t, t_batch, condition)
            
            # Update x_t
            if t > 0:
                # Add noise for next step
                noise = torch.randn_like(x_t)
                x_t = pred + torch.sqrt(self.betas[t]) * noise
            else:
                # Final step: no noise
                x_t = pred
        
        # Convert to class predictions
        predictions = torch.argmax(x_t, dim=-1)
        
        return predictions


class DenoisingNetwork(nn.Module):
    """
    Neural network for denoising in diffusion process.
    
    Args:
        hidden_dim: Dimension of hidden representations
        num_classes: Number of output classes
    """
    
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        
        self.input_proj = nn.Linear(num_classes, hidden_dim)
        
        # Transformer layers for denoising
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        
    def forward(
        self,
        x_t: torch.Tensor,
        t_emb: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoise input.
        
        Args:
            x_t: Noisy input [batch, seq_len, num_classes]
            t_emb: Time embedding [batch, hidden_dim]
            condition: Conditioning [batch, seq_len, hidden_dim]
            
        Returns:
            Denoised output [batch, seq_len, num_classes]
        """
        # Project input
        x = self.input_proj(x_t)  # [batch, seq_len, hidden_dim]
        
        # Add time embedding
        x = x + t_emb.unsqueeze(1)
        
        # Add conditioning
        x = x + condition
        
        # Transform
        x = self.transformer(x)
        
        # Project to output
        output = self.output_proj(x)
        
        return output
