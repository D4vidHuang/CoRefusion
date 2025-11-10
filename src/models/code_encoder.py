"""
Code encoder module for extracting representations from source code.

This module provides encoders that transform source code into rich
representations suitable for the diffusion model.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple


class CodeEncoder(nn.Module):
    """
    Encoder for source code using pre-trained language models.
    
    Args:
        model_name: Name of pre-trained model (e.g., "microsoft/codebert-base")
        hidden_dim: Output dimension
        freeze_encoder: Whether to freeze the pre-trained encoder
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        hidden_dim: int = 768,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        # Load pre-trained model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        encoder_dim = self.encoder.config.hidden_size
        
        # Projection layer
        self.projection = nn.Linear(encoder_dim, hidden_dim)
        
        # Additional feature encoders
        self.complexity_encoder = nn.Sequential(
            nn.Linear(5, 64),  # 5 complexity metrics
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        complexity_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode source code.
        
        Args:
            input_ids: Tokenized code [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            complexity_features: Optional complexity metrics [batch, seq_len, 5]
            
        Returns:
            Code embeddings [batch, seq_len, hidden_dim]
        """
        # Encode with pre-trained model
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get token-level representations
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, encoder_dim]
        
        # Project to target dimension
        embeddings = self.projection(hidden_states)
        
        # Add complexity features if provided
        if complexity_features is not None:
            complexity_emb = self.complexity_encoder(complexity_features)
            embeddings = embeddings + complexity_emb
        
        return embeddings
    
    def encode_code(
        self,
        code: str,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode a code string.
        
        Args:
            code: Source code string
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoding = self.tokenizer(
            code,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return encoding


class HierarchicalCodeEncoder(nn.Module):
    """
    Hierarchical encoder that processes code at multiple levels.
    
    Encodes code at:
    - Token level
    - Line level  
    - Function level
    - File level
    
    Args:
        model_name: Name of pre-trained model
        hidden_dim: Hidden dimension
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        hidden_dim: int = 768,
    ):
        super().__init__()
        
        self.token_encoder = CodeEncoder(
            model_name=model_name,
            hidden_dim=hidden_dim,
        )
        
        # Line-level encoder
        self.line_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        
        # Function-level attention
        self.function_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        
        # File-level aggregation
        self.file_pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        line_boundaries: Optional[torch.Tensor] = None,
        function_boundaries: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode code hierarchically.
        
        Args:
            input_ids: Tokenized code [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            line_boundaries: Line boundary indices [batch, num_lines, 2]
            function_boundaries: Function boundary indices [batch, num_functions, 2]
            
        Returns:
            Dictionary with embeddings at different levels
        """
        # Token-level encoding
        token_embeddings = self.token_encoder(input_ids, attention_mask)
        
        # Line-level encoding
        if line_boundaries is not None:
            line_embeddings = self._encode_lines(token_embeddings, line_boundaries)
        else:
            line_embeddings, _ = self.line_encoder(token_embeddings)
        
        # Function-level encoding
        if function_boundaries is not None:
            function_embeddings = self._encode_functions(
                line_embeddings, function_boundaries
            )
        else:
            function_embeddings, _ = self.function_attention(
                line_embeddings, line_embeddings, line_embeddings
            )
        
        # File-level encoding
        file_embedding = self.file_pooling(
            function_embeddings.transpose(1, 2)
        ).squeeze(-1)
        
        return {
            "token": token_embeddings,
            "line": line_embeddings,
            "function": function_embeddings,
            "file": file_embedding,
        }
    
    def _encode_lines(
        self,
        token_embeddings: torch.Tensor,
        line_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate token embeddings into line embeddings.
        
        Args:
            token_embeddings: Token-level embeddings [batch, seq_len, hidden_dim]
            line_boundaries: Line boundaries [batch, num_lines, 2]
            
        Returns:
            Line embeddings [batch, num_lines, hidden_dim]
        """
        batch_size, num_lines, _ = line_boundaries.shape
        hidden_dim = token_embeddings.shape[-1]
        
        line_embeddings = []
        for i in range(batch_size):
            lines = []
            for j in range(num_lines):
                start, end = line_boundaries[i, j]
                if start < end:
                    line_tokens = token_embeddings[i, start:end]
                    line_emb = line_tokens.mean(dim=0)
                else:
                    line_emb = torch.zeros(hidden_dim, device=token_embeddings.device)
                lines.append(line_emb)
            line_embeddings.append(torch.stack(lines))
        
        return torch.stack(line_embeddings)
    
    def _encode_functions(
        self,
        line_embeddings: torch.Tensor,
        function_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate line embeddings into function embeddings.
        
        Args:
            line_embeddings: Line-level embeddings [batch, num_lines, hidden_dim]
            function_boundaries: Function boundaries [batch, num_functions, 2]
            
        Returns:
            Function embeddings [batch, num_functions, hidden_dim]
        """
        # Similar to _encode_lines but for functions
        batch_size, num_functions, _ = function_boundaries.shape
        hidden_dim = line_embeddings.shape[-1]
        
        function_embeddings = []
        for i in range(batch_size):
            functions = []
            for j in range(num_functions):
                start, end = function_boundaries[i, j]
                if start < end:
                    func_lines = line_embeddings[i, start:end]
                    func_emb = func_lines.mean(dim=0)
                else:
                    func_emb = torch.zeros(hidden_dim, device=line_embeddings.device)
                functions.append(func_emb)
            function_embeddings.append(torch.stack(functions))
        
        return torch.stack(function_embeddings)
