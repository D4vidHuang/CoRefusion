"""
Training script for the CoRefusion model.

This script handles the training loop, checkpointing, and experiment logging.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import yaml

from code_encoder import CodeEncoder
from diffusion_model import DiffusionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoRefusionTrainer:
    """
    Trainer for the CoRefusion model.
    
    Args:
        config: Configuration dictionary
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model components
        self.code_encoder = CodeEncoder(
            model_name=config["model"]["encoder"],
            hidden_dim=config["model"]["hidden_dim"],
        ).to(self.device)
        
        self.diffusion_model = DiffusionModel(
            hidden_dim=config["model"]["hidden_dim"],
            num_timesteps=config["diffusion"]["num_timesteps"],
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            list(self.code_encoder.parameters()) + list(self.diffusion_model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        
        # Initialize scheduler
        self.scheduler = None  # Will be set after dataloader is created
        
        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
        """
        # Initialize scheduler
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = self.config["training"]["warmup_steps"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"Training on {self.device}")
        logger.info(f"Total training steps: {num_training_steps}")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validation
            val_loss = self.validate(val_loader)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Logging
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })
            
            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, "best_model.pt")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config["training"].get("patience", 5):
                logger.info("Early stopping triggered")
                break
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch+1}.pt")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.code_encoder.train()
        self.diffusion_model.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Encode code
            code_embeddings = self.code_encoder(input_ids, attention_mask)
            
            # Compute diffusion loss
            loss = self.diffusion_model.compute_loss(labels, code_embeddings)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.code_encoder.parameters()) + 
                list(self.diffusion_model.parameters()),
                self.config["training"]["gradient_clip"]
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.code_encoder.eval()
        self.diffusion_model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validation"):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Encode code
            code_embeddings = self.code_encoder(input_ids, attention_mask)
            
            # Compute diffusion loss
            loss = self.diffusion_model.compute_loss(labels, code_embeddings)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, filename: str):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            filename: Checkpoint filename
        """
        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "code_encoder_state_dict": self.code_encoder.state_dict(),
            "diffusion_model_state_dict": self.diffusion_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        logger.info(f"Checkpoint saved: {filename}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CoRefusion model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    if config.get("use_wandb", True):
        wandb.init(
            project="corefusion",
            config=config,
            name=config.get("experiment_name", "default"),
        )
    
    # TODO: Load data
    # train_loader, val_loader = load_data(config)
    
    # Initialize trainer
    trainer = CoRefusionTrainer(config)
    
    # Train
    # trainer.train(train_loader, val_loader, config["training"]["num_epochs"])
    
    logger.info("Training complete!")
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
