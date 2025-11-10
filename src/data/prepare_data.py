"""
Data preparation script for CoRefusion.

This script handles data collection, preprocessing, and splitting.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparator:
    """
    Handles data preparation pipeline.
    
    Args:
        config: Configuration dictionary
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.raw_data_dir = Path(config["paths"]["raw_data"])
        self.processed_data_dir = Path(config["paths"]["processed_data"])
        
    def prepare(self):
        """Run the full data preparation pipeline."""
        logger.info("Starting data preparation pipeline...")
        
        # Step 1: Collect data
        logger.info("Step 1: Collecting data...")
        raw_data = self.collect_data()
        
        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data...")
        processed_data = self.preprocess_data(raw_data)
        
        # Step 3: Split data
        logger.info("Step 3: Splitting data...")
        train_data, val_data, test_data = self.split_data(processed_data)
        
        # Step 4: Save processed data
        logger.info("Step 4: Saving processed data...")
        self.save_data(train_data, val_data, test_data)
        
        logger.info("Data preparation complete!")
        self.print_statistics(train_data, val_data, test_data)
        
    def collect_data(self) -> List[Dict]:
        """
        Collect raw data from sources.
        
        Returns:
            List of raw data samples
        """
        # TODO: Implement data collection
        # This could involve:
        # - Querying GitHub API for repositories
        # - Cloning repositories
        # - Running RefactoringMiner
        # - Extracting code and annotations
        
        logger.info("Data collection not yet implemented")
        return []
        
    def preprocess_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Preprocess raw data.
        
        Args:
            raw_data: List of raw data samples
            
        Returns:
            List of preprocessed data samples
        """
        processed_data = []
        
        for sample in raw_data:
            try:
                processed_sample = self._preprocess_sample(sample)
                processed_data.append(processed_sample)
            except Exception as e:
                logger.warning(f"Failed to preprocess sample: {e}")
                continue
                
        return processed_data
        
    def _preprocess_sample(self, sample: Dict) -> Dict:
        """
        Preprocess a single data sample.
        
        Args:
            sample: Raw data sample
            
        Returns:
            Preprocessed sample
        """
        # TODO: Implement preprocessing steps
        # - Parse code
        # - Extract features
        # - Normalize
        # - Generate labels
        
        return sample
        
    def split_data(
        self,
        data: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            data: List of data samples
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
        
    def save_data(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
    ):
        """
        Save processed data to files.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
        """
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.processed_data_dir / "train.json", "w") as f:
            json.dump(train_data, f, indent=2)
            
        with open(self.processed_data_dir / "val.json", "w") as f:
            json.dump(val_data, f, indent=2)
            
        with open(self.processed_data_dir / "test.json", "w") as f:
            json.dump(test_data, f, indent=2)
            
        logger.info(f"Data saved to {self.processed_data_dir}")
        
    def print_statistics(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
    ):
        """
        Print dataset statistics.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
        """
        logger.info("Dataset Statistics:")
        logger.info(f"  Training samples: {len(train_data)}")
        logger.info(f"  Validation samples: {len(val_data)}")
        logger.info(f"  Test samples: {len(test_data)}")
        logger.info(f"  Total samples: {len(train_data) + len(val_data) + len(test_data)}")


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Prepare data for CoRefusion")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize preparator
    preparator = DataPreparator(config)
    
    # Run preparation
    preparator.prepare()


if __name__ == "__main__":
    main()
