# Getting Started with CoRefusion

This guide will help you get started with the CoRefusion project for your thesis research.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended for training)
- At least 16GB RAM

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/D4vidHuang/CoRefusion.git
cd CoRefusion
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
# Run tests to verify installation
pytest tests/ -v
```

## Quick Start

### Data Preparation

1. **Download or prepare your dataset:**

```bash
# Create necessary directories
mkdir -p data/raw data/processed

# TODO: Add scripts for data collection
# python src/data/collect_data.py --config experiments/configs/data_config.yaml
```

2. **Preprocess the data:**

```bash
# Preprocess collected data
python src/data/prepare_data.py --config experiments/configs/data_config.yaml
```

### Training

Train the model using the provided configuration:

```bash
# Train with default configuration
python src/models/train.py --config experiments/configs/train_config.yaml

# Train with custom seed
python src/models/train.py --config experiments/configs/train_config.yaml --seed 42
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir experiments/logs
```

Or use Weights & Biases (recommended):

```bash
# Set up wandb
wandb login
# Training will automatically log to wandb
```

### Evaluation

Evaluate a trained model:

```bash
python src/evaluation/evaluate.py \
    --checkpoint experiments/checkpoints/best_model.pt \
    --config experiments/configs/eval_config.yaml
```

### Inference

Run inference on new code:

```bash
python src/models/infer.py \
    --checkpoint experiments/checkpoints/best_model.pt \
    --input path/to/code.py \
    --output predictions.json
```

## Project Structure

Here's a quick overview of the most important files and directories:

```
CoRefusion/
├── src/                      # Source code
│   ├── models/              # Model implementations
│   │   ├── diffusion_model.py
│   │   ├── code_encoder.py
│   │   └── train.py
│   ├── data/                # Data processing
│   ├── evaluation/          # Evaluation scripts
│   └── utils/               # Utility functions
├── experiments/              # Experiment configurations
│   └── configs/             # Config files
├── thesis/                   # Thesis document
│   └── chapters/            # Thesis chapters
├── notebooks/                # Jupyter notebooks
├── data/                     # Dataset storage
└── results/                  # Experimental results
```

## Development Workflow

### Running Experiments

1. **Create a configuration file** in `experiments/configs/`
2. **Run the experiment:**
   ```bash
   python src/models/train.py --config experiments/configs/your_config.yaml
   ```
3. **Monitor results** in `experiments/logs/` or on wandb
4. **Analyze results** using notebooks in `notebooks/analysis/`

### Adding New Features

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Format code:
   ```bash
   black src/
   isort src/
   ```

5. Commit and push:
   ```bash
   git add .
   git commit -m "Add your feature"
   git push origin feature/your-feature-name
   ```

### Working on the Thesis

The thesis is in LaTeX format in the `thesis/` directory.

To compile the thesis:

```bash
cd thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX editor (e.g., Overleaf, TeXstudio).

## Useful Commands

### Data Management

```bash
# Check dataset statistics
python src/data/analyze_dataset.py --data_dir data/processed

# Visualize data distribution
python src/visualization/plot_data_stats.py --data_dir data/processed
```

### Model Management

```bash
# List available checkpoints
ls -lh experiments/checkpoints/

# Resume training from checkpoint
python src/models/train.py \
    --config experiments/configs/train_config.yaml \
    --resume experiments/checkpoints/checkpoint_epoch_10.pt
```

### Code Quality

```bash
# Format code
black src/ --line-length 100

# Sort imports
isort src/

# Lint code
flake8 src/
pylint src/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **Out of memory errors:**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Slow training:**
   - Check GPU utilization: `nvidia-smi`
   - Increase num_workers for data loading
   - Use distributed training for multiple GPUs

3. **Import errors:**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

4. **CUDA errors:**
   - Check CUDA version compatibility: `nvcc --version`
   - Reinstall PyTorch with correct CUDA version

### Getting Help

- Check the [documentation](docs/)
- Open an issue on GitHub
- Contact the author: [your.email@example.com]

## Next Steps

1. **Explore the notebooks:** Start with `notebooks/exploratory/01_data_exploration.ipynb`
2. **Run baseline experiments:** Use configurations in `experiments/configs/`
3. **Read the thesis chapters:** Understand the methodology in `thesis/chapters/`
4. **Customize for your research:** Adapt the code and configurations for your specific needs

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
