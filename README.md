# CoRefusion: Diffusion LLM-based Code Refactoring Localization

This is the repository for YongchengHuang's MSc thesis on Diffusion LLM-based Code Refactoring Localization.

## Overview

This research explores the application of diffusion models combined with Large Language Models (LLMs) to automatically identify and localize code segments that require refactoring. The goal is to improve code quality and maintainability by leveraging state-of-the-art generative AI techniques.

## Research Questions

1. How can diffusion models be effectively applied to code refactoring localization tasks?
2. What are the advantages of combining diffusion models with LLMs for code analysis compared to traditional approaches?
3. How does the proposed approach perform across different programming languages and codebases?

## Repository Structure

```
├── thesis/                 # Thesis document and LaTeX files
├── src/                   # Source code for the research
│   ├── models/           # Model architectures and implementations
│   ├── data/             # Data loading and preprocessing
│   ├── evaluation/       # Evaluation metrics and scripts
│   ├── utils/            # Utility functions
│   └── visualization/    # Visualization tools
├── data/                  # Dataset storage
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned and processed data
│   ├── external/         # External datasets
│   └── interim/          # Intermediate data transformations
├── experiments/           # Experiment configurations and outputs
│   ├── configs/          # Configuration files for experiments
│   ├── logs/             # Training and experiment logs
│   └── checkpoints/      # Model checkpoints
├── results/               # Results and analysis
│   ├── figures/          # Generated figures
│   ├── tables/           # Result tables
│   └── analysis/         # Analysis scripts and outputs
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/      # Exploratory data analysis
│   └── analysis/         # Results analysis notebooks
├── docs/                  # Documentation
│   ├── literature/       # Literature review and summaries
│   ├── methodology/      # Detailed methodology documentation
│   └── notes/            # Research notes
├── presentations/         # Presentation slides
└── tests/                # Unit and integration tests
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Git

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/D4vidHuang/CoRefusion.git
cd CoRefusion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

## Quick Start

### Data Preparation

```bash
# Download and prepare datasets
python src/data/prepare_data.py --config experiments/configs/data_config.yaml
```

### Training

```bash
# Train the diffusion model
python src/models/train.py --config experiments/configs/train_config.yaml
```

### Evaluation

```bash
# Evaluate the trained model
python src/evaluation/evaluate.py --checkpoint experiments/checkpoints/best_model.pt
```

## Experiments

All experiment configurations are stored in `experiments/configs/`. To reproduce experiments:

```bash
# Run specific experiment
python src/models/train.py --config experiments/configs/experiment_1.yaml
```

## Results

Results are automatically saved to the `results/` directory. Key findings and visualizations can be found in:
- `results/figures/` - Generated plots and visualizations
- `results/tables/` - Performance metrics and comparison tables
- `notebooks/analysis/` - Detailed analysis notebooks

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{huang2024corefusion,
  title={Diffusion LLM-based Code Refactoring Localization},
  author={Huang, Yongcheng},
  year={2024},
  school={[Your University]},
  type={MSc thesis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Author: Yongcheng Huang (D4vidHuang)
- Email: [Your Email]
- GitHub: [@D4vidHuang](https://github.com/D4vidHuang)

## Acknowledgments

- Thesis Supervisor: [Supervisor Name]
- Institution: [University Name]
- Research Group: [Group Name]
