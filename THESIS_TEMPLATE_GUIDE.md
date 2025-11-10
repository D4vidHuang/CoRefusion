# CoRefusion Thesis Template Guide

Welcome to your MSc thesis template for "Diffusion LLM-based Code Refactoring Localization"!

This repository contains everything you need to successfully complete your thesis research. This guide provides an overview of what has been created and how to use it.

## 📚 What's Included

### 1. Complete Thesis Document Structure (LaTeX)

**Location:** `thesis/`

A fully structured thesis with 9 chapters:

- **Chapter 1: Introduction** - Motivation, problem statement, research questions, and contributions
- **Chapter 2: Background** - Foundational concepts (refactoring, LLMs, diffusion models)
- **Chapter 3: Literature Review** - Comprehensive review of related work
- **Chapter 4: Methodology** - Detailed description of your proposed approach
- **Chapter 5: Implementation** - Technical implementation details
- **Chapter 6: Experiments** - Experimental setup and configurations
- **Chapter 7: Results** - Experimental findings
- **Chapter 8: Discussion** - Interpretation, implications, and limitations
- **Chapter 9: Conclusion** - Summary and future work

Plus appendices with supplementary materials and reproducibility information.

**To compile:**
```bash
cd thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 2. Complete Source Code Implementation

**Location:** `src/`

Ready-to-use implementation templates:

- **`models/diffusion_model.py`** - Full diffusion model architecture
- **`models/code_encoder.py`** - Code encoder with hierarchical encoding
- **`models/train.py`** - Complete training loop
- **`data/prepare_data.py`** - Data preparation pipeline
- **`evaluation/evaluate.py`** - Evaluation framework

All code includes:
- Comprehensive docstrings
- Type hints
- Error handling
- Logging

### 3. Experiment Configuration

**Location:** `experiments/configs/`

Pre-configured YAML files for:
- Model hyperparameters
- Training settings
- Optimization parameters
- Data paths

Easy to modify and extend for your experiments.

### 4. Comprehensive Documentation

**Location:** `docs/`

- **GETTING_STARTED.md** - Setup and quick start guide
- **RESEARCH_PLAN.md** - Detailed 28-week timeline with milestones
- **CONTRIBUTING.md** - Development guidelines
- **methodology/DETAILED_METHODOLOGY.md** - In-depth technical approach

### 5. Jupyter Notebooks

**Location:** `notebooks/`

- **exploratory/01_data_exploration.ipynb** - Data analysis template
- Ready structure for analysis notebooks

### 6. Presentation Templates

**Location:** `presentations/`

- **thesis_defense_template.md** - Complete defense presentation with 25+ slides
- Includes backup slides and presentation tips

### 7. Project Infrastructure

- **requirements.txt** - Core dependencies
- **requirements-dev.txt** - Development tools
- **setup.py** - Package installation
- **.gitignore** - Proper version control
- **README.md** - Project overview

## 🚀 Getting Started

### Step 1: Set Up Your Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Optional dev tools
```

### Step 2: Start with the Research Plan

Read `docs/RESEARCH_PLAN.md` to understand the complete research timeline:

- Phase 1: Literature Review (Weeks 1-4)
- Phase 2: Data Collection (Weeks 5-7)
- Phase 3: Methodology Design (Weeks 8-10)
- Phase 4: Implementation (Weeks 11-14)
- Phase 5: Experimentation (Weeks 15-18)
- Phase 6: Analysis (Weeks 19-21)
- Phase 7: Writing (Weeks 22-26)
- Phase 8: Defense Preparation (Weeks 27-28)

### Step 3: Begin Literature Review

1. Start reading the papers referenced in `thesis/references.bib`
2. Add your own references as you find them
3. Take notes in `docs/literature/`
4. Begin drafting Chapter 2 (Background) and Chapter 3 (Literature Review)

### Step 4: Start Data Collection

1. Review `src/data/prepare_data.py`
2. Modify for your specific data sources
3. Run data collection and preprocessing
4. Use `notebooks/exploratory/01_data_exploration.ipynb` to analyze your data

### Step 5: Implement Your Approach

1. Review the code templates in `src/models/`
2. Customize the architecture for your specific approach
3. Configure experiments in `experiments/configs/`
4. Start training with `python src/models/train.py --config experiments/configs/train_config.yaml`

### Step 6: Run Experiments

1. Execute your experimental plan from Chapter 6
2. Log results with Weights & Biases
3. Analyze results in notebooks
4. Generate figures for the thesis

### Step 7: Write the Thesis

1. Fill in the TODOs in each chapter
2. Add your figures to `thesis/figures/`
3. Update `thesis/references.bib` with your citations
4. Compile regularly to check for errors

### Step 8: Prepare Defense

1. Use `presentations/thesis_defense_template.md` as a starting point
2. Create slides with your actual results
3. Practice your presentation multiple times
4. Prepare for common questions

## 📁 Directory Structure Explained

```
CoRefusion/
├── thesis/                    # Your thesis document
│   ├── chapters/             # Individual chapters
│   ├── figures/              # Thesis figures
│   ├── main.tex              # Main LaTeX file
│   └── references.bib        # Bibliography
│
├── src/                       # Source code
│   ├── models/               # Model implementations
│   ├── data/                 # Data processing
│   ├── evaluation/           # Evaluation scripts
│   ├── utils/                # Utility functions
│   └── visualization/        # Visualization tools
│
├── experiments/               # Experiment management
│   ├── configs/              # Configuration files
│   ├── logs/                 # Training logs
│   └── checkpoints/          # Model checkpoints
│
├── data/                      # Dataset storage
│   ├── raw/                  # Original data
│   ├── processed/            # Processed data
│   ├── external/             # External datasets
│   └── interim/              # Intermediate data
│
├── results/                   # Experimental results
│   ├── figures/              # Generated plots
│   ├── tables/               # Result tables
│   └── analysis/             # Analysis outputs
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory/          # Data exploration
│   └── analysis/             # Results analysis
│
├── docs/                      # Documentation
│   ├── literature/           # Literature notes
│   ├── methodology/          # Methodology details
│   └── notes/                # Research notes
│
├── presentations/             # Presentations
└── tests/                     # Unit tests
```

## 🎯 Key Milestones

Track your progress with these milestones:

- [ ] **Month 1:** Literature review complete, Chapter 2 & 3 drafted
- [ ] **Month 2:** Dataset collected and preprocessed
- [ ] **Month 3:** Methodology designed, Chapter 4 drafted
- [ ] **Month 4:** Implementation complete, Chapter 5 drafted
- [ ] **Month 5:** All experiments complete, Chapter 6 & 7 drafted
- [ ] **Month 6:** All chapters complete, full thesis draft
- [ ] **Month 7:** Final revision, defense preparation, submission

## 💡 Tips for Success

### Research Tips

1. **Keep detailed notes** - Document everything in `docs/notes/`
2. **Regular backups** - Commit to Git frequently
3. **Reproducibility** - Use configuration files and set random seeds
4. **Track experiments** - Use Weights & Biases or similar
5. **Stay organized** - Follow the directory structure

### Writing Tips

1. **Write early and often** - Don't wait until the end
2. **Get feedback regularly** - Share drafts with your supervisor
3. **Be honest about limitations** - Acknowledge what doesn't work
4. **Support claims with evidence** - Always cite sources
5. **Proofread carefully** - Use tools like Grammarly

### Time Management Tips

1. **Follow the timeline** - Use the research plan
2. **Set weekly goals** - Break down large tasks
3. **Buffer time** - Things always take longer than expected
4. **Prioritize** - Focus on critical experiments first
5. **Take breaks** - Avoid burnout

## 🔧 Customization

This template is designed to be customized:

### Modifying the Architecture

Edit `src/models/diffusion_model.py` and `src/models/code_encoder.py` to implement your specific approach.

### Adding New Experiments

1. Create a new config file in `experiments/configs/`
2. Run with: `python src/models/train.py --config your_config.yaml`
3. Document in Chapter 6

### Changing the Thesis Structure

Feel free to reorganize chapters if your university has different requirements. The content is comprehensive but adaptable.

### Adding Languages

The template uses English, but you can add translations or use your preferred language. Check your university requirements.

## 📖 Additional Resources

### Learning Materials

- **PyTorch Tutorial:** https://pytorch.org/tutorials/
- **Transformers Documentation:** https://huggingface.co/docs/transformers/
- **Diffusion Models Explained:** Look for papers by Ho et al. and Song et al.
- **LaTeX Guide:** https://www.overleaf.com/learn

### Tools

- **Overleaf:** Online LaTeX editor (alternative to local compilation)
- **Mendeley/Zotero:** Reference management
- **Grammarly:** Writing assistance
- **Draw.io:** Create diagrams

### Datasets

- **GitHub:** Source code repositories
- **RefactoringMiner:** Refactoring detection tool
- **CodeSearchNet:** Code search dataset
- **Defects4J:** Bug dataset (for comparison)

## 🆘 Getting Help

If you encounter issues:

1. **Check the documentation** in `docs/`
2. **Review example notebooks** in `notebooks/`
3. **Consult your supervisor** - Regular meetings are crucial
4. **Search online** - Stack Overflow, GitHub issues
5. **Ask your research group** - Colleagues can help

## 📝 Next Steps

Your immediate next steps:

1. ✅ Read this guide completely
2. ✅ Set up your development environment
3. ✅ Read `docs/RESEARCH_PLAN.md` in detail
4. ✅ Start your literature review
5. ✅ Schedule regular meetings with your supervisor
6. ✅ Begin filling in the thesis chapters as you progress
7. ✅ Start collecting data
8. ✅ Familiarize yourself with the codebase

## 🎓 Final Notes

This template provides a comprehensive starting point, but remember:

- **Your research is unique** - Adapt as needed
- **Quality over quantity** - Focus on clear contributions
- **Learning is the goal** - The process matters more than perfection
- **Ask for help** - Research is collaborative
- **Enjoy the journey** - This is an exciting opportunity!

## 📞 Contact

For questions about this template:
- GitHub: @D4vidHuang
- Email: [your.email@example.com]

For thesis-specific questions:
- Consult your supervisor
- Refer to your university guidelines

---

**Good luck with your thesis! 🚀**

You have all the tools you need for success. Now it's time to make it happen!
