# Thesis Document

This directory contains the LaTeX source files for the MSc thesis.

## Structure

- `main.tex` - Main thesis document
- `chapters/` - Individual chapters
- `figures/` - Thesis figures and diagrams
- `references.bib` - Bibliography database
- `templates/` - LaTeX templates and style files

## Building the Thesis

### Using LaTeX

```bash
# Compile the thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using Overleaf

Upload all files in this directory to an Overleaf project.

## Chapter Organization

1. **Introduction** - Research motivation, objectives, and contributions
2. **Background** - Foundational concepts and related work
3. **Literature Review** - Comprehensive review of relevant research
4. **Methodology** - Detailed description of the proposed approach
5. **Implementation** - System design and technical details
6. **Experiments** - Experimental setup and configurations
7. **Results** - Findings and analysis
8. **Discussion** - Interpretation of results and implications
9. **Conclusion** - Summary, contributions, and future work

## Writing Guidelines

- Keep sentences clear and concise
- Use active voice when appropriate
- Define technical terms on first use
- Maintain consistent terminology throughout
- Include sufficient detail for reproducibility
- Support claims with evidence and citations
