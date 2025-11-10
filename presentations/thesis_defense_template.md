# Thesis Defense Presentation Template

## Diffusion LLM-based Code Refactoring Localization

**Yongcheng Huang**

MSc Thesis Defense

[Date]

---

## Slide 1: Title Slide

- **Title:** Diffusion LLM-based Code Refactoring Localization
- **Author:** Yongcheng Huang
- **Supervisor:** [Supervisor Name]
- **Date:** [Defense Date]
- **Institution:** [University Name]

---

## Slide 2: Outline

1. Introduction and Motivation
2. Background
3. Problem Statement
4. Proposed Approach
5. Implementation
6. Experiments
7. Results
8. Discussion
9. Conclusions and Future Work

---

## Slide 3: Motivation

**Why Code Refactoring?**

- Improves code quality and maintainability
- Reduces technical debt
- Facilitates future modifications
- Essential for software evolution

**The Challenge:**

- Manual identification is time-consuming
- Requires expert knowledge
- Inconsistent across developers
- Doesn't scale to large codebases

---

## Slide 4: Problem Statement

**Goal:** Automatically identify and localize code segments requiring refactoring

**Challenges:**
- Semantic understanding of code
- Context-dependent decisions
- Multi-language support
- Fine-grained localization

---

## Slide 5: Research Questions

1. **RQ1:** How can diffusion models be applied to code refactoring localization?

2. **RQ2:** What advantages do diffusion models + LLMs offer over traditional approaches?

3. **RQ3:** How does the approach generalize across programming languages?

4. **RQ4:** What types of refactorings can be effectively localized?

---

## Slide 6: Background - Code Refactoring

**Common Refactoring Types:**

- Extract Method
- Rename Variable/Method
- Move Method/Class
- Extract Variable
- Inline Method

**Code Smells:**

- Long methods
- Duplicate code
- Large classes
- Complex conditionals

---

## Slide 7: Background - Large Language Models

**LLMs for Code:**

- Pre-trained on large code corpora
- Understand code semantics
- Examples: CodeBERT, GraphCodeBERT, CodeT5

**Advantages:**
- Rich semantic representations
- Transfer learning
- Multi-language support

---

## Slide 8: Background - Diffusion Models

**Key Concepts:**

- Iterative denoising process
- Probabilistic modeling
- Originally for continuous data
- Adapted for discrete data

**Advantages:**
- Uncertainty quantification
- Gradual refinement
- Flexible conditioning

---

## Slide 9: Proposed Approach - Overview

**CoRefusion Architecture:**

1. **Code Encoder:** Pre-trained LLM + structural features
2. **Diffusion Module:** Discrete diffusion for localization
3. **Localization Decoder:** Fine-grained predictions

**Key Innovation:** Combining semantic understanding of LLMs with probabilistic modeling of diffusion

---

## Slide 10: Proposed Approach - Architecture Diagram

[Include architecture diagram showing:]
- Code input → Encoder → Embeddings
- Embeddings → Diffusion module
- Diffusion → Decoder → Predictions

---

## Slide 11: Implementation - Technology Stack

**Core Components:**
- PyTorch for deep learning
- Transformers (Hugging Face)
- Diffusers library

**Additional Tools:**
- tree-sitter for parsing
- Weights & Biases for tracking
- pytest for testing

---

## Slide 12: Dataset

**Construction:**
- Collected from GitHub repositories
- Identified refactorings with RefactoringMiner
- Multiple programming languages

**Statistics:**
- X,XXX code files
- X languages (Java, Python, JavaScript, C++)
- XX,XXX refactoring annotations

**Split:** 70% train, 15% validation, 15% test

---

## Slide 13: Experimental Setup

**Baseline Methods:**

**Traditional:**
- PMD, SonarQube
- Metric-based detection

**ML-based:**
- Random Forest
- LSTM, CNN
- Graph Neural Networks

**LLM-based:**
- Fine-tuned CodeBERT
- GraphCodeBERT
- GPT-3.5 with prompting

---

## Slide 14: Evaluation Metrics

**Primary Metrics:**
- Precision
- Recall
- F1-Score
- Accuracy

**Additional Metrics:**
- Mean Average Precision (MAP)
- Line-level accuracy
- Per-refactoring-type performance

---

## Slide 15: Results - Overall Performance

[Include table showing:]

| Method | Precision | Recall | F1 | Accuracy |
|--------|-----------|--------|----|---------| 
| PMD | 0.45 | 0.38 | 0.41 | 0.72 |
| CodeBERT | 0.74 | 0.70 | 0.72 | 0.85 |
| **CoRefusion** | **0.82** | **0.79** | **0.80** | **0.89** |

**Key Finding:** 6-8% improvement over best baseline

---

## Slide 16: Results - Ablation Study

[Include chart showing contribution of components:]

- Full Model: F1 = 0.80
- w/o Diffusion: F1 = 0.74 (-6%)
- w/o Pre-training: F1 = 0.69 (-11%)
- w/o Structural Features: F1 = 0.76 (-4%)

**Insight:** All components contribute significantly

---

## Slide 17: Results - Cross-Language Performance

[Include chart showing per-language results:]

- Java: F1 = 0.82
- Python: F1 = 0.80
- JavaScript: F1 = 0.78
- C++: F1 = 0.76

**Finding:** Consistent performance across languages

---

## Slide 18: Results - Qualitative Analysis

**Success Cases:**
- Complex Extract Method scenarios
- Context-dependent patterns
- Multi-line refactorings

**Failure Cases:**
- Domain-specific code
- Rare refactoring types
- Very large files (>1000 lines)

---

## Slide 19: Discussion - Why Diffusion Helps

**Key Advantages:**

1. **Uncertainty Modeling:** Natural handling of ambiguous cases
2. **Gradual Refinement:** Iterative improvement of predictions
3. **Context Integration:** Effective conditioning on code embeddings
4. **Distribution Learning:** Captures patterns in refactoring locations

---

## Slide 20: Discussion - Practical Implications

**Applications:**

- IDE integration for real-time suggestions
- Code review automation
- CI/CD pipeline integration
- Educational tools

**Impact:**

- Reduces developer time on manual review
- Improves code quality consistency
- Facilitates continuous refactoring

---

## Slide 21: Limitations

**Current Limitations:**

1. Computational cost higher than simpler methods
2. Performance degrades on rare refactoring types
3. Limited interpretability
4. Requires substantial training data

**Acknowledged but Outside Scope:**
- Actual refactoring execution
- Real-world deployment study
- Longitudinal impact analysis

---

## Slide 22: Contributions

**Scientific Contributions:**

1. Novel application of diffusion models to code refactoring
2. Comprehensive evaluation across languages and refactoring types
3. Insights into model behavior and limitations

**Practical Contributions:**

1. Open-source implementation
2. Curated dataset
3. Comprehensive baseline comparisons

---

## Slide 23: Future Work

**Short-term:**

- Improve computational efficiency
- Enhanced interpretability
- Extended language support

**Long-term:**

- Automatic refactoring execution
- Interactive systems with developer feedback
- Integration with formal methods
- Broader code analysis tasks

---

## Slide 24: Conclusions

**Summary:**

- Diffusion models + LLMs offer promising approach to code refactoring localization
- Achieves state-of-the-art performance across multiple languages
- Demonstrates potential of generative models for SE tasks

**Key Takeaway:**

Combining semantic understanding with probabilistic modeling enables more effective code quality maintenance

---

## Slide 25: Thank You

**Questions?**

**Contact:**
- Email: [your.email@university.edu]
- GitHub: github.com/D4vidHuang/CoRefusion

**Resources:**
- Code: [GitHub link]
- Dataset: [Data repository link]
- Thesis: [University repository link]

---

## Backup Slides

### Additional Results

[Include extra results, detailed tables, additional visualizations]

### Technical Details

[Include implementation details, hyperparameters, training curves]

### Related Work

[Include more detailed comparison with related work]

---

## Presentation Tips

1. **Time Management:** Aim for 20-30 minutes, leaving time for questions
2. **Visuals:** Use diagrams, charts, and examples over text
3. **Storytelling:** Follow a clear narrative from problem to solution
4. **Practice:** Rehearse multiple times, anticipate questions
5. **Clarity:** Explain technical concepts clearly for non-expert audience
6. **Confidence:** Know your work well, be honest about limitations
7. **Engagement:** Make eye contact, use natural gestures
8. **Backup:** Prepare extra slides for potential questions

## Common Questions to Prepare For

1. Why diffusion models specifically?
2. How does this compare to [specific baseline]?
3. What about real-world deployment?
4. How would you handle [edge case]?
5. What's the computational cost?
6. Can this generalize to other SE tasks?
7. What was the biggest challenge?
8. What would you do differently?
