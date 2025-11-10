# Contributing to CoRefusion

Thank you for your interest in contributing to the CoRefusion project!

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We follow standard Python coding conventions:

- **PEP 8** for Python code style
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **Type hints** for function signatures
- **Docstrings** for all public functions and classes (Google style)

### Running Code Formatters

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

## Testing

All new code should include tests:

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the structure of `src/`
- Use descriptive test names
- Include docstrings explaining what is being tested

Example:
```python
def test_code_encoder_output_shape():
    """Test that code encoder produces expected output shape."""
    encoder = CodeEncoder()
    input_ids = torch.randint(0, 1000, (2, 512))
    attention_mask = torch.ones(2, 512)
    
    output = encoder(input_ids, attention_mask)
    
    assert output.shape == (2, 512, 768)
```

## Documentation

- Update documentation when adding new features
- Include docstrings with:
  - Brief description
  - Args (with types)
  - Returns (with types)
  - Raises (if applicable)
  - Examples (for complex functions)

Example:
```python
def train_model(config: Dict, data_loader: DataLoader) -> Dict[str, float]:
    """
    Train the model with given configuration.
    
    Args:
        config: Training configuration dictionary
        data_loader: DataLoader for training data
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If configuration is invalid
        
    Example:
        >>> config = {'lr': 1e-4, 'epochs': 10}
        >>> metrics = train_model(config, train_loader)
    """
    pass
```

## Git Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, descriptive commits:
   ```bash
   git commit -m "Add feature X that does Y"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots (if UI changes)
   - Test results

## Commit Message Guidelines

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(encoder): add hierarchical code encoder

Implement a hierarchical encoder that processes code at multiple
levels (token, line, function, file).

Closes #123
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG.md (if applicable)
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

## Questions or Issues?

- Open an issue for bugs or feature requests
- Use discussions for questions
- Email the maintainer for sensitive matters

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Maintain professional communication

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
