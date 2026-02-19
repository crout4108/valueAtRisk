# Contributing to Value at Risk Calculator

Thank you for your interest in contributing to the Value at Risk (VaR) Calculator! We welcome contributions from the community to help improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How Can I Contribute?

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues or errors in the existing code
- **New features**: Add new VaR calculation methods or risk metrics
- **Documentation**: Improve README, docstrings, or add examples
- **Tests**: Add or improve unit tests and integration tests
- **Performance improvements**: Optimize existing algorithms
- **Code quality**: Refactor code for better maintainability

## Getting Started

### Prerequisites

- Python 3.x
- Git
- Virtual environment (recommended)

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/valueAtRisk.git
   cd valueAtRisk
   ```

3. **Set up upstream remote**
   ```bash
   git remote add upstream https://github.com/crout4108/valueAtRisk.git
   ```

4. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run tests to verify setup**
   ```bash
   python -m unittest discover -s tests -p "test_*.py" -v
   ```

## Development Workflow

### 1. Create a Feature Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - for new features
- `fix/` - for bug fixes
- `docs/` - for documentation updates
- `test/` - for adding tests
- `refactor/` - for code refactoring

### 2. Make Your Changes

- Write clean, readable code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py" -v

# Run specific test file
python -m unittest tests.test_var -v
python -m unittest tests.test_historical_var -v
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: Brief description of what you did"
```

Good commit message examples:
- `Add Monte Carlo VaR calculation method`
- `Fix variance calculation in edge case with single asset`
- `Update documentation for setWeights method`
- `Add unit tests for confidence interval validation`

### 5. Keep Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
```

### 6. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) style guidelines with some specifics:

- **Indentation**: 4 spaces (no tabs)
- **Line length**: Maximum 100 characters (flexible for readability)
- **Naming conventions**:
  - Classes: `PascalCase` (e.g., `ValueAtRisk`)
  - Functions/methods: `camelCase` (existing code) or `snake_case` (preferred for new code)
  - Variables: `camelCase` or `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

### Docstrings

All public classes and methods must have docstrings following [PEP 257](https://peps.python.org/pep-0257/):

```python
def calculate_var(self, confidence_level, window=252):
    """
    Calculate Value at Risk for the portfolio.

    Args:
        confidence_level (float): Confidence level between 0 and 1
        window (int, optional): Time window for calculation. Defaults to 252.

    Returns:
        float: Calculated VaR value

    Raises:
        ValueError: If confidence_level is not between 0 and 1

    Example:
        >>> var_calc = ValueAtRisk(0.95, prices, weights)
        >>> result = var_calc.calculate_var(0.99, window=1)
    """
```

### Code Quality

- **Type hints**: Consider adding type hints for new code
- **Error handling**: Use appropriate exceptions with descriptive messages
- **Comments**: Add comments for complex algorithms or non-obvious logic
- **DRY principle**: Don't repeat yourself - extract common code into functions

## Testing Guidelines

### Writing Tests

- Write tests for all new features and bug fixes
- Place tests in the `tests/` directory
- Name test files with `test_` prefix (e.g., `test_var.py`)
- Name test methods with `test_` prefix (e.g., `test_var_calculation`)

### Test Structure

```python
import unittest
import numpy as np
from VaR import ValueAtRisk

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = np.array([...])
        self.weights = np.array([...])

    def test_basic_functionality(self):
        """Test basic functionality of new feature"""
        var_calc = ValueAtRisk(0.95, self.sample_data, self.weights)
        result = var_calc.new_method()
        self.assertGreater(result, 0)

    def test_edge_case(self):
        """Test edge case handling"""
        # Test code here
```

### Test Coverage

- Aim for high test coverage (> 80%)
- Test both happy paths and edge cases
- Test error conditions and exceptions
- Test boundary values

## Documentation

### What to Document

1. **Code documentation**: Docstrings for all public APIs
2. **README updates**: Update README.md if adding major features
3. **Examples**: Add usage examples for new features
4. **Mathematical background**: Document algorithms and formulas used

### Documentation Best Practices

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include practical examples
- Document assumptions and limitations

## Pull Request Process

### Before Submitting

- [ ] Code follows project coding standards
- [ ] All tests pass
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up-to-date with upstream main

### Submitting a Pull Request

1. **Push your changes** to your fork
2. **Open a Pull Request** on GitHub
3. **Fill out the PR template** with:
   - Description of changes
   - Related issue numbers (if applicable)
   - Testing performed
   - Breaking changes (if any)

### PR Title Format

Use descriptive titles:
- `Add: Monte Carlo VaR implementation`
- `Fix: Variance calculation for single asset portfolios`
- `Docs: Update API documentation for HistoricalVaR`
- `Test: Add integration tests for VaR calculations`

### Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

## Reporting Bugs

### Before Submitting a Bug Report

- Check if the bug has already been reported
- Collect information about the bug:
  - Python version
  - Library versions (`pip list`)
  - Minimal code to reproduce the issue
  - Expected vs actual behavior

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Code snippet that reproduces the issue
2. Input data used
3. Error message received

**Expected behavior**
What you expected to happen.

**Environment**
- Python version:
- pandas version:
- numpy version:
- OS:

**Additional context**
Any other context about the problem.
```

## Suggesting Enhancements

### Enhancement Proposal Template

```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How would you implement this feature?

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Mathematical formulas, references, or examples.
```

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Search closed issues for similar questions
3. Open a new issue with the `question` label

## Attribution

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.

---

Thank you for contributing to Value at Risk Calculator!
