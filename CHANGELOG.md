# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive docstrings for all classes and methods following PEP 257
- Module-level documentation with mathematical foundations
- CONTRIBUTING.md with detailed contribution guidelines
- CODE_OF_CONDUCT.md for community standards
- This CHANGELOG.md to track version history

### Changed
- Enhanced inline code comments for better code readability
- Improved documentation of mathematical formulas and algorithms

## [1.0.0] - 2020-09-02

### Added
- Initial release of Value at Risk Calculator
- Parametric VaR calculation using variance-covariance method
- Historical VaR calculation using historical simulation method
- Support for multi-asset portfolios
- Flexible time window configuration (daily, annualized, custom)
- Portfolio variance calculation with two approaches:
  - Covariance matrix method (exact)
  - Portfolio return approximation
- Integration with Yahoo Finance via pandas_datareader
- Configurable confidence intervals
- Results available in both percentage and dollar amounts
- Comprehensive test suite with 34 unit tests
- README with usage examples and API documentation
- MIT License

### Features
- `ValueAtRisk` class for parametric VaR
  - `covMatrix()` - Calculate covariance matrix
  - `calculateVariance()` - Calculate portfolio variance
  - `var()` - Calculate parametric VaR
  - `setCI()` - Update confidence interval
  - `setPortfolio()` - Update portfolio data
  - `setWeights()` - Update portfolio weights

- `HistoricalVaR` class for historical VaR
  - Inherits from `ValueAtRisk`
  - `var()` - Calculate historical VaR using percentiles
  - Support for rolling window calculations

### Technical Details
- Python 3.x support
- Dependencies: pandas, numpy, scipy, pandas_datareader, matplotlib
- Log returns calculation for better statistical properties
- Normal distribution assumption for parametric VaR
- Percentile-based approach for historical VaR

## Release Notes

### Version 1.0.0 - Initial Release

This is the first stable release of the Value at Risk Calculator. The library provides:

1. **Two VaR Calculation Methods**:
   - Parametric (assumes normal distribution)
   - Historical (uses empirical distribution)

2. **Flexible Configuration**:
   - Adjustable confidence levels
   - Multiple time horizons
   - Portfolio rebalancing support

3. **Robust Testing**:
   - 34 comprehensive unit tests
   - 100% test pass rate
   - Edge case coverage

4. **Well-Documented**:
   - Detailed README with examples
   - Mathematical background
   - API documentation
   - Usage guidelines

### Known Limitations

- Parametric VaR assumes normal distribution (may underestimate tail risk)
- Historical VaR limited by historical data availability
- Both methods may underestimate risk during extreme market events
- Assumes stationary returns (future behaves like past)

### Future Roadmap

See README.md section "Future Enhancements" for planned features including:
- Conditional VaR (CVaR) implementation
- Monte Carlo simulation method
- Cornish-Fisher VaR for non-normal distributions
- Portfolio optimization integration
- Interactive visualization dashboard
- Additional data sources
- Backtesting framework
- Risk decomposition by asset

---

## Version Format

Versions follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

## Categories

Changes are grouped by:
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

[Unreleased]: https://github.com/crout4108/valueAtRisk/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/crout4108/valueAtRisk/releases/tag/v1.0.0
