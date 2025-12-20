# AAC Audio Codec Implementation

Educational implementation of a simplified AAC (Advanced Audio Coding) encoder/decoder based on MPEG-2/MPEG-4 AAC standards.

## Project Structure

```
aac-codec/
├── README.md
├── requirements.txt
├── pyproject.toml
├── docs/
│   └── mm-2025-hw-v0.1.pdf    # Assignment specification
├── part1/                      # Level 1: SSC + Filterbank (1 point)
│   ├── ssc.py                  # Sequence Segmentation Control
│   ├── filterbank.py           # MDCT/IMDCT transforms
│   ├── aac_coder_1.py          # Main encoder/decoder
│   ├── demo_aac_1.py           # Demo and metrics
│   ├── types.py                # Type definitions
│   ├── constants.py            # Configuration constants
│   └── tests/                  # Test suite
├── part2/                      # Level 2: + TNS (2 points)
│   └── (to be implemented)
└── part3/                      # Level 3: + Psychoacoustic + Quantizer + Huffman (4 points)
    └── (to be implemented)
```

## Features by Part

### Part 1 (1 point) ✓ Structure Ready
- **Sequence Segmentation Control (SSC)**: Detects transients and selects frame types
- **Filterbank**: MDCT/IMDCT with KBD and SIN windows
- **Frame Types**: OLS, LSS, ESH, LPS
- **Perfect Reconstruction**: Using overlap-add

### Part 2 (+1 point)
- All Part 1 features
- **Temporal Noise Shaping (TNS)**: Linear prediction on MDCT coefficients

### Part 3 (+2 points)
- All Part 1 & 2 features
- **Psychoacoustic Model**: Masking thresholds
- **Quantization**: Perceptual quantization with scale factors
- **Huffman Coding**: Entropy coding of coefficients

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Usage

```python
from part1 import aac_coder_1, i_aac_coder_1, demo_aac_1

# Encode WAV file
aac_seq = aac_coder_1("input.wav")

# Decode back to WAV
audio = i_aac_coder_1(aac_seq, "output.wav")

# Run demo with SNR calculation
snr = demo_aac_1("input.wav", "output.wav")
print(f"SNR: {snr:.2f} dB")
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=part1 --cov-report=html

# Run specific module
pytest part1/tests/test_ssc.py -v
```

## Implementation Status

- [x] Part 1 structure created
  - [x] Type definitions
  - [x] Constants
  - [x] Function signatures and docstrings
  - [x] Test structure
  - [ ] Implementation
- [ ] Part 2 structure
- [ ] Part 3 structure

## Technical Specifications

- **Sample Rate**: 48 kHz
- **Channels**: Stereo (2 channels)
- **Frame Size**: 2048 samples (50% overlap)
- **MDCT Output**: 1024 coefficients (long), 128 coefficients (short)
- **Window Types**: Kaiser-Bessel-Derived (KBD), Sinusoidal (SIN)

## Development

### Quick Start

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (runs quality checks automatically)
make setup-hooks

# Run all quality checks manually
make quality

# Run tests
make test
```

### Code Quality

This project uses automated code quality tools:

- **Black** - Code formatting
- **isort** - Import sorting
- **Ruff** - Fast linting
- **Mypy** - Type checking
- **Pydocstyle** - Docstring validation
- **Bandit** - Security checks
- **Pre-commit** - Automatic checks before commit

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all checks
make quality

# See all available commands
make help
```

### Pre-Commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually
pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

### Testing Philosophy

- Unit tests for all helper functions
- Integration tests for main components
- Fixtures for common test data
- High coverage target (>90%)

```bash
# Run tests
pytest

# With coverage
pytest --cov=part1 --cov-report=html

# Or use Makefile
make test-cov
```

## References

- MPEG-2 AAC Standard (ISO/IEC 13818-7)
- MPEG-4 AAC Standard (ISO/IEC 14496-3)
- 3GPP TS 26.403 (Enhanced aacPlus)

## License

Educational use only.
