# AAC Audio Codec Implementation

Educational implementation of simplified AAC (Advanced Audio Coding) encoder/decoder based on MPEG-2/MPEG-4 AAC standards.

## Project Structure

```
aac-codec/
├── docs/
│   ├── mm-2025-hw-v0.1.pdf        # Assignment specification
│   └── TableB219.mat              # Psychoacoustic band tables
├── part1/                          # Level 1: SSC + Filterbank
│   ├── ssc.py                      # Sequence Segmentation Control
│   ├── filterbank.py               # MDCT/IMDCT transforms
│   ├── aac_coder_1.py              # Encoder/decoder pipeline
│   ├── demo_aac_1.py               # Demo with SNR
│   ├── types.py                    # Type definitions
│   ├── constants.py                # Constants
│   └── tests/                      # Test suite
├── part2/                          # Level 2: + TNS
│   ├── tns.py                      # Temporal Noise Shaping
│   ├── aac_coder_2.py              # Extended encoder/decoder
│   ├── demo_aac_2.py               # Demo with SNR
│   ├── types.py                    # Extended types
│   ├── constants.py                # Extended constants
│   ├── ssc.py, filterbank.py       # Copied from part1
│   └── tests/                      # Extended test suite
└── part3/                          # Level 3: Full Codec
    ├── psychoacoustic.py           # Psychoacoustic model
    ├── quantizer.py                # Perceptual quantization
    ├── aac_coder_3.py              # Full encoder/decoder
    ├── demo_aac_3.py               # Demo with metrics
    ├── types.py                    # Full types
    ├── constants.py                # Full constants
    ├── ssc.py, filterbank.py, tns.py
    └── tests/                      # Full test suite
```

## Implementation Status

### Part 1: SSC + Filterbank
- **SSC**: Frame type detection (OLS, LSS, ESH, LPS)
- **Filterbank**: MDCT/IMDCT with KBD and SIN windows
- **Perfect reconstruction** via overlap-add

### Part 2: + TNS
- All Part 1 features
- **TNS**:
  - MDCT normalization by band energy
  - LPC coefficient computation
  - Quantization (4-bit, step 0.1)
  - Filter stability checking
  - FIR/IIR filtering

### Part 3: Full Codec
- All Part 1 & 2 features
- **Psychoacoustic**:
  - Spreading function
  - FFT magnitude/phase prediction
  - Predictability & tonality computation
  - SMR calculation
- **Quantizer**:
  - Non-uniform quantization
  - Scalefactor estimation & refinement
  - DPCM encoding
- **Huffman**: (uses provided huff_utils.py)

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e ".[dev]"
```

### Usage

#### Part 1: Basic Codec
```python
from part1 import aac_coder_1, i_aac_coder_1, demo_aac_1

# Encode/decode
aac_seq = aac_coder_1("input.wav")
audio = i_aac_coder_1(aac_seq, "output.wav")

# Demo with SNR
snr = demo_aac_1("input.wav", "output.wav")
print(f"SNR: {snr:.2f} dB")
```

#### Part 2: With TNS
```python
from part2 import aac_coder_2, i_aac_coder_2, demo_aac_2

# Encode/decode with TNS
aac_seq = aac_coder_2("input.wav")
audio = i_aac_coder_2(aac_seq, "output.wav")

# Demo
snr = demo_aac_2("input.wav", "output.wav")
```

#### Part 3: Full Codec
```python
from part3 import aac_coder_3, i_aac_coder_3, demo_aac_3

# Full encode/decode
aac_seq = aac_coder_3("input.wav", "coded.mat")
audio = i_aac_coder_3(aac_seq, "output.wav")

# Demo with all metrics
snr, bitrate, compression = demo_aac_3("input.wav", "output.wav", "coded.mat")
print(f"SNR: {snr:.1f} dB | Bitrate: {bitrate/1000:.1f} kbps | Compression: {compression:.1f}x")
```

### Running Tests

```bash
# All tests
pytest

# Specific part
pytest part1/tests/ -v
pytest part2/tests/ -v
pytest part3/tests/ -v

# With coverage
pytest --cov=part1 --cov=part2 --cov=part3 --cov-report=html

# Specific module
pytest part2/tests/test_tns.py -v
pytest part3/tests/test_psychoacoustic.py -v
```

## Development

### Code Quality Tools

Configured in `pyproject.toml`:
- **Black** (88 chars) - formatting
- **Ruff** - fast linting (E, W, F, I, N, UP, B, C4, SIM)
- **Mypy** - strict type checking
- **isort** - import sorting (black profile)
- **Pydocstyle** - NumPy convention
- **Bandit** - security checks

```bash
# Format
black part1/ part2/ part3/

# Lint
ruff check part1/ part2/ part3/

# Type check
mypy part1/ part2/ part3/

# All checks
black --check . && ruff check . && mypy .
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```


## References

- MPEG-2 AAC: ISO/IEC 13818-7
- MPEG-4 AAC: ISO/IEC 14496-3
- 3GPP TS 26.403
- Course docs: `docs/mm-2025-hw-v0.1.pdf`
