# AAC Codec Project

Educational AAC (Advanced Audio Coding) encoder/decoder implementation based on MPEG-2/MPEG-4 AAC and 3GPP TS 26.403 specifications.

## Project Structure

```
aac-codec/
├── part1/          # Level 1: SSC + Filterbank (COMPLETE)
├── part2/          # Level 2: + TNS (IN PROGRESS)
├── part3/          # Level 3: + Psychoacoustic + Quantizer + Huffman (STUBS)
├── docs/           # Specifications and band tables
└── pyproject.toml  # Project config (uv managed)
```

## Implementation Status

| Part | Components | Status |
|------|------------|--------|
| 1 | SSC, Filterbank, MDCT/IMDCT | Complete |
| 2 | TNS (Temporal Noise Shaping) | Stubs only |
| 3 | Psychoacoustic, Quantizer, Huffman | Stubs only |

## Key Specifications

- **Sample rate**: 48kHz stereo
- **Frame size**: 2048 samples (50% overlap)
- **Short frames**: 8 subframes of 256 samples each
- **MDCT output**: 1024 coeffs (long), 128 coeffs (short)
- **Window types**: KBD (Kaiser-Bessel-Derived), SIN (Sinusoid)
- **Frame types**: OLS, LSS, ESH, LPS

## Part 2: TNS Algorithm Summary

TNS shapes quantization noise in the temporal domain using LPC in frequency domain.

### Processing Steps (Encoder)

1. **Normalize MDCT coeffs** by band energy (Eq 2-4)
   - P(j) = sum of X(k)^2 in band j
   - Sw(k) = sqrt(P(j)) for coeffs in band j
   - Smooth Sw: forward then backward averaging
   - Xw(k) = X(k) / Sw(k)

2. **Compute LPC coefficients** (order p=4)
   - Solve Ra = r (normal equations)
   - R = autocorrelation matrix of Xw
   - r = autocorrelation vector

3. **Quantize LPC coeffs**: 4-bit, step=0.1, range [-0.8, 0.8]

4. **Check filter stability**: all poles of inverse filter inside unit circle

5. **Apply FIR filter** to original MDCT:
   - H_TNS(z) = 1 - a1*z^-1 - a2*z^-2 - ... - ap*z^-p

### Decoding

Apply IIR inverse filter: H_TNS^(-1)(z) = 1 / H_TNS(z)

## Band Tables (TableB219.mat)

- **B219a**: 69 bands for long frames (1024 MDCT coeffs)
- **B219b**: 42 bands for short frames (128 MDCT coeffs)
- Columns: index, w_low, w_high, bval, qsthr

Load with:
```python
from scipy.io import loadmat
mat_data = loadmat("docs/TableB219.mat")
B219a = mat_data['B219a']  # long frames
B219b = mat_data['B219b']  # short frames
```

## Running

```bash
# Demo part 1
uv run python -m part1.demo_aac_1

# Demo part 2 (after implementation)
uv run python -m part2.demo_aac_2

# Tests
uv run pytest part1/tests/ -v
uv run pytest part2/tests/ -v
```

## Key Files

### Part 1 (reference)
- `part1/ssc.py`: Frame type detection via transient detection
- `part1/filterbank.py`: MDCT/IMDCT with KBD/SIN windows
- `part1/aac_coder_1.py`: Main encode/decode pipeline

### Part 2 (to implement)
- `part2/tns.py`: TNS functions (11 functions, all stubs)
- `part2/aac_coder_2.py`: Extended pipeline with TNS
- `part2/constants.py`: TNS parameters (TNS_ORDER=4, etc.)

## Dependencies

- numpy, scipy, soundfile
- scipy.signal.lfilter for filtering
- scipy.linalg.solve_toeplitz for LPC (or numpy.linalg.solve)
- numpy.polynomial.polynomial.Polynomial.roots() for stability check
