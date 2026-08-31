# Rotated Scaled Lloyd-Max (RSLM) Vector Quantization for Approximate Nearest Neighbor Search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/google-research/blob/master/rslm/rslm.ipynb)

A self-contained, standalone Python implementation of the **Rotated Scaled Lloyd-Max (RSLM)** family of vector quantization codecs.

RSLM enables high-ratio compression of dense floating-point embeddings (e.g., text, multimodal, image representations) down to **1.0 – 4.0 bits per dimension (7.4x – 28.4x compression)** with high reconstruction fidelity for approximate nearest neighbor (ANN) search.

---

## Paper & Citation

This code is the official reference implementation for the research paper:

> **Rotated Scaled Lloyd-Max Vector Quantization**
> *Link:* [arXiv / Paper Link (forthcoming upon public publication)]

If you find this codebase or algorithm useful in your research, please cite our paper once published:

```bibtex
@article{rslm2026,
  author    = {Lenhardt, Rastislav and Dobos, Teodora and Vecchiato, Thomas and I{\v{s}}a, Ji{\v{r}}{\'\i} and Ginzburg, Igor},
  title     = {{RSLM: Training-Free Vector Quantization for Approximate Nearest Neighbor Search}},
  journal   = {arXiv preprint},
  year      = {2026},
}
```

---

## Key Features

- **Cascaded Fast Walsh-Hadamard Transform (FWHT)**: Randomized orthogonal rotations uniformize coordinate distributions into independent Gaussians without altering Euclidean distances.
- **Optimal Lloyd-Max Codebooks**: Precomputed 1D, 2D, and 4D vector quantization codebooks optimized for standard Gaussian marginals.
- **2-Byte `Ue7m9` Scale Factor**: A compact unsigned floating-point format (7 exponent bits, 9 mantissa bits) that reconstructs vector energy in 16 bits.
- **Multi-Stage Residual Quantization**:
  - **Direct Quantization (1-Stage)**: Whole-vector compression ($\hat{x} = \text{RSLM}(x)$).
  - **Relative to Partition Centers (2-Stage)**: Residual quantization relative to $K$-means cluster centroids ($\hat{x} = c + \text{RSLM}(x - c)$).
  - **Relative to Approximate Vectors (3-Stage Refinement)**: Precomputes full-corpus coarse approximate vectors using lightweight RSLM1 ($1.36$ bpd) and encodes fine-grained residuals relative to approximate vectors ($\hat{x} = a(x) + \text{RSLM}_k(x - a(x))$).

---

## Supported Codec Family

| Codec | Bit-rate | Codebook Type | Scale Factor | 100d Direct Size | 100d Relative Size | Compression Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Rslm4`** | 4 bits / dim | 1D Lloyd-Max (16 centroids) | 2-byte `Ue7m9` | 52 Bytes | 54 Bytes | **7.4x – 7.7x** |
| **`Rslm4Lite`** | 4 bits / dim | 1D Lloyd-Max | Embedded in MSB (0 extra bytes) | 50 Bytes | — | **8.0x** |
| **`Rslm3`** | 3 bits / dim | 1D Lloyd-Max (8 centroids, packed) | 2-byte `Ue7m9` | 41 Bytes | 43 Bytes | **9.3x – 9.8x** |
| **`Rslm2`** | 2 bits / dim | 2D Joint Lloyd-Max (16 2D centroids) | 2-byte `Ue7m9` | 27 Bytes | 29 Bytes | **13.8x – 14.8x** |
| **`Rslm1`** | 1 bit / dim | 4D Joint Lloyd-Max (16 4D centroids) | 2-byte `Ue7m9` | 15 Bytes | 17 Bytes | **23.5x – 26.7x** |

---

## Quick Start & Reproduction

The complete, end-to-end benchmark on the **GloVe-100** dataset is provided in the standalone notebook [`rslm.ipynb`](rslm.ipynb). It has zero internal dependencies and runs with standard scientific Python packages.

### Option 1: Open in Google Colab
Click the badge above on the top to open in Google Colab.

### Option 2: Run Locally via Jupyter

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter Notebook
jupyter notebook rslm.ipynb
```

---

## Disclaimer

*This is not an officially supported Google product.*

This project is intended for research reproduction and demonstration purposes only. Issues and contributions are handled on a best effort basis.
