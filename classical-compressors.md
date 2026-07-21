# Classical Compressors

- **Created**: 2026-07-21
- **Last Updated**: 2026-07-21
- **Status**: `Not Started`
- **Description**: Mechanisms and engineering of non-neural lossless codecs.
- **Related**:
  - [[compression]] — The compression–prediction duality turns classical compressors into next-symbol predictors.

---

## Basics

- **Lossless vs lossy**: lossless compression reproduces the original bytes exactly, while lossy compression discards information to reduce size further. gzip, zstd, xz, bzip2, PNG, and FLAC are lossless; JPEG is ordinarily lossy.
- **General-purpose vs domain-specific**: gzip, zstd, and xz operate on arbitrary bytes, while codecs such as PNG and FLAC build in inductive biases about images and audio. Domain-specific codecs win when those assumptions match the data.
- **Compression pipeline**: raw data → reversible representation or transform → model (matches, deltas, or symbol probabilities) → entropy coding (Huffman, arithmetic/range, or ANS) or bit-packing → framed bitstream/file format.
- **No free lunch**: no lossless compressor can shorten every input, and no codec is best across every source and engineering objective. Compression trades size against speed, memory, and random access by imposing assumptions that help on some data and hurt on others.
- **Compression–prediction duality**: associate a sequence with code length $L(x)$ with the score $p(x)=2^{-L(x)}$. Then
  $$
  \begin{aligned}
  p(x_t \mid x_{<t})
  &= \frac{p(x_{1:t})}{p(x_{1:t-1})} \\
  &= \frac{2^{-L(x_{1:t})}}{2^{-L(x_{1:t-1})}} \\
  &= 2^{-\left[L(x_{1:t})-L(x_{1:t-1})\right]}.
  \end{aligned}
  $$
  A continuation that adds fewer compressed bits is therefore more probable. Scoring each possible next symbol this way turns a classical compressor into a predictor and generator. [[compression]] develops the information-theoretic connection.

## Main Families

### General-purpose byte-stream compressors

These codecs treat input as opaque bytes, exploiting generic repetition and symbol statistics without knowing its type, modality, or schema.

| Format / tool | Core mechanism | Main tradeoff |
| --- | --- | --- |
| gzip / DEFLATE | LZ77 matches + Huffman coding; fixed 32 KiB window | Compatibility |
| zstd | LZ77-family matching + Huffman/FSE; configurable window and optional dictionary | Strong general-purpose speed/ratio balance |
| xz, usually LZMA2 | Large LZ dictionary + adaptive probability model + range coding | Better ratios, slower compression |
| bzip2 | Burrows–Wheeler transform + Huffman coding | Older high-ratio block codec; not Lempel–Ziv |
| LZ4 / Snappy | Speed-focused LZ-family matching | Throughput over ratio |

### Type- and format-aware compressors

These methods show that representation can expose structure before modeling, that compression gains often come from inductive biases matched to the data, and that real systems optimize decode cost, memory bandwidth, and random access alongside size. Compression-based model comparisons therefore measure both representation and prediction and must account for what structural information each method receives.

| Format / tool | Core mechanism | Main use |
| --- | --- | --- |
| ALP | Adaptive, vectorized floating-point encoding | Floating-point columns |
| FSST | Static substring-to-symbol table | Short string columns and random access |
| FastLanes | Cascaded, data-parallel encodings | Typed and correlated columns |
| OpenZL | Self-describing graph of format-specific transforms/codecs | Structured formats with a universal decoder; falls back to zstd |

## Reading List

### Generic compression foundations and implementations

- [ ] [1977] [Ziv,Lempel] A Universal Algorithm for Sequential Data Compression - [paper](https://ieeexplore.ieee.org/document/1055714)
- [ ] [1996] GZIP File Format Specification - [RFC 1952](https://datatracker.ietf.org/doc/html/rfc1952)
- [ ] [1996] DEFLATE Compressed Data Format Specification - [RFC 1951](https://datatracker.ietf.org/doc/html/rfc1951)
- [ ] [2021] [Collet,Kucherawy] Zstandard Compression - [RFC 8878](https://datatracker.ietf.org/doc/html/rfc8878)
- [ ] [code] [YannCollet,Meta] zstd reference implementation - [code](https://github.com/facebook/zstd)
- [ ] [Pavlov] LZMA SDK - [code and specification](https://www.7-zip.org/sdk.html)
- [ ] .xz container and its LZMA2 filter - [specification](https://tukaani.org/xz/xz-file-format.txt)
- [ ] bzip2 and libbzip2 - [manual](https://sourceware.org/bzip2/manual/manual.pdf)

### Lightweight columnar compression

- [ ] [2024] [Afroozeh,Kuffó,Boncz] ALP: Adaptive Lossless floating-Point Compression - [paper](https://doi.org/10.1145/3626717), [author PDF](https://ir.cwi.nl/pub/33334/33334.pdf), [code](https://github.com/cwida/ALP)
- [ ] [2020] [Boncz,Neumann,Leis] FSST: Fast Random Access String Compression - [paper](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf), [code](https://github.com/cwida/fsst)
- [ ] [2025] [Afroozeh,Boncz] The FastLanes File Format - [paper](https://www.vldb.org/pvldb/vol18/p4629-afroozeh.pdf)

### Format-aware compression

- [ ] [2025] [Collet et al.] OpenZL: A Graph-Based Model for Compression - [paper](https://arxiv.org/abs/2510.03203), [code](https://github.com/facebook/openzl), [Meta overview](https://engineering.fb.com/2025/10/06/developer-tools/openzl-open-source-format-aware-compression-framework/), [talk](https://www.youtube.com/watch?v=PyPViJiCewM)
