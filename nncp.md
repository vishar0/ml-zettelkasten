# [NNCP: Lossless Data Compression with Neural Networks](https://bellard.org/nncp/)

- **Created**: 2026-06-07
- **Last Updated**: 2026-06-10
- **Status**: `In Progress`

---

- [x] [2019] NNCP: Lossless Data Compression with Neural Networks - [paper](https://bellard.org/nncp/nncp.pdf)
- [ ] [2019] Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context - [paper](https://arxiv.org/abs/1901.02860). Used by NNCP v2. [[papers-ml-fundamentals]]
- [ ] [2021] NNCP v2: Lossless Data Compression with Transformer - [paper](https://bellard.org/nncp/nncp_v2.1.pdf)
- [ ] NNCP v3/v3.3 - pure C version; LibNC for tensorops not open?

## [2019] NNCP: Lossless Data Compression with Neural Networks

- **Date**: 2026-06-07
- **Paper**: <https://bellard.org/nncp/nncp.pdf>
- **Code**: <https://bellard.org/nncp/>

---

- **Abstract**:
  - > We describe our implementation of a lossless data compressor using neural networks. We tuned Long Short-Term Memory and Transformer based models in order to achieve a fast training convergence. We evaluated the performance on the widely used enwik8 Hutter Prize benchmark.
- **Core idea**:
  - NNCP stands for **Neural Network Compression Program**.
  - The compressor uses a neural network as an adaptive probability model, then uses arithmetic coding to convert the predicted probabilities into bits.
  - At each time step $t$, the model predicts the next symbol distribution $p(s_t | s_{<t})$, the arithmetic coder encodes the actual symbol $s_t$ at cost approximately $-\log_2 p(s_t)$ bits, and the model is updated on $s_t$.
  - The decoder runs the same loop: predict, decode the next symbol, update on that decoded symbol. Because the decoder recreates the same online training trajectory, the trained model weights do not need to be transmitted.
- **Compression = probabilistic prediction + arithmetic coding**:
  - For a model $M$ and data $X$, ideal code length is:
    $$L(X | M) = -\log_2 P_M(X)$$
  - For a sequence of symbols:
    $$-\log_2 P_M(X) = \sum_t -\log_2 P_M(s_t | s_{<t})$$
  - Arithmetic coding is the near-optimal mechanism that realizes this bound for any probabilistic model.
  - Intuition: if the model assigns probability `0.9` to the next symbol, the symbol costs about `0.152` bits; if it assigns probability `0.01`, it costs about `6.64` bits.
  - So the arithmetic coder is mostly not the hard part. The hard part is producing good next-symbol probabilities online.
- **Arithmetic coding intuition**:
  - Given an alphabet like `{A, B, C}`, the model partitions `[0, 1)` into intervals proportional to the current probabilities.
  - The actual symbol selects one subinterval. The next symbol recursively subdivides that selected interval using the updated conditional distribution.
  - Decoding uses the same model and the same interval refinements. The decoder keeps a code value, checks which interval it falls into, emits that symbol, renormalizes into the chosen interval, and repeats.
  - Concrete encoding example with a fixed distribution:

    ```text
    A: p = 0.5 -> interval [0.0, 0.5)
    B: p = 0.3 -> interval [0.5, 0.8)
    C: p = 0.2 -> interval [0.8, 1.0)

    sequence = B A C
    ```

  - Encoding starts with the full interval `[0, 1)`.

    ```text
    start: [0.0, 1.0)

    encode B:
      B occupies [0.5, 0.8)
      interval becomes [0.5, 0.8)

    encode A inside [0.5, 0.8):
      current width = 0.3
      A gets the first 50% of the interval
      interval becomes [0.5, 0.5 + 0.3 * 0.5) = [0.5, 0.65)

    encode C inside [0.5, 0.65):
      current width = 0.15
      C gets the last 20% of the interval
      interval becomes [0.5 + 0.15 * 0.8, 0.65) = [0.62, 0.65)
    ```

  - Any number in `[0.62, 0.65)` now represents the whole sequence `BAC`. For example, choose code value `z = 0.625`.
  - Decoding reverses the process. Start from `z = 0.625` and the initial intervals:

    ```text
    A: [0.0, 0.5)
    B: [0.5, 0.8)
    C: [0.8, 1.0)
    ```

  - Since `0.625` lies in `[0.5, 0.8)`, the first decoded symbol is `B`. Then renormalize by zooming into the `B` interval:

    ```text
    z <- (z - lower_B) / width_B
       = (0.625 - 0.5) / 0.3
       = 0.416666...
    ```

  - Now `0.416666...` lies in `A`'s interval `[0.0, 0.5)`, so the second decoded symbol is `A`. Renormalize again:

    ```text
    z <- (0.416666... - 0.0) / 0.5
       = 0.833333...
    ```

  - Now `0.833333...` lies in `C`'s interval `[0.8, 1.0)`, so the third decoded symbol is `C`.
  - In practice, the decoder also needs to know when to stop, either from the original length stored in the compressed file format or from an explicit EOF symbol.
  - Practical arithmetic coders do this with integer intervals and consume/emit bits as interval bounds become known. Conceptually, symbol costs still track `-log2(probability)`.
- **Adaptive coding / prequential evaluation**:
  - Adaptive coding means the probability model learns while the stream is being compressed.
  - The encoder loop is:

    ```text
    predict p(s_t | s_<t)
    arithmetic-code the actual symbol s_t
    update the model on s_t
    repeat
    ```

  - The decoder can mirror the same loop because after decoding `s_t`, it also knows the same prefix:

    ```text
    predict p(s_t | s_<t)
    arithmetic-decode s_t
    update the model on s_t
    repeat
    ```

  - Simple count-model example for alphabet `{A, B, C}` with add-one initialization:

    ```text
    initial counts: A=1, B=1, C=1
    initial probs:  A=1/3, B=1/3, C=1/3

    sequence = B B A

    step 1:
      predict B with p=1/3
      encode/decode B
      update counts -> A=1, B=2, C=1

    step 2:
      predict B with p=2/4
      encode/decode B
      update counts -> A=1, B=3, C=1

    step 3:
      predict A with p=1/5
      encode/decode A
      update counts -> A=2, B=3, C=1
    ```

  - NNCP replaces the count update with a neural-network training step:

    ```text
    predict next symbol
    encode/decode actual symbol
    compute loss on actual symbol
    update weights deterministically
    ```

  - This is why the trained weights do not need to be stored in the compressed file: the decoder recreates them by performing the same updates on the same decoded sequence.
  - The data is presented once, left to right. NNCP is therefore evaluated in a prequential setting: predict first, observe the true symbol, update, repeat.
  - This makes the result a measure of **single-pass convergence speed** as well as final modeling quality. Early mistakes cost real bits.
  - This differs from standard language model evaluation, where a model is usually trained offline for many updates, frozen, and then evaluated on a held-out test set.
- **Symbols**:
  - Without preprocessing, each symbol is one raw byte:

    ```text
    Ns = 256
    s_t = byte at position t
    ```

  - With preprocessing, each symbol is a byte sequence from a learned vocabulary:

    ```text
    Ns ~= 16,000
    s_t = byte-sequence symbol
    ```

  - Larger models use the subword-like preprocessor.
- **Preprocessing**:
  - The small LSTM reuses the CMIX/lstm-compress text preprocessor for comparability.
  - Larger models use a BPE-inspired byte-sequence preprocessor:
    - convert uppercase letters to lowercase plus escape codes,
    - add escape codes to reduce word-boundary variants,
    - iteratively merge symbol pairs chosen by expected zeroth-order entropy reduction,
    - remove symbols below a frequency threshold.
  - The preprocessor does not assume word separators, so it can apply to arbitrary byte streams/languages.
- **Models**:
  - NNCP evaluates two pure neural sequence models:
    - LSTM
    - Transformer
  - LSTM:
    - modified bounded LSTM cell,
    - layer normalization,
    - truncated BPTT over 20 time steps,
    - batch size 16,
    - Adam with `beta1 = 0`, `beta2 = 0.9999`, `epsilon = 1e-5`,
    - no dropout and no gradient clipping.
  - Transformer:
    - Transformer-XL-like recurrence/windowing,
    - learned relative positional embeddings,
    - context window `M = 128`,
    - truncated-like BPTT over 64 time steps,
    - batch size 1,
    - Adam with `beta1 = 0`, `beta2 = 0.9999`, `epsilon = 1e-5`,
    - no dropout and no gradient clipping.
- **Determinism**:
  - Lossless decompression requires the encoder and decoder to produce identical probabilities at every step.
  - Bellard developed a custom C neural compute library, the NC library, to make evaluation and training deterministic across CPU/OS combinations.
  - The library avoids external BLAS/ML dependencies and relies on basic IEEE 754-2008 32-bit floating point operations.
  - It represents forward evaluation as bytecode, automatically derives backward bytecode, and has an incremental per-timestep forward mode for decompression.
- **enwik8 setup**:
  - enwik8 is the first `100,000,000` bytes of the English Wikipedia XML dump used in large text compression benchmarks.
  - It is a lossless byte-exact benchmark, not a cleaned language modeling corpus like text8.
  - Reported metric:
    $$\text{bpb} = \frac{8 \cdot \text{compressed bytes}}{100000000}$$
  - **bpb** means **bits per input byte**. The uncompressed file uses exactly `8 bpb`, since each input byte is 8 bits.
  - A compressor with `2 bpb` stores each original byte using 2 compressed bits on average, i.e. a `4x` compression ratio relative to raw bytes.
  - A result of `1.34 bpb` means the compressed file is about `1.34 / 8 = 16.75%` of the original size, or about `5.97x` smaller than raw bytes.
  - bpb is also the average negative log probability assigned to the observed input bytes after accounting for preprocessing and arithmetic coding:
    $$\text{bpb} \approx \frac{1}{N_\text{bytes}}\sum_t -\log_2 p(s_t | s_{<t})$$
  - Lower bpb means the model assigned higher probability to the true next symbols throughout the file.
- **Main enwik8 results**:
  - `gzip -9`: `36,445,248` bytes, `2.92 bpb`
  - `xz -9`: `24,865,244` bytes, `1.99 bpb`
  - `lstm-compress`: `20,494,577` bytes, `1.64 bpb`
  - `CMIX v17`: `14,877,373` bytes, `1.19 bpb`
  - `LSTM small`: `20,500,039` bytes, `1.64 bpb`
  - `Transformer`: `18,126,936` bytes, `1.45 bpb`
  - `LSTM large1`: `16,981,765` bytes, `1.36 bpb`
  - `LSTM large2`: `16,791,077` bytes, `1.34 bpb`
- **Interpretation**:
  - The paper does not beat CMIX on enwik8.
  - Its result is that relatively simple pure neural compressors can get surprisingly close to strong hand-engineered compressors.
  - In this 2019 setup, LSTM beats Transformer because the Transformer converges more slowly in the single-pass online setting, even though Transformers perform well in standard offline language modeling benchmarks.
- **Comparison caveats**:
  - These results should not be directly compared to ordinary language model perplexity:
    - NNCP does a single pass over the file, rather than many training epochs.
    - The bpb is averaged over the whole file, not just a held-out tail such as the last 5 MB.
    - A normal LM benchmark usually ignores the cost of transmitting model parameters.
  - Conceptually, compression should include model/program size:
    $$L(M) + L(X | M)$$
  - NNCP avoids transmitting trained weights because both encoder and decoder learn them online, but the model architecture/program still needs to be known by the decoder.
  - The paper does not include the size of the preprocessing dictionary or decompression program in the table. Bellard notes that the compressed preprocessing dictionary is about `60 kB`, around `0.005 bpb` on enwik8.
- **Takeaway**:
  - NNCP is best read as a clean experiment in online neural density modeling for lossless compression.
  - The core question is not just "how good is the final model?", but "how fast can the model become good while seeing the data once?"

## [2021] NNCP v2: Lossless Data Compression with Transformer

- **Date**: 2026-06-07
- **Paper**: <https://bellard.org/nncp/nncp_v2.1.pdf>
- **Code**: <https://bellard.org/nncp/>

---

- **Core idea**:
  - NNCP v2 is a follow-up to the 2019 NNCP paper focused on making the Transformer variant competitive.
  - The v1 paper found that the Transformer had worse compression than the LSTM because it converged too slowly in the online compression setting.
  - v2 improves the Transformer enough to beat CMIX on enwik9, at the cost of more compute and a more complex training schedule.
- **Model**:
  - The model is based on Transformer-XL.
  - Changes relative to the standard Transformer-XL setup:
    - learned relative positional embeddings instead of sinusoidal relative positional embeddings,
    - untied embeddings,
    - scaled relative positional bias term to improve initial convergence,
    - `GELU` instead of `ReLU` in the feed-forward layer,
    - unusual initialization where weights, except biases and layer-norm weights, are initialized to the same value,
    - second feed-forward linear transform is scaled to improve convergence,
    - no dropout during normal compression/decompression.
  - Dropout is used only during the retraining phase.
- **Preprocessor**:
  - v2 reuses the NNCP v1 preprocessor.
  - This means the model still predicts preprocessed byte-sequence symbols rather than raw bytes for the larger models.
- **Training / compression schedule**:
  - v2 explicitly optimizes for both compression and decompression speed.
  - It uses identical computational steps in the encoder and decoder by encoding or decoding a single symbol, or batch of symbols, per training step.
  - Training segments have length `192` symbols.
  - Batch size is `64`, mainly to exploit GPU parallelism.
  - There is large overlap between training segments. This increases compute, but improves GPU utilization and keeps encoder/decode computation symmetric.
- **Retraining over already decoded data**:
  - The most important change from v1 is periodic retraining on already decompressed data.
  - Motivation: even after Transformer improvements, online convergence is still slower than the previous LSTM model.
  - v2 therefore periodically retrains on recent history to improve compression.
  - Bellard describes the example setting as:

    ```text
    retrain the past 10M symbols every 500k symbols
    equivalent of 20 epochs
    ```

  - This is no longer pure one-pass learning in the strict sense. It is still a valid compressor because the decoder has already decoded the same past data and can perform the same retraining deterministically.
  - Dropout is used during retraining to reduce overfitting.
- **Optimizer**:
  - Adam with:

    ```text
    beta1 = 0
    beta2 = 0.9999
    epsilon = 1e-9
    ```

  - With `beta1 = 0`, Adam has no first-moment momentum and is close to RMSProp with bias correction.
  - This is plausible for online compression because stale directional momentum can hurt adaptation when the stream changes, while `beta2` still provides stable per-parameter scaling.
  - Bellard notes that separate Adam contexts for normal training and retraining are important because gradient norms differ.
  - Gradient normalization is essential to avoid divergence.
  - No warmup is used; the learning rate decreases linearly during training.
- **Implementation / determinism**:
  - Unlike the 2019 paper's custom C NC library, v2 is implemented in PyTorch so it can run on a GPU.
  - It uses PyTorch deterministic mode to keep encoder and decoder identical.
  - Bellard notes the determinism guarantee only holds with the exact same hardware and software versions.
  - 16-bit floating point operations are used to reduce runtime.
  - This is more practical for GPU experimentation, but less robustly portable than the deterministic custom C approach.
- **Main enwik8 results**:
  - `gzip -9`: `36,445,248` bytes, `2.92 bpb`
  - `xz -9`: `24,865,244` bytes, `1.99 bpb`
  - `CMIX v18`: `14,838,332` bytes, `1.19 bpb`
  - `NNCP v1`: `16,292,774` bytes, `1.30 bpb`
  - `NNCP v2 base`: `15,600,675` bytes, `1.25 bpb`
  - `NNCP v2 large`: `15,020,691` bytes, `1.20 bpb`
  - On enwik8, v2 improves substantially over v1 but still does not quite beat CMIX.
- **Main enwik9 results**:
  - `gzip -9`: `322,591,995` bytes, `2.58 bpb`
  - `xz -9`: `197,331,816` bytes, `1.58 bpb`
  - `CMIX v18`: `115,714,367` bytes, `0.926 bpb`
  - `NNCP v1`: `119,167,224` bytes, `0.953 bpb`
  - `NNCP v2 base`: `114,217,584` bytes, `0.914 bpb`
  - `NNCP v2 large`: `112,219,309` bytes, `0.898 bpb`
  - On enwik9, v2 beats CMIX despite not beating it on enwik8.
- **Scale transition**:
  - v2 highlights an important scale transition:
    - at `100 MB` enwik8, CMIX remains slightly better,
    - at `1 GB` enwik9, the larger Transformer model beats CMIX.
  - This suggests that higher-capacity neural models need enough data/compute to overcome their slower adaptation and exploit their capacity.
- **Comparison caveats**:
  - Bellard again warns that the results are not directly comparable to standard language-modeling results:
    - bpb is averaged over the whole file rather than a test dataset,
    - model parameters would have to be stored if the model were pretrained/frozen.
  - The preprocessing dictionary and decompression program sizes are again not included in the table.
  - For v2, the retraining phase also complicates the "single pass" interpretation: the compressed stream is decoded once, but the model may train multiple times over already decoded recent history.
- **Takeaway**:
  - v2 is the first practical NNCP Transformer result that beats the best text compressors on enwik9.
  - The win depends on more compute, GPU execution, larger models, overlapping training segments, and retraining over decoded history.
  - The paper is less about a pure one-pass online learner than v1, and more about making a Transformer-based adaptive compressor strong enough at billion-byte scale.
