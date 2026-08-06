# Vision

- **Created**: 2025-07-21
- **Last Updated**: 2026-08-04
- **Status**: `In Progress`

---

- [X] [2021] CLIP: Learning Transferable Visual Models From Natural Language Supervision — [paper](https://arxiv.org/abs/2103.00020)
- [ ] [2022] OpenCLIP: Reproducible scaling laws for contrastive language-image learning — [paper](https://arxiv.org/abs/2212.07143)
  - [ ] OpenCLIP code: <https://github.com/mlfoundations/open_clip>
- [X] [2023] MetaCLIP: Demystifying CLIP Data — [paper](https://arxiv.org/abs/2309.16671)
- [x] [2020] ImageGPT: Generative Pretraining from Pixels — [paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)
- [ ] [2020] VQGAN: Taming Transformers for High-Resolution Image Synthesis - [paper](https://arxiv.org/abs/2012.09841)
- [X] [2022] RQ-VAE & RQ-Transformer: Autoregressive Image Generation using Residual Quantization — [paper](https://arxiv.org/abs/2203.01941)
- [X] [2022] MaskGIT: Masked Generative Image Transformer — [paper](https://arxiv.org/abs/2202.04200)
- [x] [2021] ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- [ ] [2021] DINO: Emerging Properties in Self-Supervised Vision Transformers — [paper](https://arxiv.org/abs/2104.14294)
- [ ] [2023] DINOv2: Learning Robust Visual Features without Supervision — [paper](https://arxiv.org/abs/2304.07193)
- [ ] [2025] DINOv3 - [paper](https://arxiv.org/abs/2508.10104)
- [ ] [2024] VAR: Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction - [paper](https://arxiv.org/abs/2404.02905)
- [ ] [paper](https://arxiv.org/abs/2506.22355)
- [ ] Cambrian (Saining): <https://cambrian-mllm.github.io/>, [paper](https://arxiv.org/abs/2406.16860)
- [ ] SAM papers
- [ ] Flamingo: a Visual Language Model for Few-Shot Learning — [paper](https://arxiv.org/abs/2204.14198)

## [2021] CLIP: Learning Transferable Visual Models From Natural Language Supervision

- **Date**: 2025-07-21
- **Arxiv**: <https://arxiv.org/abs/2103.00020>
- **Paperpile**: <https://app.paperpile.com/view/?id=9ca94ee0-8932-434a-8c28-5d67ba36741d>
- **Assistant**: <https://chatgpt.com/share/687ea62a-ad50-8005-8e98-9defc0dc34bc>
- **CLIP Loss Implementation**: <https://colab.research.google.com/drive/1YqGGDPXh6fyAmdJXyb3163c698DOVze2?usp=sharing>

---

- (1) Intro
  - Motivation
    - Fully supervised object classification methods so far have been on fixed set of classes, not open domain.
    - Weakly supervised methods like URU (trained on instagram hashtags) showed such methods at scale to be effective pre-trained models, and improved ImageNet performance by 5% when finetuned. But still a fixed set of classes.
    - Everything so far use fixed set of classes, static softmax classifiers to perform prediction and lack a mechanism for dynamic outputs. This severely limits flexibility and limits zero-shot capabilities.
    - Fully open domain classification trained with natural language supervision has been tried in the past, but not at scale like URU and other methods focused on weak supervision but with fixed vocab. CLIP closes this gap.
  - CLIP
    - Dataset of 400 million (image, text) pairs, series of 8 models trained at spanning 2 orders of magnitude of compute. Observed that transfer performance is a smoothly predictable function of compute (scaling laws).
    - Learns to perform a wide set of tasks including OCR, geo-localization, action recognition, etc.
    - Outperforms the best available ImageNet model while also being more computationally efficient.
- (2) Approach
  - > Learning  from  natural  language  has  several  potential strengths  over  other  training  methods.   It’s  much  easier to scale natural language supervision compared to standard crowd-sourced labeling for image classification since it does not require annotations to be in a classic “machine learning compatible format” such as the canonical 1-of-N majority vote “gold label”. Instead, methods which work on natural language can learn passively from the supervision contained in the vast amount of text on the internet.  Learning from natural language also has an important advantage over most unsupervised or self-supervised learning approaches in that it doesn’t “just” learn a representation but also connects that representation to language which enables flexible zero-shot transfer. In the following subsections, we detail the specific approach we settled on.
  - > Given a batch of N (image, text) pairs, CLIP is trained to predict which of the N × N possible (image, text) pairings across a batch actually occurred. To do this, CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the $N^2 - N$ incorrect pairings. We optimize a symmetric cross entropy loss over these similarity scores.
  - At test time for zero-shot classification given a dataset with target classes: image is fed through the image encoder, and the target classes through the text encoder, and the target class with the max cosine similarity between the given image and each of the target classes is chosen.
  - Psuedocode in Fig 3. Notable things:
    - L2 normalization of image and text embeddings before dot product for cosine similarity.
    - Learned temperature parameter rather than being a hyperparam. `logits = np.dot(I_e, T_e^T) * np.exp(t)`, where `I_e` is the image embedding (L2 normalized) of shape `(N, E)`, `T_e` is the text embedding (L2 normalized) of shape `(N, E)`, and `t` is the learned temperature parameter. `logits` is the final scaled pairwise cosine similarities of shape `(N, N)`.
    - Symmetric cross-entropy loss: ((cross-entropy loss of correct text for each image in batch) + (cross-entropy loss of correct image for each text in batch)) / 2
      - Basically, logits is an `(N, N)` matrix. Do softmax reduction over rows first and compute negative log-likelihood for image-to-text matching, and then softmax reduction over cols and compute negative log-likelihood for text-to-image matching, and finally take the average of the two.
  - See my CLIP loss impl in <https://colab.research.google.com/drive/1YqGGDPXh6fyAmdJXyb3163c698DOVze2?usp=sharing>
  - (3) Experiments / Analyses
    - > In computer vision, zero-shot learning usually refers to the study of generalizing to unseen object categories in image classification (Lampert et al., 2009).  We instead use the term in a broader sense and study generalization to unseen datasets.  We motivate this as a proxy for performing un- seen tasks, as aspired to in the zero-data learning paper of Larochelle et al. (2008). While much research in the field of unsupervised learning focuses on the representation learn- ing capabilities of machine learning systems, we motivate studying zero-shot transfer as a way of measuring the task- learning capabilities of machine learning systems. In this view, a dataset evaluates performance on a task on a spe- cific distribution.
      - Inspirational, considering how I felt about CV in 2018.
      - Some organizational thoughts: <https://chatgpt.com/c/687fe733-2c4c-8005-afb5-6555db88dd39>
    - > Our focus on studying zero-shot transfer as an evaluation of task learning is inspired by work demonstrating task learn- ing in the field of NLP. To our knowledge Liu et al. (2018) first identified task learning as an “unexpected side-effect” when a language model trained to generate Wikipedia ar- ticles learned to reliably transliterate names between lan- guages. While GPT-1 (Radford et al., 2018) focused on pre-training as a transfer learning method to improve supervised fine-tuning, it also included an ablation study demonstrat- ing that the performance of four heuristic zero-shot transfer methods improved steadily over the course of pre-training, without any supervised adaption. This analysis served as the basis for GPT-2 (Radford et al., 2019) which focused exclu- sively on studying the task-learning capabilities of language models via zero-shot transfer.
    - > CLIP is pre-trained to predict if an image and a text snippet are paired together in its dataset. To perform zero-shot clas- sification, we reuse this capability. For each dataset, we use the names of all the classes in the dataset as the set of poten- tial text pairings and predict the most probable (image, text) pair according to CLIP. In a bit more detail, we first compute the feature embedding of the image and the feature embed- ding of the set of possible texts by their respective encoders. The cosine similarity of these embeddings is then calculated, scaled by a temperature parameter τ, and normalized into a probability distribution via a softmax. Note that this predic- tion layer is a multinomial logistic regression classifier with L2-normalized inputs, L2-normalized weights, no bias, and temperature scaling. When interpreted this way, the image encoder is the computer vision backbone which computes a feature representation for the image and the text encoder is a hypernetwork (Ha et al., 2016) which generates the weights of a linear classifier based on the text specifying the visual concepts that the classes represent.
      - Alternative interpreation of zero-shot classification: logistic regression where the inputs the image embeddings and the weights are the text embeddings, and the text encoder becomes a hypernetwork.
    - Section 3.1.4 on prompt engineering is interesting.
    - > While zero-shot CLIP generalizes well to many natural im- age distributions as investigated in Section 3.3, we’ve ob- served that zero-shot CLIP still generalizes poorly to data that is truly out-of-distribution for it. An illustrative exam- ple occurs for the task of OCR as reported in Appendix E. CLIP learns a high quality semantic OCR representation that performs well on digitally rendered text, which is common in its pre-training dataset, as evidenced by performance on Rendered SST2. However, CLIP only achieves 88% accu- racy on the handwritten digits of MNIST. An embarrassingly simple baseline of logistic regression on raw pixels outper- forms zero-shot CLIP. Both semantic and near-duplicate nearest-neighbor retrieval verify that there are almost no im- ages that resemble MNIST digits in our pre-training dataset. This suggests CLIP does little to address the underlying problem of brittle generalization of deep learning models. Instead CLIP tries to circumvent the problem and hopes that by training on such a large and varied dataset that all data will be effectively in-distribution. This is a naive assumption that, as MNIST demonstrates, is easy to violate.
      - "CLIP tries to circumvent the problem and hopes that by training on such a large and varied dataset that all data will be effectively in-distribution".

## [2023] MetaCLIP: Demystifying CLIP Data

- **Date**: 2025-07-23
- **Arxiv**: <https://arxiv.org/abs/2309.16671>
- **Paperpile**: <https://app.paperpile.com/view/?id=5df5730f-947e-4a5b-b3f1-dad04a0687ec>

---

- "We believe that the main ingredient to the success of CLIP is its data and not the model architecture or pre-training objective."
- MetaCLIP (Metadata-Curated Language-Image Pretraining) aims to reveal CLIP's data curation process.

## [2020] [ImageGPT: Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)

- **Date**: 2026-08-04

---

- **Question**: Does the GPT recipe transfer from language to vision? More precisely, can a model trained only to predict pixels learn semantic visual representations useful for classification?
- **Core result**: Yes, at sufficient scale. A decoder-only Transformer trained without labels on low-resolution images learns features that transfer well under linear probing, full fine-tuning, and low-data classification.
  - This supports the idea that learning a tractable model of $p(x)$ can produce useful features for learning $p(y\mid x)$.
  - cf. the modality-general compression/prediction argument in [[compression]].

- **Motivation and framing**:
  - > Unsupervised pre-training played a central role in the resurgence of deep learning. Starting in the mid-2000s, approaches such as the Deep Belief Network (Hinton et al., 2006) and Denoising Autoencoder (Vincent et al., 2008) were commonly used in neural networks for computer vision (Lee et al., 2009) and speech recognition (Mohamed et al., 2009). It was believed that a model which learned the data distribution $P(X)$ would also learn beneficial features for the subsequent supervised modeling of $P(Y\mid X)$ (Lasserre et al., 2006; Erhan et al., 2010). However, advancements such as piecewise linear activation functions (Nair & Hinton, 2010), improved initializations (Glorot & Bengio, 2010), and normalization strategies (Ioffe & Szegedy, 2015; Ba et al., 2016) removed the need for pre-training in order to achieve strong results.
  - > Other research cast doubt on the benefits of deep unsupervised representations and reported strong results using a single layer of learned features (Coates et al., 2011), or even random features (Huang et al., 2014; May et al., 2017). The approach fell out of favor as the state of the art increasingly relied on directly encoding prior structure into the model and utilizing abundant supervised data to directly learn representations (Krizhevsky et al., 2012; Graves & Jaitly, 2014). Retrospective study of unsupervised pre-training demonstrated that it could even hurt performance in modern settings (Paine et al., 2014).
  - > Instead, unsupervised pre-training flourished in a different domain. After initial strong results for word vectors (Mikolov et al., 2013), it pushed the state of the art forward in Natural Language Processing on most tasks (Dai & Le, 2015; Peters et al., 2018; Howard & Ruder, 2018; Radford et al., 2018; Devlin et al., 2018). Interestingly, the training objective of a dominant approach like BERT, the prediction of corrupted inputs, closely resembles that of the Denoising Autoencoder, which was originally developed for images.
  - > As a higher-dimensional, noisier, and more redundant modality than text, images are believed to be difficult for generative modeling. Here, self-supervised approaches designed to encourage the modeling of more global structure (Doersch et al., 2015) have shown significant promise. A combination of new training objectives (Oord et al., 2018), more recent architectures (Gomez et al., 2017), and increased model capacity (Kolesnikov et al., 2019) has allowed these methods to achieve state-of-the-art performance in low-data settings and sometimes even outperform supervised representations in transfer-learning settings.
  - > Given that it had been a decade since the original wave of generative pre-training methods for images, and considering their substantial impact in NLP, this class of methods was due for a modern re-examination and comparison with recent progress in self-supervised methods. We re-evaluate generative pre-training on images and demonstrate that, when using a flexible architecture (Vaswani et al., 2017), a tractable and efficient likelihood-based training objective (Larochelle & Murray, 2011; Oord et al., 2016), and significant compute resources (2,048 TPU cores), generative pre-training is competitive with other self-supervised approaches and learns representations that significantly improve the state of the art in low-resolution unsupervised representation-learning settings.
  - > This is especially promising, as our architecture uses a dense connectivity pattern which does not encode the 2D spatial structure of images, yet is able to match and even outperform approaches which do.
- **Approach**:
  - Resize each image to a low spatial resolution, flatten it in raster order, and treat it as a one-dimensional token sequence.
  - The architecture is essentially GPT-2: a decoder-only Transformer with learned positional embeddings and dense self-attention.
    - It receives no explicit knowledge of the image's two-dimensional structure.
    - The autoregressive ordering does introduce one weak spatial bias: pixels are encountered in raster order.
  - In the paper's notation, each pre-norm Transformer block is
    $$
    n^\ell=\operatorname{LayerNorm}(h^\ell),
    \qquad
    a^\ell=h^\ell+\operatorname{MultiHeadAttention}(n^\ell),
    \qquad
    h^{\ell+1}=a^\ell+\operatorname{MLP}\!\left(\operatorname{LayerNorm}(a^\ell)\right).
    $$
    - > Layer norms precede both the attention and MLP operations, and all operations lie strictly on residual paths. We find that such a formulation allows us to scale the Transformer with ease.
  - Two unsupervised objectives are compared:
    - **Autoregressive pixel prediction**:
      $$
      p(x)=\prod_{i=1}^{n}p(x_i\mid x_{<i};\theta),
      \qquad
      \mathcal{L}_{\mathrm{AR}}
      =\mathbb{E}_{x\sim\mathcal{X}}[-\log p(x)].
      $$
    - **BERT-style masked-pixel prediction**: independently mask $15\%$ of positions, then predict each masked pixel from the unmasked pixels:
      $$
      \mathcal{L}_{\mathrm{BERT}}
      =
      \mathbb{E}_{x\sim\mathcal{X}}
      \mathbb{E}_{M}
      \left[
        \sum_{i\in M}-\log p(x_i\mid x_{[1,n]\setminus M})
      \right].
      $$

- **Making pixel sequences tractable**:
  - Dense attention is quadratic in sequence length, so directly modeling a $224\times224$ RGB image is infeasible.
  - Images are resized to $32^2$, $48^2$, or $64^2$ pixels.
  - Rather than model the three RGB channel values as three tokens per pixel, the authors cluster RGB values into a learned 512-color palette. Each pixel then becomes one of 512 discrete tokens, shortening the sequence by $3\times$.
    - **Input resolution (IR)** describes the original RGB representation, such as $32^2\times3$.
    - **Model resolution (MR)** is the actual Transformer context length after palette quantization, such as $32^2$.
  - Models range from iGPT-S at 76M parameters to iGPT-XL at 6.8B parameters. The scale is central to the result rather than incidental.
- **Using the learned representation**:
  - > One way to measure representation quality is to fine-tune for image classification. Fine-tuning adds a small classification head to the model, which is used to optimize a classification objective, and adapts all weights. Pre-training can be viewed as a favorable initialization or as a regularizer when used in combination with early stopping (Erhan et al., 2010).
  - > Another approach for measuring representation quality uses the pre-trained model as a feature extractor. In particular, given labeled examples $(X,Y)$, the model is applied to $X$ to produce features $f_X$. A linear classifier is then trained on $(f_X,Y)$. Linear probing captures the intuition that good features should linearly separate the classes of transfer tasks. Furthermore, linear probes help disentangle feature quality from model architecture: in fine-tuning, one model may outperform another because its architecture is more suited for the downstream task rather than because of better pre-training.
  - In iGPT's implementation, **linear probing** freezes the model, average-pools sequence representations from a chosen Transformer layer, and trains only the linear classifier.
  - > In supervised pre-training, representation quality tends to increase monotonically with depth, such that the best representations lie at the penultimate layer (Zeiler & Fergus, 2014). Indeed, since a linear layer produces class logits from pre-logits, a good classifier necessarily achieves high accuracy on a linear probe of its pre-logits. If a downstream task also involves classification, it is empirically validated that penultimate features perform well. With generative pre-training, it is not obvious whether a task like pixel prediction is relevant to image classification. This suggests that the penultimate layer of a model trained for pixel prediction might not produce the most useful representations for classification. Latent-variable models such as VAEs can avoid this issue by explicitly learning a representation of the input data, but deep autoregressive generative models have the same width and connectivity pattern at every layer. Our first experiment studies how representation quality varies over one set of candidate representations: different layers of a generative model. We observe a very different behavior from supervised learning: representations first improve as a function of depth and then, starting around the middle layer, begin to deteriorate until the penultimate layer (Figure 2).
  - > This behavior potentially suggests that these generative models operate in two phases. In the first phase, each position gathers information from its surrounding context in order to build a more global image representation. In the second phase, this contextualized input is used to solve the conditional next-pixel prediction task. This could resemble the behavior of encoder-decoder architectures common across deep learning, but learned within a monolithic architecture via a pre-training objective.
    - Therefore, evaluating only the final layer understates representation quality; on CIFAR-10 it costs 2.4 percentage points.
  - **Fine-tuning** averages the final-layer sequence features and trains the entire model with a classification head.
    - Jointly retaining the generative loss, $\mathcal{L}_{\mathrm{GEN}}+\mathcal{L}_{\mathrm{CLF}}$, works better than classification loss alone.
- **Evaluation setup**:
  - > Although supervised pre-training is the dominant paradigm for image classification, curating large labeled image datasets is both expensive and time-consuming. Instead of further scaling up labeling efforts, we can aspire to learn general-purpose representations from the much larger set of available unlabeled images and fine-tune them for classification. We investigate this setting using ImageNet as a proxy for a large unlabeled corpus, and small classic labeled datasets (CIFAR-10, CIFAR-100, STL-10) as proxies for downstream tasks. For our largest model, we use an additional 100 million unlabeled web images, filtered to be similar to ImageNet. Even in cases where labels are available, unsupervised or self-supervised pre-training can still provide benefits in data efficiency or fine-tuning speed. We investigate this setting by pre-training without labels and then fine-tuning or linear probing with labels.

- **Results**:
  - **Linear probes with iGPT-L**:
    - CIFAR-10: **96.3%**.
    - CIFAR-100: **82.8%**.
    - STL-10: **95.5%**.
    - These results beat the paper's supervised-transfer and unsupervised-transfer baselines on the three low-resolution datasets, although iGPT is pretrained at a resolution closely matched to CIFAR.
  - **ImageNet linear probe**:
    - iGPT-XL reaches **68.7%** using the best single layer.
    - Concatenating five nearby layers gives **72.0%**, but uses a 15,360-dimensional feature vector.
    - This is competitive with contemporary self-supervised methods, not best-in-class: the paper reports SimCLR at 76.5% while using standard ImageNet resolution and far fewer parameters.
  - **Full fine-tuning with iGPT-L** reaches **99.0%** on CIFAR-10 and **88.5%** on CIFAR-100. On ImageNet, the $48^2$ model reaches **72.6%**, versus 53.2% when the same architecture is trained from scratch in the reported baseline.
  - Better autoregressive validation likelihood correlates with better linear-probe accuracy throughout training. Larger models also learn better representations, including when compared at the same generative loss.
  - In the low-data CIFAR-10 setting, fixed iGPT features plus logistic regression reach 73.2% with four labels per class and 87.6% with 25 labels per class, though specialized semi-supervised methods remain stronger.
- **Autoregressive vs. masked prediction**:
  - Autoregressive pretraining produces substantially better frozen features: the best BERT-style probe is more than one point worse on CIFAR-10 and about six points worse on ImageNet.
  - Full fine-tuning closes most or all of that gap. This distinguishes **representation quality accessible to a linear probe** from **usefulness as an initialization that can be adapted end to end**.
  - The masked model also has a train/test mismatch: because it sees corrupted inputs during pretraining, evaluation works better when predictions are ensembled over several random masks.
- **Takeaways and limitations**:
  - > Many self-supervised approaches focus on designing auxiliary objectives which support the learning of useful representations without attempting to directly model the input data. Examples include surrogate classification (Dosovitskiy et al., 2015), jigsaw-puzzle solving (Noroozi & Favaro, 2016), and rotation prediction (Gidaris et al., 2018). A cluster of similar approaches based on contrastive losses comparing various views and transformations of input images have recently driven significant progress in self-supervised learning (Hjelm et al., 2018; Bachman et al., 2019; Tian et al., 2019). Among contrastive approaches, our work is most similar to Contrastive Predictive Coding (Oord et al., 2018), which also utilizes an autoregressive prediction objective, but in a learned latent space, and to Selfie (Trinh et al., 2019), which trains a bidirectional self-attention architecture on top of a standard convolutional network to differentiate correct from incorrect patches.
  - **Main conceptual contribution**: next-token prediction is not inherently linguistic. Even raw pixel prediction can force a sufficiently large sequence model to learn semantic features.
  - **Generative quality and representation quality align empirically** in this setup: improved likelihood tracks improved classification features.
  - **Weak inductive bias can be overcome by scale**, but not efficiently. iGPT-L uses roughly 2-3 times as many parameters as similarly performing ImageNet models, and training the largest model required enormous compute.
  - Low input resolution and dense attention are fundamental constraints. Later image generators address this by modeling compressed latent or discrete visual tokens instead of every raw pixel; see VQGAN, RQ-Transformer, and MaskGIT below.
  - Relation to ViT: both apply Transformers to images, but iGPT is a decoder-only generative model over rasterized pixels, whereas ViT is an encoder over image patches trained for recognition.
  - > Finally, our results, considered together with Donahue & Simonyan (2019), suggest revisiting the representation-learning capabilities of other families of generative models, such as flows (Dinh et al., 2014; Kingma & Dhariwal, 2018) and VAEs, in order to study whether they show similarly competitive representation-learning capabilities.

## [2022] RQ-VAE & RQ-Transformer: Autoregressive Image Generation using Residual Quantization

**Date**: 2025-09-11
**Arxiv**: <https://arxiv.org/abs/2203.01941>
**Paperpile**: <https://app.paperpile.com/view/?id=ae51403e-e299-414c-9fd2-46f6fa0272ea>

---

- **Abstract**:
  - > For autoregressive (AR) modeling of high-resolution images, vector quantization (VQ) represents an image as a sequence of discrete codes. A short sequence length is important for an AR model to reduce its computational costs to consider long-range interactions of codes. However, we postulate that previous VQ cannot shorten the code sequence and generate high-fidelity images together in terms of the rate-distortion trade-off. In this study, we propose the two-stage framework, which consists of Residual-Quantized VAE (RQ-VAE) and RQ-Transformer, to effectively generate high-resolution images. Given a fixed codebook size, RQ-VAE can precisely approximate a feature map of an image and represent the image as a stacked map of discrete codes. Then, RQ-Transformer learns to predict the quantized feature vector at the next position by predicting the next stack of codes. Thanks to the precise approximation of RQ-VAE, we can represent a 256× 256 image as 8× 8 resolution of the feature map, and RQ-Transformer can efficiently reduce the computational costs. Consequently, our framework outperforms the existing AR models on various benchmarks of unconditional and conditional image generation. Our approach also has a significantly faster sampling speed than previous AR models to generate high-quality images.
- **Intro**:
  - Vector Quantization (VQ) is fundamental for enabling autoregressive (AR) models to generate high resolution images.
    - VQ takes an image and outputs a sequence of discrete codes/tokens, which are flattened in raster-scan order and used to train an autoregressive next-token-prediction model.
  - **Trade-off in terms of sequence length of discrete codes/tokens for AR image-generation models**:
    - Long sequence length -> computational inefficiency of AR model.
    - Short sequence length -> **rate-distortion trade-off**.
      - > VQ-VAE requires an exponentially increasing size of codebook to reduce the resolution of the quantized feature map, while conserving the quality of reconstructed images. However, a huge codebook leads to the increase of model parameters and the codebook collapse problem, which makes the training of VQ-VAE unstable.
  - **Contributions**:
    - **(1) RQ-VAE**: VQ-VAE with Residual VQ instead of standard VQ; helps reduce token sequence length without compromising reconstruction quality or needing to exponentially increase the codebook size.
    - **(2) RQ-Transformer**: Autoregressive next-token prediction adapted to RVQ codes (predict tokens corresponding to all RVQ levels at once per timestep).
- **Methods**:
  - **(1) RQ-VAE (Residual Quantized VAE)**:
    - Replaces VQ in VQ-VAE with RVQ (Residual Vector Quantizer). See SoundStream [[papers-speech]].
    - **Helps reduce spatial resolution of the vector-quantized feature map** for AR modeling without needing to exponentially grow the codebook size to combat loss in approximation.
    - In this work, a single shared codebook is shared across all $Q$ levels of RVQ instead of one codebook per level.
    - **RQ-VAE Loss $L_{RQ-VAE} = L_{reconstruction} + \alpha L_{commitment}$**
      - **(a) Reconstruction loss** $L_{reconstruction} = \lVert X - \hat{X} \rVert^2_2$, where $X$ is the original image and $\hat{X}$ is the reconstructed image.
      - **(b) Commitment loss** $L_{commitment} = \sum_{q=1}^{Q} \lVert Z - \hat{Z}_q.\text{detach()} \rVert^2_2$, where $Z$ is the latent representation before quantization and $\hat{Z}_q$ is the reconstructed latent representation corresponding to RVQ level $q$.
        - cf. commitment loss in vq-wav2vec [[papers-speech]] for a slight difference in formulation.
        - > Note that $L_{commitment}$ is the sum of quantization errors from every $q$, not a single term $\lVert Z - \hat{Z}\rVert^2_2$. It aims to make $\hat{Z}_q$ sequentially decrease the quantization error of $Z$ as $q$ increases. Thus, RQ-VAE approximates the feature map in a coarse-to-fine manner and keeps the training stable.
        - $\hat{Z}_q.\text{detach()}$ is the straight-through estimator (STE) application to bypass the non-differentiable codebook lookup step.
        - **Codebook entries are updated via EMA** (exponential moving average) as in SoundStream [[papers-speech]] and <https://github.com/vishar0/emg-tokenizer> (`vector_quantizer.VectorQuantizer._update_codebook`).
  - **(2) RQ-Transformer** (Fig 2):
    - **Input**: Discrete codes/tokens from RQ-VAE of shape $H \times W \times Q$.
    - **Naive approach**: Autoregressive modeling over 1D flattened sequence of codes. **$H \times W \times $Q$ autoregressive steps**.
    - **RQ-Transformer**: Autoregressive modeling factorized across spatial ($H \times W$) and depth ($Q$) dimensions. **$H \times W + Q$ autoregressive steps** with the depth transformer executing in parallel over all spatial points.
      - **(a) Spatial Transformer**: Sum the quantized latent representations for all quantizer levels $q \in \{1,...Q\}$ per spatial timestep $t \in \{1,...,H \times W\}$, feed into a transformer that autoregressively outputs a latent per timestep $t$ to be fed into the depth transformer. **$H \times W$ autoregressive steps**.
      - **(b) Depth Transformer**: In parallel over all spatial timesteps $t \in \{1,...,H \times W\}$, take the output latent from spatial transformer corresponding to that timestep and autoregressively predict output codes for each depth (quantization level) $q \in \{1,...Q\}$. **$Q$ parallel autoregressive steps**.
    - The **reduced spatial resolution of RVQ feature map codes improves computational efficiency and helps learn long-range interactions**.
    - **Strategies to mitigate exposure bias**:
      - > Exposure bias is known to deteriorate the perfor mance of an AR model due to the error accumulation from the discrepancy of predictions in training  and inference. During an inference of RQ-Transformer, the prediction errors can also accumulate along with the quantization level/depth $Q$, since finer estimation of the feature vector becomes harder as $q$ increases.
      - **(i) Soft Labeling of Target Codes (Training-time)**: Instead of treating the chosen codebook entry as a hard one-hot target for RQ-Transformer training, use a soft label distribution over the nearest few codebook entries, but annealed with a temperature paper so that this approaches the hard one-hot target as training progresses. Fixes training stability and encourages robust codebook usage.
      - **(ii) Stochastic Sampling for Codes of RQ-VAE (Inference-time)**: Similarly, temperature sampling codes to feed into RQ-Transformer during inference. Improves inference diversity and prevents collapse to dull generations.

## [2022] MaskGIT: Masked Generative Image Transformer

- **Date**: 2025-05-09
- **Arxiv**: <https://arxiv.org/abs/2202.04200>
- **Paperpile**: <https://app.paperpile.com/view/?id=48910966-4885-4350-a09d-52e3e767c136>

---

- **Abstract**:
  - > Generative transformers have experienced rapid popularity growth in the computer vision community in synthesizing high-fidelity and high-resolution images. The best generative transformer models so far, however, still treat an image naively as a sequence of tokens, and decode an image sequentially following the raster scan ordering (i.e., line-by-line). We find this strategy neither optimal nor efficient. This paper proposes a novel image synthesis paradigm using a bidirectional transformer decoder, which we term MaskGIT. During training, MaskGIT learns to predict randomly masked tokens by attending to tokens in all directions. At inference time, the model begins with generating all tokens of an image simultaneously, and then refines the image iteratively conditioned on the previous generation. Our experiments demonstrate that MaskGIT significantly outperforms the state-of-the-art transformer model on the ImageNet dataset, and accelerates autoregressive decoding by up to 64x. Besides, we illustrate that MaskGIT can be easily extended to various image editing tasks, such as inpainting, extrapolation, and image manipulation.
- **Intro**:
  - Inspired by the successs of autoregressive models (transformer, GPT) in NLP, **generative transformer models have received growing interests in image synthesis**.
    - Generally, **autoregressive modeling for image generation is done in two stages**:
      - **Stage 1**: Vector quantize an image into a sequence of discrete tokens.
      - **Stage 2**: Train a transformer to generate the discrete tokens sequentially and autoregressively based on the previously generated tokens.
    - Stage 1 gets most of the focus while Stage 2 is a drop in replacement from NLP.
    - For Stage 2 (autoregressive modeling of discrete image tokens), **even SoTA methods treat an image naively as a flattened 1D sequence of tokens from left to right line-by-line (raster scan order)**.
      - Neither optimal nor efficient. Unlike text, images are not sequential.
  - **Masked Generative Image Transformer (MaskGIT)**:
    - **Bidirectional transformer for image synthesis**.
    - **Training (Fig 3)**: Similar to mask prediction in BERT.
    - **Inference (Fig 2)**: A novel non-autoregressive decoding method to synthesize an image in constant number of steps. At each step, all tokens are predicted in parallel but only the most confident ones are kept for the next autoregressive step, with the remaining token masked out. The mask ratio is decreased until all tokens are generated with a few steps of refinement.
      - Predicitons within each step are parallelizable.
      - Order of magnitude faster decoding.
      - For 32x32 image tokens, 8 steps with MaskGIT instead of 256 steps with raster scan order autoregressive decoding.
      - Mask ratio scheduling (i.e., fraction of tokens masked at each step) significantly affects generation quality. Propose to use cosine schedule.
    - > MaskGIT’s multidirectional nature makes it readily extendable to image manipulation tasks that are otherwise difficult for autoregressive models. Fig 1 shows a new application of class-conditional image  editing in which MaskGIT regenerates content inside the bounding box based on the given class while keeping the context (outside of the box) unchanged. This task, which is either infeasible for autoregressive model or difficult for GAN models, is trivial for our model.
- **Method**:
  - **Training (Fig 3)**:
    - Stage 1 (image tokenization) uses the same setup as in VQGAN.
    - Stage 2 (autoregressive modeling) learns a **bidirectional transformer with Masked Visual Token Modeling (MVTM)**.
      - **(i) Tokenize**: Obtain discrete tokens by feeding the image to a VQ-encoder such as in VQGAN.
      - **(ii) Mask**: Sample a mask ratio $\gamma$ from 0 to 1, and uniformly select as many tokens to replace with `[MASK]`.
      - **(iii) Model**: Feed through a bi-directional tranformer to optimize with negative log-likelihood loss corresponding to the masked tokens.
  - **Iterative Decoding (Fig 2)**: Start with a blank canvas with all the tokens masked out. Loop for $T$ steps:
    - **(i) Predict**: Model inference to predict output token probabilities corresponding to masked positions.
    - **(ii) Sample**: At each masked position in the current step, sample an output token based on the predicted probabilities. The prediciton probability corresponding to the sampled output token is used as a confidence score, with the unmasked tokens in the current step receiving a confidence score of 1.0.
    - **(iii) Mask Schedule**: Using a mask scheduling function $\gamma(r)$ with $r \in [0,1)$, compute the number of tokens to mask at the current step: $n = \lceil \gamma(\frac{t}{T})N \rceil$, where $t$ is the current decoding step count, $T$ is the total number of decoding steps, and $N$ is the total number of tokens.
    - **(iv) Mask**: Mask $n$ of the least confident tokens according to the confidence score computed in (2).
  - **Masking Design**: Significantly affects the quality of image generation.
    - **Mask scheduling function** $\gamma(r)$ that computes the token mask ratio given an input $r \in [0,1]$.
      - Inference: $r = t/T$, where $T$ is the total number of decoding steps and $t \in \{0, 1, 2, ..., T-1\}$ is the current decoding step.
      - Training: $r$ is randomly sampled from $[0,1)$ to simulate various decoding scenarios.
    - **Properties of mask scheduling function**:
      - $\gamma(r)$ must be a continuous monotonically descreasing function wrt $r \in [0,1]$.
      - $\gamma(0) \to 1$ (all tokens masked out at decoding step $t=0$).
      - $\gamma(1) \to 0$ (all tokens unmasked at decoding step $t=T$).
    - **Choices of mask scheduling function (Fig 8)**:
      - Linear function
      - Concave function (less to more tokens unmasked per step) - cosine, square, cubic, exponential
      - Convex function (more to less tokens unmasked per step) - square root, logarithmic
  - **Experiments**:
    - **Metrics to measure image generation quality**: <https://chatgpt.com/share/68bce46c-3050-8005-906e-d4374a78d582>
      - **Frechet Inception Distance (FID)**
      - **Inception Score (IS)**
    - Outperforms VQGAN in quality (owing to bidirectional nature), speed (parallelism of decoding), and versatility (extends to image inpainting/outpaintaing/editing beyond image synthesis).

## [2021] ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- **Date**: 2026-02-10
- **Arxiv**: <https://arxiv.org/abs/2010.11929>
- **Paperpile**: <https://app.paperpile.com/view/?id=de534f6a-4352-4d95-a867-9a041d8ce7f9>

---

- **Abstract**:
  - > While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited.  In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place.  We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.  When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring sub- stantially fewer computational resources to train.
- **Intro**:
  - > Thanks to Transformers’ computational efficiency and scalability, it has become possible to train models of unprecedented size, with over 100B parameters
  - > Inspired  by  the  Transformer  scaling  successes  in  NLP,  we  experiment  with  applying  a  standard Transformer directly to images, with the fewest possible modifications. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Trans- former. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.
  - > When trained on mid-sized datasets such as ImageNet without strong regularization,  these mod- els yield modest accuracies of a few percentage points below ResNets of comparable size.  **This seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data**.
  - > However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that **large scale training trumps inductive bias**.  Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints.
- **[Fig1] Method**:
  - > In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible. An advantage of this intentionally simple setup is that scalable NLP Transformer architectures – and their efficient implementations – can be used almost out of the box.
  - > **Inductive bias.** We note that **Vision Transformer has much less image-specific inductive bias than CNNs**. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model.  In ViT, only MLP layers are local and transla- tionally equivariant, while the self-attention layers are global.  The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution (as de- scribed below). Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.
  - > Few-shot accuracies are obtained by solving a **regularized least-squares regression problem** that maps the (frozen) representation of a subset of training images to ${−1, 1}^K$ target vectors. This formulation allows us to recover the exact solution in closed form. Though we mainly focus on fine-tuning performance, we sometimes use linear few-shot accuracies for fast on-the-fly evaluation where fine-tuning would be too costly.

## [2021] DINO: Emerging Properties in Self-Supervised Vision Transformers

- **Date**: 2026-02-10
- **Arxiv**: <https://arxiv.org/abs/2104.14294>
- **Paperpile**: <https://app.paperpile.com/view/?id=49b41748-5ef1-4a15-9105-68d688d292db>

---

- **Abstract**:
  - > In this paper, we question if self-supervised learning pro- vides new properties to Vision Transformer (ViT) [19] that stand out compared to convolutional networks (convnets). Beyond the fact that adapting self-supervised methods to this architecture works particularly well, we make the follow- ing observations: first, self-supervised ViT features contain explicit information about the semantic segmentation of an image, which does not emerge as clearly with supervised ViTs, nor with convnets. Second, these features are also ex- cellent k-NN classifiers, reaching 78.3% top-1 on ImageNet with a small ViT. Our study also underlines the importance of momentum encoder [33], multi-crop training [10], and the use of small patches with ViTs. We implement our findings into a simple self-supervised method, called DINO, which we interpret as a form of self-distillation with no labels. We show the synergy between DINO and ViTs by achieving 80.1% top-1 on ImageNet in linear evaluation with ViT-Base.
- **Intro**:
  - > we study the impact of **self-supervised pre-training on ViT features**.
  - > Of particular interest, **we have identified several interesting properties that do not emerge with supervised ViTs, nor with convnets**:
    - > Self-supervised  ViT  features  explicitly  contain  the scene layout and, in particular, object boundaries, as shown in Figure 1. This information is directly accessible in the self-attention modules of the last block.
    - > Self-supervised ViT features perform particularly well with a basic nearest neighbors classifier (k-NN) without any finetuning, linear classifier nor data augmentation, achieving 78.3% top-1 accuracy on ImageNet.
    - > The emergence of segmentation masks seems to be a property shared across self-supervised methods. However, the good performance with k-NN only emerge when com- bining certain components such as momentum encoder [33] and multi-crop augmentation [10].
  - [Fig2] DINO: Self-distillation with no labels
    - Simplifies SSL training by directly predicting the output of a teacher network, built with a momentum encoder, by using a standard cross-entropy loss.
    - > Training DINO with ViT takes just **two 8-GPU servers over 3 days** to achieve 76.1% on ImageNet linear benchmark,  which outperforms self-supervised systems based on convnets of comparable sizes with significantly reduced compute require- ments
- **[Fig2] Method**:
  - (1) The model passes two different random transformations of an input image to the student and teacher networks. Both networks have the same architecture but different parameters.
  - (2) The output of the teacher network is centered with a mean computed over the batch.
  - (3) Each networks outputs a K dimensional feature that is normalized with a temperature softmax over the feature dimension.
  - (4) Their similarity is then measured with a cross-entropy loss.
  - (5) We apply a stop-gradient (sg) operator on the teacher to propagate gradients only through the student.
  - (6) The teacher parameters are updated with an exponential moving average (ema) of the student parameters.
- TODO
