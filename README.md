# README

## Mission of This Work

The mission of this project is to experiment with new approaches to deep learning by integrating principles from adaptive loss balancing, contrastive learning, supervised learning, and self-supervised learning. By drawing from many published learnings around transformers and visual comprehension tasks, we seek to learn an embedding representation of a nullable grid of discrete states, the ARC grid, and to represent the transformation pattern from many example inputs to their respective outputs in order to "predict" the solution to a given challenge grid which follows the pattern demonstrated in the sparse examples.

## Key Papers Inspiring this Work

### AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

**Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (2020)

This paper (ViT) demonstrates that a pure Transformer applied directly to sequences of image patches can perform very well on image classification, removing the reliance on CNNs. By splitting an image into fixed-size 16x16 patches and treating them as token sequences, the Vision Transformer attains excellent results on benchmarks like ImageNet and CIFAR-100 when pre-trained on large datasets, while requiring substantially fewer computational resources to train than comparable convolutional networks.

### GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks

**Authors:** Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich (2018)

GradNorm presents a gradient normalization algorithm that automatically balances training in deep multitask models by dynamically tuning gradient magnitudes. It improves accuracy and reduces overfitting across multiple tasks compared to single-task networks and static baselines, matching or surpassing exhaustive grid search methods while relying on only a single hyperparameter (α). This replaces what was previously a tedious, exponentially scaling search process with a few training runs regardless of the number of tasks.

### Gradient Surgery for Multi-Task Learning

**Authors:** Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn (2020)

This paper identifies three conditions in multi-task optimization that cause detrimental gradient interference and proposes a form of "gradient surgery" — projecting a task's gradient onto the normal plane of any other task's conflicting gradient. This model-agnostic approach leads to substantial gains in efficiency and performance across challenging multi-task supervised learning and reinforcement learning problems, and can be combined with existing multi-task architectures.

### Counting and Algorithmic Generalization with Transformers (arXiv:2310.08661)

**Authors:** Simon Ouellette, Rolf Pfister, Hansueli Jud (2023)

This paper analyzes algorithmic generalization in Transformers when counting is required, either implicitly or explicitly. It shows that standard Transformer architectural choices — specifically layer normalization and softmax-normalized attention weights — hinder out-of-distribution performance on counting tasks. By ablating these problematic operations, the authors demonstrate that a modified, lightweight Transformer can achieve strong algorithmic generalization on counting.

### DINOv2: Learning Robust Visual Features without Supervision

**Authors:** Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski (2023)

DINOv2 shows that self-supervised pretraining methods can produce all-purpose visual features — features that work across image distributions and tasks without finetuning — when trained on enough curated data from diverse sources. The work contributes an automatic pipeline for building a dedicated, diverse image dataset and trains a ViT model with 1B parameters, then distills it into smaller models that surpass the best available all-purpose features (OpenCLIP) on most benchmarks at both image and pixel levels.

### Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning

**Authors:** Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko (2020)

BYOL introduces a self-supervised image representation learning approach that relies on two neural networks (online and target) that learn from each other. The online network is trained to predict the target network's representation of the same image under a different augmented view, while the target network is updated as a slow-moving average of the online network. Unlike prior state-of-the-art methods, BYOL achieves competitive results without requiring negative pairs, reaching 74.3% top-1 accuracy on ImageNet with a ResNet-50.

### A Simple Framework for Contrastive Learning of Visual Representations

**Authors:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton (2020)

SimCLR presents a simple framework for contrastive self-supervised learning that requires no specialized architectures or memory banks. Key findings include: (1) composition of data augmentations is critical for defining effective predictive tasks, (2) a learnable nonlinear projection head between the representation and the contrastive loss substantially improves learned representations, and (3) contrastive learning benefits from larger batch sizes and longer training. SimCLR achieves 76.5% top-1 accuracy on ImageNet with a linear classifier, matching supervised ResNet-50 performance.
