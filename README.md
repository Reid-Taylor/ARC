# README

## Key Papers Inspiring This Project

### GradNorm: Gradient Normalization for Adaptive Loss Balancing

GradNorm introduces a method for dynamically balancing multiple loss functions in multi-task learning. By normalizing gradient magnitudes, it ensures that all tasks contribute meaningfully to the training process. This paper inspires our approach to balancing competing objectives in our model, ensuring robust and adaptive learning.

### Gradient Surgery: PCGrad

After implementing GradNorm and experimenting with the multi-task learning objective provided by this approach, we observed many failings to converge upon an ideal solution. After a pointed literature review aimed at addressing this, we discovered PCGrad and gradient surgery as an approach to better balance gradient descent by removing gradient conflicts, while still normalizing the gradient magnitude. Throughout the examples and benchmarks provided within the literature, as well as empirically through my own experiments here, performance of PCGrad is superior to GradNorm, for multi-task gradient balancing.

### SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

SimCLR demonstrates the power of self-supervised learning through contrastive methods. By maximizing agreement between augmented views of the same data, it achieves state-of-the-art performance in representation learning. This work informs our use of contrastive techniques to learn meaningful representations without reliance on labeled data; additionally, this work demonstrates the value of composed augmentations, as opposed to singular augmentations to samples.

### BYOL: Bootstrap Your Own Latent Space

BYOL presents a novel self-supervised learning framework that avoids the need for negative samples. By leveraging a target network and a predictor, it achieves strong performance through iterative bootstrapping. BYOL also demonstrates methods to parameterize the domain of application using residual adapter modules.

### MT-SLVR

This paper presents a novel method to learn simultaneously task-sensitive and task-invariant embeddings for different tasks, as prescribed by the user. Via specialized architecture and opt-in loss functions, we can choose whether the encoder should learn embeddings capable of determining if an augmentation has occurred, or we can train the model to be 'blind' to that augmentation.

Note: We have removed task-invariant losses from the implementation currently, since we are augmenting tasks with many possible augmentations--some deigned to be task-sensitive, while others are task-invariant augmentations.

## Mission of This Work

The mission of this project is to experiment with new approaches to deep learning by integrating principles from adaptive loss balancing, contrastive learning, supervised learning, and self-supervised learning. By drawing on the strengths of GradNorm, SimCLR, and BYOL, we aim to develop models that are robust, efficient, and capable of learning from limited or unlabeled data.
