# Physics Guided Machine Learning on Real Lensing Images
This repository contains the evaluation tasks for the project proposal of the project named "Physics Guided Machine Learning on Real Lensing Images" under ML4Sci for GSoC 2025.

## Overview

This tasks tackle two primary objectives:

1. **Common Task:** Classify strong gravitational lensing images into substructure classes using conventional CNNs.  
2. **Specific Task:** Develop Physics-Informed Neural Networks (PINNs) that incorporate the underlying gravitational lensing physics into the learning process..

## Repository Structure

We explored four main approaches to integrate physics into the network training:

- **Laplacian Approach:** An auxiliary branch predicts the gravitational potential, and a Laplacian loss is enforced over it.
- **Poisson Approach:** The network enforces the Poisson equation by comparing the Laplacian of the predicted potential with a scaled version of the source image.
- **Fourier Approach:** A Fourier-based loss compares the frequency domain representation of the predicted distortion field to that of the input image.
- **Autoencoder Approach:** A standard autoencoder is trained with physics-informed losses to levarage the resultant latent space to learn a classifier.

---

## Gravitational Lensing Background

Gravitational lensing occurs when the mass distribution of a foreground object bends the light from a background source. This process is described by the lens equation:

$$
\vec{\theta} = \vec{\beta} + \nabla \psi(\vec{\theta})
$$

where:
- theta is the observed image position,
- beta is the true (unlensed) source position,
- psi(theta) is the lensing potential, and
- nabla(psi(theta))) is the deflection angle.

Under the thin-lens approximation, the lensing potential psi satisfies the Poisson equation:

$$
\nabla^2 \psi = 2 \kappa
$$

where kappa (convergence) denotes the projected mass density. These equations form the basis for our physics-informed neural network approaches.

---

## Approach 1: Auxiliary Branch with Laplacian Regularization

**Concept:**  
An auxiliary branch is appended to a CNN to predict the gravitational potential psi. The predicted potential is then regularized via a Laplacian loss that enforces a smooth (harmonic) behavior, as expected in regions free of mass substructure.

**Mathematical Formulation:**

The physical expectation in homogeneous regions is:

$$
\nabla^2 \psi = 0.
$$

Accordingly, the loss term is defined as:

$$
L_{\text{Laplacian}} = \left\| \nabla^2 \psi \right\|^2.
$$

This Laplacian loss is added to the standard classification loss, guiding the network to learn both discriminative features and physically consistent potential representations.

---

## Approach 2: CNNs with Poisson Equation Enforcement

**Concept:**  
In this approach, the network is trained to enforce the Poisson equation. Instead of solely relying on smoothness, we enforce that the Laplacian of the predicted potential psi approximates twice the source image intensity I_source (where I_source is obtained by converting the input image to grayscale).

**Mathematical Formulation:**

$$
\nabla^2 \psi = 2 \, I_{\text{source}}
$$

The corresponding loss is defined as:

$$
L_{\text{Poisson}} = \left\| \nabla^2 \psi - 2\,I_{\text{source}} \right\|^2.
$$

This loss term drives the network to generate potentials that conform to the expected mass distribution derived from the source intensity.

---

## Approach 3: CNNs with Fourier-Based Loss Enforcement

**Concept:**  
This method leverages Fourier analysis to enforce physical constraints in the frequency domain. An auxiliary branch predicts a distortion field (e.g., related to the gravitational potential). The Fourier transform of the predicted field is compared to the Fourier transform of the grayscale input image I_gray.

**Mathematical Formulation:**

Let \( \mathcal{F}(\cdot) \) denote the Fourier transform. Then, the loss is defined as:

$$
L_{\text{Fourier}} = \left\| \left| \mathcal{F}(\text{predicted}) \right| - \left| \mathcal{F}(I_{\text{gray}}) \right| \right\|^2.
$$

This constraint ensures that the frequency components of the predicted distortion field match those expected from the physical processes of gravitational lensing.

---

## Approach 4: Autoencoder-Based Approaches with Physics Constraints

**Concept:**  
A standard autoencoder is trained to reconstruct gravitational lensing images while learning a compact latent representation \( z \). In addition to the usual losses, physics-informed loss terms are applied to the autoencoder:

1. **Reconstruction Loss:**  
   Ensures the input image \( I \) is accurately reconstructed:

$$
L_{\text{recon}} = \| I - \hat{I} \|^2.
$$
   
3. **Classification Loss:**  
   For a classifier built on the latent space:

$$
L_{\text{cls}} = -\sum_{i} y_i \log(\hat{y}_i).
$$
   
5. **PDE Loss (Smoothness Constraint):**  
   Enforces the reconstruction to be smooth by penalizing the Laplacian:

$$
L_{\text{PDE}} = \left\| \nabla^2 \hat{I} \right\|^2.
$$
   
7. **Energy Constraint Loss (Latent Regularization):**  
   Regularizes the latent space to have zero mean and unit variance:
   
$$
L_{\text{Energy}} = \| \mu_z \|^2 + \| \sigma_z - 1 \|^2,
$$

   
   where mu and sigma denote the mean and standard deviation of the latent vectors.

The overall loss for this approach is a weighted sum:

$$
L = L_{\text{cls}} + \lambda_{\text{recon}} \, L_{\text{recon}} + \lambda_{\text{PDE}} \, L_{\text{PDE}} + \lambda_{\text{Energy}} \, L_{\text{Energy}}.
$$

This physics-informed regularization guides the autoencoder to learn a latent space that is both discriminative and physically consistent.

---

## Benchmarking Results

Benchmark results (measured in terms of Accuracy and One-Versus-Rest AUC) were gathered using multiple pretrained backbones, including:

- ResNet18  
- EfficientNet\_b0  
- EfficientNet\_b1  
- MobileNetV2  
- DenseNet121  

### Auxiliary Branch with Laplacian Regularization
| Backbone         | Accuracy | OVR AUC |
|------------------|----------|---------|
| ResNet18         | TBD      | TBD     |
| EfficientNet_b0  | TBD      | TBD     |
| EfficientNet_b1  | TBD      | TBD     |
| MobileNetV2      | TBD      | TBD     |
| DenseNet121      | TBD      | TBD     |

### CNNs with Poisson Equation Enforcement
| Backbone         | Accuracy | OVR AUC |
|------------------|----------|---------|
| ResNet18         | TBD      | TBD     |
| EfficientNet_b0  | TBD      | TBD     |
| EfficientNet_b1  | TBD      | TBD     |
| MobileNetV2      | TBD      | TBD     |
| DenseNet121      | TBD      | TBD     |

### CNNs with Fourier-Based Loss Enforcement
| Backbone         | Accuracy | OVR AUC |
|------------------|----------|---------|
| ResNet18         | TBD      | TBD     |
| EfficientNet_b0  | TBD      | TBD     |
| EfficientNet_b1  | TBD      | TBD     |
| MobileNetV2      | TBD      | TBD     |
| DenseNet121      | TBD      | TBD     |

### Autoencoder-Based Approaches with Physics Constraints
| Backbone         | Accuracy | OVR AUC |
|------------------|----------|---------|
| ResNet18         | TBD      | TBD     |
| EfficientNet_b0  | TBD      | TBD     |
| EfficientNet_b1  | TBD      | TBD     |
| MobileNetV2      | TBD      | TBD     |
| DenseNet121      | TBD      | TBD     |
