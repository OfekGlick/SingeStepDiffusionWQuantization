# Quantization of Single-Step Diffusion Models

This project explores the application of post-training quantization (PTQ) to single-step diffusion models to improve computational efficiency and reduce inference time. Diffusion models, while powerful, typically require iterative denoising steps, making them computationally expensive. This work investigates whether PTQ can help speed up the process without significantly compromising the model's performance.

## Project Description

Diffusion models are a type of generative model that iteratively reverse the process of adding noise to data. However, their iterative nature results in high computational costs.
This project aims to apply quantization techniques to reduce the precision of the model's weights and activations, potentially lowering the inference time while maintaining reasonable performance.

The project focuses on quantizing a pre-trained single-step diffusion model and evaluating the results using various bit-widths (2, 4, 8 bits) on an ImageNet 64x64 dataset.

## Goal

The main goal is to assess the feasibility and effectiveness of using post-training quantization on single-step diffusion models, specifically:

1. Reducing inference time by lowering the model's precision.
2. Evaluating the impact of different quantization levels (2, 4, 8 bits) on the model's performance, as measured by the Fréchet Inception Distance (FID) score.

## Results

The experiments showed that:

- **8-bit and 4-bit quantization** provided significant computational benefits with only a small increase in the FID score (around 1.5-2 points).
- **2-bit quantization** resulted in a substantial drop in performance, with the FID score exceeding 100.
- **Increasing the number of sampling steps** did not always improve performance, indicating that adding steps may not be beneficial under quantization.
- **Tailored quantization frameworks** or techniques like Quantization Aware Training (QAT) often perform better than generic approaches.

These findings suggest that quantization, particularly at 8-bit and 4-bit precision, can be an effective tool for speeding up diffusion models with minimal loss in performance.


## Acknowledgments

This repository borrows some of its code from the [BRECQ](https://github.com/yhhhli/BRECQ) repository
