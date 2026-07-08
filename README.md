# 🧠 Glassbox-MNIST: Interpretable MNIST from Scratch

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Float32-deepgreen.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visuals-orange.svg)

> A lightweight, fully connected neural network built entirely from scratch in NumPy, featuring Vanilla Saliency Maps to visualize *exactly* what the network is "looking at" when classifying digits.

<p align="center">
  <img width="1200" height="400" alt="saliency_sample_11" src="https://github.com/user-attachments/assets/fc494ac1-66d6-4590-845c-17e54c7e6278" />
</p>

## 🚀 About The Project

Most MNIST tutorials stop at accuracy. This project goes a step further into **Explainable AI (XAI)**. 

Built completely without deep learning frameworks (no PyTorch, no TensorFlow), this project implements forward propagation, backpropagation, and categorical cross-entropy from scratch using optimized `float32` matrix operations. Furthermore, it implements **"Reverse Backpropagation"** to generate Saliency Maps—highlighting the exact pixels that drove the model's final prediction.

### ✨ Key Features
* **100% Framework-Free:** Core math and gradient descent implemented purely in NumPy.
* **Explainable AI:** Generates heatmaps indicating pixel-level feature importance for any given prediction.
* **Memory Optimized:** Uses `float32` casting to halve the memory footprint and leverage hardware acceleration.
* **Math-to-Code Mapping:** Clean, documented code that maps directly to the underlying calculus and linear algebra.

## 🛠️ Tech Stack
* **NumPy:** Matrix multiplication and array manipulation.
* **Pandas:** Data loading and CSV parsing.
* **Matplotlib:** Image rendering and heatmap overlays.

## 💻 How to Run

### 1. Clone the repository
```bash
git clone [https://github.com/nethul/Glassbox-MNIST](https://github.com/nethul/Glassbox-MNIST)
cd Glassbox-MNIST



