# Evaluating Cell-Based Neural Architectures on Embedded Systems
This repository contains the PyTorch implementation of the thesis **Evaluating cell-based neural architectures on embedded systems**

By Ilja van Ipenburg.
The thesis is also included in the repository.

# Abstract
Neural Architectures Search (NAS) methodologies used to discover state-of-the-art neural networks without human intervention, have seen a growing interest in recent years. One subgroup of NAS methodologies consists of cell-based neural architectures, which are architectures made up of convolutional cells, which are repeatedly stacked on top of each other to form a complete neural network. The way in which these cells can be stacked is defined by their meta-architecture. Stochastic Neural Architecture Search (SNAS) found cell architectures achieving state-of-the-art accuracy of 97.02%, while significantly reducing search time compared to other NAS methods. Cell-based neural architectures are an interesting target for usage on embedded systems, which are computer systems usually performing a single function within a larger system. These systems are often tightly constrained in resources. This research explores the effect that the architecture of the cells found by SNAS has on the accuracy, latency, and power usage on an embedded system. To answer this, architectures were found within a defined meta-architecture, and evaluated on the NVIDIA Jetson Nano. Multiple architectures were found achieving a lower latency and power usage while maintaining a comparable accuracy, one achieving 97.49% accuracy. 

# Requirements
```
Python >= 3.5.5, PyTorch >= 0.4.1, torchvision >= 0.2.1
```

# Trained architectures
Please contact me for the 54 trained architectures.