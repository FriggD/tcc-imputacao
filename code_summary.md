# Code Base Summary

This document provides a summary of each Python file in the codebase and their main functions.

## tensorHelpers.py
PyTorch tensor manipulation utilities:
- `indexesFromSentence`: Converts sentences to index arrays
- `tensorFromSentence`: Creates tensors from sentences
- `tensorsFromPair`: Processes input/output pairs into tensors

## train.py
Model training functionality:
- `trainIters`: Main training loop for multiple iterations
- `train`: Single training iteration implementation
- Includes optimizer setup and loss calculation

## vcftogen.py
VCF (Variant Call Format) file processing:
- `readLines`: Reads specified number of lines from file
- `getFileHandler`: Creates file handlers with proper encoding
- `callRateCheck`: Validates call rates in genetic data
- `main`: Orchestrates VCF processing workflow

## logger.py
Logging configuration and setup:
- Configures logging for the application
- Sets up file and console output handlers
- Defines logging format and levels

## plot.py
Visualization utilities:
- `showPlot`: Creates and displays plots of training data
- Handles matplotlib configuration for visualization

## evaluation.py
Contains functions for model evaluation:
- `evaluateRandomly`: Tests the model on random samples
- `evaluateAll`: Comprehensive evaluation on all data
- `evaluate`: Core evaluation function for a single input

## helpers.py
Utility functions for various operations:
- `asMinutes`: Converts seconds to minutes format
- `timeSince`: Calculates time elapsed
- `split_every_n_elements`: Splits strings into fixed-size chunks

## impute.py
Core imputation functionality:
- `Lang` class for language processing
- `ModelData` class for handling training data
- Functions for reading and processing genetic data
- MAF (Minor Allele Frequency) computation
- Main execution logic

## attnDecoder.py
This file implements an attention-based decoder RNN (Recurrent Neural Network) as a PyTorch module. It contains:
- `AttnDecoderRNN` class with attention mechanism for sequence decoding
- Methods for initialization, forward pass with attention, and hidden state initialization
- Uses GRU (Gated Recurrent Unit) with attention mechanism for improved sequence processing

## device.py
A utility file that handles device selection for PyTorch:
- Automatically selects CUDA (GPU) if available, otherwise falls back to CPU
- Provides a global 'device' variable used across the project

## encoder.py
Implements the encoder part of the sequence-to-sequence model:
- `EncoderRNN` class for encoding input sequences
- Uses GRU for sequence processing
- Includes embedding layer for input processing
- Methods for initialization, forward pass, and hidden state initialization