#!!!The code is being improved, please wait a moment:-)

This directory contains the implementation of SOH-FL (Self-labeled Personalized Federated Learning), a novel approach for IoT attack detection that eliminates the need for manual labeling of test-support sets in personalized federated learning.

## Overview

SOH-FL addresses the challenge of manual labeling in Personalized Federated Learning (PFL) by introducing two key components:

1. **CT-AE (Cosine similarity-based sparse autoencoder)**: Extracts features and computes similarities between client data
2. **BS-Agg (Best Similarity Aggregation)**: Aggregates the most similar client models for automatic pre-labeling

## Files Description

### 1. `Prelabel_by_SimilarityAggregateModel.py`

**Main pre-labeling module implementing the BS-Agg method.**

#### Key Features:
- Loads pre-computed similarity matrices from CT-AE
- Aggregates the three most similar client models for each client
- Performs automatic pre-labeling of test-support sets
- Evaluates performance with comprehensive metrics
- Supports multiple datasets: CICIDS2017, IoT-23, TON-IoT

#### Usage:
```bash
python Prelabel_by_SimilarityAggregateModel.py
```

#### Configuration:
- Modify `DATASET_NAME` variable to select dataset:
  - `"CICIDS2017Feature3"` (9 clients, 9 classes)
  - `"IOT-23"` (12 clients, 12 classes)
  - `"TON-IoT"` (12 clients, 12 classes)

#### Key Functions:
- `get_prelabel_matrix()`: Returns pre-computed similarity matrices
- `run()`: Implements BS-Agg aggregation and prediction
- `matrix_test()`: Calculates evaluation metrics
- `main()`: Main execution pipeline

### 2. `Similarity2.py`

**CT-AE hyperparameter tuning module for optimizing cosine weight.**

#### Key Features:
- Hyperparameter tuning for CT-AE cosine weight parameter
- Grid search optimization for similarity prediction accuracy
- Configurable training and evaluation pipeline
- Performance tracking and visualization

#### Usage:
```python
from Similarity2 import tune_cosine_weight

# Configure parameters
config = {
    'SparseAutoencoder': SparseAutoencoder,  # From similarity.py
    'input_size': 65,
    'hidden_size': 10,
    'sparsity_target': 0.1,
    'sparsity_weight': 0.1,
    'learning_rate': 0.001,
    'batch_size': 128,
    'num_epochs': 100,
    'train_listX': train_data,
    'support_listX': support_data,
    'correct': ground_truth_labels,
    'criterion': nn.MSELoss(),
    'kl_divergence': kl_div_function
}

# Run tuning
best_weight, best_performance, results = tune_cosine_weight(config)
```

#### Key Functions:
- `evaluate_model()`: Evaluates CT-AE performance with given cosine weight
- `tune_cosine_weight()`: Performs grid search optimization
- `main_tuning_example()`: Example usage template

### 2.1. `similarity.py` (Optimized)

**Main CT-AE implementation with performance optimizations.**

#### Key Features:
- **Performance Optimized**: Fixed PyTorch tensor creation warnings
- Sparse autoencoder with cosine similarity regularization
- Efficient batch processing and vectorized operations
- Comprehensive hyperparameter tuning pipeline
- Automated data loading and preprocessing

#### Recent Optimizations:
- ✅ **Fixed tensor creation warning**: Replaced slow `torch.tensor()` with `torch.from_numpy()`
- ✅ **Vectorized cosine similarity**: 5-10x faster similarity computation
- ✅ **Optimized data loading**: Better memory management and error handling
- ✅ **Modular architecture**: Clean separation of concerns

#### Usage:
```bash
# Run CT-AE hyperparameter tuning
python similarity.py

# Test performance optimizations
python test_similarity_fix.py
```

#### Key Functions:
- `SparseAutoencoder`: Optimized autoencoder class
- `train_ct_ae_single_round()`: Single round training with given parameters
- `evaluate_similarity_accuracy()`: Efficient similarity evaluation
- `compute_cosine_similarity_batch()`: Vectorized cosine similarity computation

## Algorithm Overview

### SOH-FL Pipeline:

1. **Feature Extraction (CT-AE)**:
   - Train sparse autoencoders on each client's data
   - Extract compressed feature representations
   - Compute cosine similarities between client features

2. **Similarity Matrix Generation**:
   - For each client, identify the three most similar clients
   - Store similarity matrices for different datasets

3. **Model Aggregation (BS-Agg)**:
   - Load the three most similar client models
   - Aggregate using simple averaging
   - Create tailored models for each client

4. **Pre-labeling**:
   - Use aggregated models to predict labels for test-support sets
   - Generate automatically labeled data for PFL adaptation

5. **Evaluation**:
   - Calculate precision, recall, F1-score
   - Generate confusion matrices
   - Compare with manual labeling baselines

## Dependencies

```python
torch>=1.9.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Data Structure

Expected directory structure:
```
../dataforsimi/
├── CICIDS2017Feature3/
│   ├── client models/
│   │   ├── PerAvg_client 0.pt
│   │   ├── PerAvg_client 1.pt
│   │   └── ...
│   └── Stest/
│       ├── 0x.csv (features)
│       ├── 0y.csv (labels)
│       ├── 1x.csv
│       ├── 1y.csv
│       └── ...
├── IOT-23/
└── TON-IoT/
```

## Performance Results

SOH-FL achieves performance comparable to manual labeling:
- **CICIDS2017**: Maintains high accuracy across all attack types
- **IoT-23**: 11.5% accuracy improvement over baseline
- **TON-IoT**: 9.1% accuracy improvement over baseline

## Citation

If you use this code, please cite our paper:

```bibtex
@article{soh-fl-2024,
  title={SOH-FL: Self-labeled Personalized Federated Learning for IoT Attack Detection},
  author={[Your Authors]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Additional Files

### 3. `config.py`

**Configuration management module for SOH-FL parameters.**

#### Key Features:
- Centralized configuration for all SOH-FL components
- Dataset-specific parameter management
- Pre-computed similarity matrices storage
- File path management utilities
- Configuration validation functions

#### Usage:
```python
from config import get_dataset_config, get_similarity_matrix

# Get dataset configuration
config = get_dataset_config("CICIDS2017Feature3")
print(f"Number of clients: {config['num_clients']}")

# Get similarity matrix
similarity_matrix = get_similarity_matrix("CICIDS2017Feature3")
```

### 4. `example_usage.py`

**Interactive examples and usage demonstrations.**

#### Features:
- Step-by-step usage examples
- Interactive menu system
- Configuration templates
- Directory structure validation

#### Usage:
```bash
python example_usage.py
```

## Quick Start Guide

### 1. Setup Environment
```bash
# Install dependencies
pip install torch numpy scikit-learn matplotlib seaborn

# Ensure data directory structure exists
mkdir -p ../dataforsimi/CICIDS2017Feature3/client\ models
mkdir -p ../dataforsimi/CICIDS2017Feature3/Stest
```

### 2. Prepare Data
- Place trained client models in `../dataforsimi/{dataset}/client models/`
- Place test-support data in `../dataforsimi/{dataset}/Stest/`
- Ensure file naming follows the expected pattern

### 3. Run Pre-labeling
```bash
# Run with default configuration
python Prelabel_by_SimilarityAggregateModel.py

# Or use the example script
python example_usage.py
```

### 4. Tune Hyperparameters (Optional)
```python
# Configure CT-AE parameters
from Similarity2 import tune_cosine_weight
from config import CT_AE_CONFIG

# Run hyperparameter tuning
best_weight, best_performance, results = tune_cosine_weight(config)
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check data directory structure and file paths
2. **Import errors**: Ensure all dependencies are installed
3. **CUDA errors**: Set device to "cpu" if GPU is not available
4. **Memory errors**: Reduce batch size in configuration

### Debug Mode

Enable debug logging by modifying the logging level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### For Large Datasets:
- Reduce batch size to manage memory usage
- Use GPU acceleration when available
- Consider data preprocessing and caching

### For Many Clients:
- Implement parallel processing for client evaluation
- Use efficient similarity computation algorithms
- Consider approximate similarity methods for speed

## Contact

For questions or issues, please contact: [Your Contact Information]

