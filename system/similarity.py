"""
SOH-FL: Self-labeled Personalized Federated Learning
CT-AE (Cosine similarity-based sparse autoencoder) Implementation

This module implements the CT-AE component of SOH-FL, which extracts features
and computes similarities between client data to determine the most suitable
models for aggregation in the BS-Agg method.

The CT-AE uses a sparse autoencoder with cosine similarity regularization
to learn meaningful representations that capture client data similarities.
"""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import torch.optim as optim
import time
import warnings

# Suppress the tensor creation warning we're about to fix
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")

# ================================
# CONFIGURATION PARAMETERS
# ================================

# General parameters
ROUNDS = 10  # Number of validation rounds
INPUT_SIZE = 65  # Input feature dimension
CLIENT_NUMBER = 9  # Number of clients

# Loss function
criterion = nn.MSELoss()

# Ground truth similarity labels (which client each client is most similar to)
# For CICIDS2017: [5, 6, 7, 4, 3, 0, 8, 1, 2]
# For IoT-23 (12 clients): [5, 6, 7, 4, 3, 0, 8, 1, 2, 11, 10, 9]
correct = [5, 6, 7, 4, 3, 0, 8, 1, 2]

# ================================
# DATA PREPARATION
# ================================

def load_and_preprocess_data(dataset_name, num_clients, sample_size=1000):
    """
    Load and preprocess training and support data for CT-AE training.

    Args:
        dataset_name (str): Name of the dataset
        num_clients (int): Number of clients
        sample_size (int): Number of samples to randomly select from training data

    Returns:
        tuple: (train_listX, train_listY, support_listX, support_listY)
    """
    train_listX = []
    train_listY = []
    support_listX = []
    support_listY = []

    # Initialize scaler
    scaler = MinMaxScaler()

    print(f"Loading data for dataset: {dataset_name}")
    print(f"Number of clients: {num_clients}")

    # Load training data
    print("Loading training data...")
    for i in range(num_clients):
        try:
            # Load training features and labels
            xfile_name = f"../dataforsimi/{dataset_name}/train/{i}x.csv"
            yfile_name = f"../dataforsimi/{dataset_name}/train/{i}y.csv"

            x = np.loadtxt(xfile_name, delimiter=',')
            y = np.loadtxt(yfile_name, delimiter=',')

            # Randomly sample data to reduce computational cost
            if x.shape[0] > sample_size:
                indices = np.random.choice(x.shape[0], sample_size, replace=False)
                sampled_x = x[indices]
                sampled_y = y[indices]
            else:
                sampled_x = x
                sampled_y = y

            # Data validation
            if np.any(np.isinf(sampled_x)) or np.any(np.isnan(sampled_x)):
                print(f"Warning: Client {i} training data contains invalid values, skipping...")
                continue

            if np.any(np.abs(sampled_x) > np.finfo(np.float64).max):
                print(f"Warning: Client {i} training data exceeds float64 range, skipping...")
                continue

            # Scale features
            x_scaled = scaler.fit_transform(sampled_x)
            train_listX.append(x_scaled)
            train_listY.append(sampled_y)

        except FileNotFoundError:
            print(f"Warning: Training data files not found for client {i}")
            continue
        except Exception as e:
            print(f"Error loading training data for client {i}: {e}")
            continue

    # Load support data
    print("Loading support data...")
    for i in range(num_clients):
        try:
            # Load support features and labels
            xfile_name = f"../dataforsimi/{dataset_name}/Stest/{i}x.csv"
            yfile_name = f"../dataforsimi/{dataset_name}/Stest/{i}y.csv"

            x = np.loadtxt(xfile_name, delimiter=',')
            y = np.loadtxt(yfile_name, delimiter=',')

            # Scale features using the same scaler
            x_scaled = scaler.fit_transform(x)
            support_listX.append(x_scaled)
            support_listY.append(y)

        except FileNotFoundError:
            print(f"Warning: Support data files not found for client {i}")
            continue
        except Exception as e:
            print(f"Error loading support data for client {i}: {e}")
            continue

    print(f"Successfully loaded data for {len(train_listX)} clients")
    return train_listX, train_listY, support_listX, support_listY

# Dataset configuration
DATASET_NAME = "CICIDS2017QuantityFeature3"
# DATASET_NAME = 'IOT-23'  # Uncomment for IoT-23 dataset
# DATASET_NAME = "TON-IoT"  # Uncomment for TON-IoT dataset

# Load and preprocess data
train_listX, train_listY, support_listX, support_listY = load_and_preprocess_data(
    DATASET_NAME, CLIENT_NUMBER, sample_size=1000
)

# ================================
# CT-AE MODEL DEFINITION
# ================================

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with cosine similarity regularization for CT-AE.

    This autoencoder learns compressed representations of client data while
    maintaining sparsity constraints and optimizing for cosine similarity
    between similar clients.
    """

    def __init__(self, input_size, hidden_size, sparsity_target, sparsity_weight):
        """
        Initialize the sparse autoencoder.

        Args:
            input_size (int): Input feature dimension
            hidden_size (int): Compressed representation dimension
            sparsity_target (float): Target sparsity level for KL divergence
            sparsity_weight (float): Weight for sparsity loss component
        """
        super(SparseAutoencoder, self).__init__()

        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

        # Encoder: progressively reduces dimensionality
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, hidden_size)
        )

        # Decoder: reconstructs original input
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )

        # Optional: Add dropout for regularization
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input data

        Returns:
            tuple: (reconstructed_output, hidden_representation)
        """
        # Encode to compressed representation
        hidden = self.encoder(x)

        # Optional dropout
        # hidden = self.dropout(hidden)

        # Decode to reconstruct input
        output = self.decoder(hidden)

        return output, hidden


def kl_divergence(p, q):
    """
    Compute KL divergence for sparsity regularization.

    Args:
        p (float): Target sparsity level
        q (torch.Tensor): Actual activation levels

    Returns:
        torch.Tensor: KL divergence loss
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    q = torch.clamp(q, eps, 1 - eps)

    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


def compute_cosine_similarity_batch(x1, x2):
    """
    Efficiently compute cosine similarity between two batches of vectors.

    Args:
        x1 (np.ndarray): First batch of vectors (shape: [n1, d])
        x2 (np.ndarray): Second batch of vectors (shape: [n2, d])

    Returns:
        np.ndarray: Cosine similarities (shape: [min(n1, n2)])
    """
    # Handle different batch sizes by taking the minimum
    min_size = min(x1.shape[0], x2.shape[0])
    x1_subset = x1[:min_size]
    x2_subset = x2[:min_size]

    # Normalize vectors
    x1_norm = x1_subset / (np.linalg.norm(x1_subset, axis=1, keepdims=True) + 1e-8)
    x2_norm = x2_subset / (np.linalg.norm(x2_subset, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity element-wise
    similarities = np.sum(x1_norm * x2_norm, axis=1)

    return similarities


def compute_cosine_similarity_pairwise(x1, x2):
    """
    Compute average cosine similarity between all pairs of vectors.
    Optimized version using matrix operations.

    Args:
        x1 (np.ndarray): First batch of vectors (shape: [n1, d])
        x2 (np.ndarray): Second batch of vectors (shape: [n2, d])

    Returns:
        float: Average cosine similarity
    """
    # Handle edge cases
    if x1.size == 0 or x2.size == 0:
        return 0.0

    # Ensure 2D arrays
    if x1.ndim == 1:
        x1 = x1.reshape(1, -1)
    if x2.ndim == 1:
        x2 = x2.reshape(1, -1)

    # Normalize vectors
    eps = 1e-8
    x1_norm = x1 / (np.linalg.norm(x1, axis=1, keepdims=True) + eps)
    x2_norm = x2 / (np.linalg.norm(x2, axis=1, keepdims=True) + eps)

    # Compute cosine similarity matrix using matrix multiplication
    # Result shape: [n1, n2]
    similarity_matrix = np.dot(x1_norm, x2_norm.T)

    # Return average similarity
    return np.mean(similarity_matrix)

# ================================
# TRAINING CONFIGURATION
# ================================

# CT-AE hyperparameters
CT_AE_CONFIG = {
    'num_epochs': 100,          # Training epochs per client
    'batch_size': 128,          # Training batch size
    'learning_rate': 0.001,     # Optimizer learning rate
    'hidden_size': 10,          # Compressed representation dimension
    'sparsity_target': 0.1,     # Target sparsity level
    'sparsity_weight': 0.1,     # Sparsity loss weight
    'sample_size': 128,         # Size of random sample for cosine similarity
}

# Cosine weight tuning parameters
COSINE_WEIGHT_RANGE = np.arange(0, 30.1, 0.1)
VALIDATION_ROUNDS = ROUNDS

# ================================
# TRAINING AND EVALUATION FUNCTIONS
# ================================

def train_ct_ae_single_round(cosine_weight, config, train_data, support_data, correct_labels):
    """
    Train CT-AE for a single round with given cosine weight.

    Args:
        cosine_weight (float): Weight for cosine similarity loss
        config (dict): Training configuration
        train_data (list): Training datasets for each client
        support_data (list): Support datasets for each client
        correct_labels (list): Ground truth similarity labels

    Returns:
        dict: Accuracy results for different top-k predictions
    """
    # Initialize model
    autoencoder = SparseAutoencoder(
        INPUT_SIZE,
        config['hidden_size'],
        config['sparsity_target'],
        config['sparsity_weight']
    )
    optimizer = optim.Adam(autoencoder.parameters(), lr=config['learning_rate'])

    # Convert data to tensors (optimized)
    tensor_list = [torch.from_numpy(arr).float() for arr in train_data]
    sample_tensor = [torch.from_numpy(arr).float() for arr in support_data]

    encoded_data = []

    # Train the model for each client
    for num, dataset in enumerate(tensor_list):
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        for epoch in range(config['num_epochs']):
            for data in dataloader:
                inputs = data
                optimizer.zero_grad()
                outputs, hidden = autoencoder(inputs)

                # Optimized sampling for cosine similarity computation
                sub_array = train_data[num]
                if len(sub_array) > config['sample_size']:
                    indices = random.sample(range(len(sub_array)), config['sample_size'])
                    sub_data = sub_array[indices]
                else:
                    sub_data = sub_array
                sub_tensor = torch.from_numpy(sub_data).float()

                # Compute encoded representations
                mid_encode = autoencoder.encoder(dataset).detach().numpy()
                mid_sub_encode = autoencoder.encoder(sub_tensor).detach().numpy()

                # Compute average cosine similarity (using pairwise method for compatibility)
                average_cosin = compute_cosine_similarity_pairwise(mid_encode, mid_sub_encode)

                # Compute losses
                reconstruction_loss = criterion(outputs, inputs)
                avg_activation = torch.mean(hidden, dim=1)
                sparsity_loss = torch.sum(kl_divergence(config['sparsity_target'], avg_activation))

                # Total loss with cosine similarity regularization
                total_loss = (reconstruction_loss +
                             cosine_weight * (1 - average_cosin) +
                             config['sparsity_weight'] * sparsity_loss)

                total_loss.backward()
                optimizer.step()

        # Store encoded representations for evaluation
        encode = autoencoder.encoder(dataset).detach().numpy()
        encoded_data.append(encode)

    # Evaluate similarity prediction accuracy
    return evaluate_similarity_accuracy(autoencoder, sample_tensor, encoded_data, correct_labels)


def evaluate_similarity_accuracy(autoencoder, sample_tensor, encoded_data, correct_labels):
    """
    Evaluate the accuracy of similarity predictions.

    Args:
        autoencoder: Trained autoencoder model
        sample_tensor: Support data tensors
        encoded_data: Encoded training data for each client
        correct_labels: Ground truth similarity labels

    Returns:
        dict: Accuracy results for different top-k predictions
    """
    accuracy_results = {
        'top_1': 0, 'top_2': 0, 'top_3': 0, 'top_4': 0,
        'top_5': 0, 'top_6': 0, 'top_7': 0, 'top_8': 0
    }

    # Encode support data
    encode_sample = []
    for dataset2 in sample_tensor:
        encode2 = autoencoder.encoder(dataset2).detach().numpy()
        encode_sample.append(encode2)

    # Evaluate each support sample
    for count, q in enumerate(encode_sample):
        similarities = []

        # Compute similarities with all client encodings
        for v in encoded_data:
            # Compute average cosine similarity between support and client data
            average_sim = compute_cosine_similarity_pairwise(q, v)
            similarities.append(average_sim)

        # Sort similarities in descending order
        sorted_indices = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

        # Check top-k accuracy
        for k in range(1, 9):
            top_k_indices = [idx for idx, _ in sorted_indices[:k]]
            if correct_labels[count] in top_k_indices:
                accuracy_results[f'top_{k}'] += 1

    return accuracy_results


# ================================
# MAIN EXECUTION PIPELINE
# ================================

def main_ct_ae_tuning():
    """
    Main function to run CT-AE hyperparameter tuning.

    This function performs grid search over cosine weight values to find
    the optimal balance between reconstruction loss and cosine similarity loss.
    """
    print("=" * 80)
    print("SOH-FL CT-AE Hyperparameter Tuning")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Clients: {CLIENT_NUMBER}")
    print(f"Cosine weight range: {COSINE_WEIGHT_RANGE[0]:.1f} - {COSINE_WEIGHT_RANGE[-1]:.1f}")
    print(f"Validation rounds: {VALIDATION_ROUNDS}")
    print("=" * 80)

    start_time = time.time()
    results = []

    # Grid search over cosine weight values
    for cosine_weight in COSINE_WEIGHT_RANGE:
        print(f"\nTesting cosine weight: {cosine_weight:.1f}")

        # Aggregate results across multiple rounds
        total_accuracy = {
            'top_1': 0, 'top_2': 0, 'top_3': 0, 'top_4': 0,
            'top_5': 0, 'top_6': 0, 'top_7': 0, 'top_8': 0
        }

        # Run multiple validation rounds
        for round_idx in range(VALIDATION_ROUNDS):
            print(f"  Round {round_idx + 1}/{VALIDATION_ROUNDS}")

            # Train and evaluate for this round
            round_results = train_ct_ae_single_round(
                cosine_weight, CT_AE_CONFIG, train_listX, support_listX, correct
            )

            # Accumulate results
            for key in total_accuracy:
                total_accuracy[key] += round_results[key]

        # Store results for this cosine weight
        result_entry = {"cosine_weight": cosine_weight}
        for key, value in total_accuracy.items():
            result_entry[f"{key}_correct"] = value

        results.append(result_entry)

        # Print summary for this cosine weight
        print(f"  Top-1 accuracy: {total_accuracy['top_1']}/{VALIDATION_ROUNDS * CLIENT_NUMBER}")
        print(f"  Top-3 accuracy: {total_accuracy['top_3']}/{VALIDATION_ROUNDS * CLIENT_NUMBER}")

    # Save results to CSV
    df = pd.DataFrame(results)
    output_file = f"ct_ae_tuning_{DATASET_NAME.lower()}.csv"
    df.to_csv(output_file, index=False)

    # Find best cosine weight
    best_idx = df['top_1_correct'].idxmax()
    best_cosine_weight = df.loc[best_idx, 'cosine_weight']
    best_top1_accuracy = df.loc[best_idx, 'top_1_correct']

    end_time = time.time()

    # Print final results
    print("\n" + "=" * 80)
    print("CT-AE HYPERPARAMETER TUNING COMPLETED")
    print("=" * 80)
    print(f"Best cosine weight: {best_cosine_weight:.1f}")
    print(f"Best top-1 accuracy: {best_top1_accuracy}/{VALIDATION_ROUNDS * CLIENT_NUMBER}")
    print(f"Results saved to: {output_file}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print("=" * 80)

    return best_cosine_weight, df


if __name__ == "__main__":
    # Run CT-AE hyperparameter tuning
    best_weight, results_df = main_ct_ae_tuning()


