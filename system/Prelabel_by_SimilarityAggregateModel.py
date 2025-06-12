"""
SOH-FL: Self-labeled Personalized Federated Learning
Pre-labeling Module using Similarity-based Model Aggregation (BS-Agg)

This module implements the BS-Agg (Best Similarity Aggregation) method for pre-labeling
test-support sets in personalized federated learning. It aggregates the most similar
client models based on cosine similarity computed by CT-AE (Cosine similarity-based
sparse autoencoder) to create tailored models for automatic labeling.

Paper: SOH-FL: Self-labeled Personalized Federated Learning for IoT Attack Detection
Authors: [Your Authors]
"""

import torch
import numpy as np
import argparse
import os
import time
import warnings
import logging
from flcore.trainmodel.models import *
from utils.mem_utils import MemReporter
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
warnings.simplefilter("ignore")
torch.manual_seed(0)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# ================================
# CONFIGURATION PARAMETERS
# ================================

# Dataset configuration
DATASET_OPTIONS = {
    "CICIDS2017Feature3": {"clients": 9, "classes": 9},
    "IOT-23": {"clients": 12, "classes": 12},
    "TON-IoT": {"clients": 12, "classes": 12}
}

# Current dataset selection
DATASET_NAME = "CICIDS2017Feature3"
# DATASET_NAME = 'IOT-23'
# DATASET_NAME = 'TON-IoT'

# Get dataset parameters
client_number = DATASET_OPTIONS[DATASET_NAME]["clients"]
num_classes = DATASET_OPTIONS[DATASET_NAME]["classes"]

# Global variables for storing results
support_listX = []  # Support set features
support_listY = []  # Support set labels
predict = []        # All predictions
true = []          # All true labels



def matrix_test(y_really, y_predict):
    """
    Calculate and display evaluation metrics including precision, recall, F1-score and confusion matrix.

    Args:
        y_really (array): True labels
        y_predict (array): Predicted labels

    Returns:
        tuple: (precision, recall, f1, confusion_matrix)
    """
    reporter = MemReporter()

    # Define labels based on number of classes
    labels = list(range(num_classes))

    # Calculate confusion matrix
    matrix = metrics.confusion_matrix(y_really, y_predict, labels=labels)

    # Calculate evaluation metrics
    precision = precision_score(y_really, y_predict, average='macro')
    recall = recall_score(y_really, y_predict, average='macro')
    f1 = f1_score(y_really, y_predict, average='macro')

    # Display results
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion matrix:\n", matrix)

    reporter.report()
    return precision, recall, f1, matrix

class DynamicFedAvgCNN_conv1(torch.nn.Module):
    """
    Dynamic version of FedAvgCNN_conv1 that can adapt to different model2 input dimensions.
    """

    def __init__(self, in_channels=1, num_classes=9, dim=224, model2_input_dim=224, reshape_dim=34):
        super().__init__()
        self.reshape_dim = reshape_dim

        # Model1: Conv1d layers (exact match to original)
        self.model1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, 2),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(16, 32, 2),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool1d(4),
            torch.nn.Flatten(),
        )

        # Additional layer to handle dimension mismatch
        self.model1_output_dim = 32 * 3  # 96 for reshape_dim=34
        if self.model1_output_dim != model2_input_dim:
            self.dimension_adapter = torch.nn.Linear(self.model1_output_dim, model2_input_dim)
        else:
            self.dimension_adapter = None

        # Model2: Linear layer with dynamic input dimension
        self.model2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=model2_input_dim, out_features=num_classes, bias=True),
            torch.nn.Sigmoid(),
        )

        # FC layers - fc1 input comes from model2 output (num_classes)
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(num_classes, 512),  # Input is model2 output
            torch.nn.ReLU(inplace=True)
        )
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        # Handle data preprocessing
        if x.dim() == 2:  # [batch_size, features]
            batch_size = x.shape[0]
            current_features = x.shape[1]

            if current_features > self.reshape_dim:
                x = x[:, :self.reshape_dim]
            elif current_features < self.reshape_dim:
                padding = torch.zeros(batch_size, self.reshape_dim - current_features)
                x = torch.cat([x, padding], dim=1)

        # Reshape for Conv1d: [batch_size, channels, length]
        x = x.reshape(-1, 1, self.reshape_dim)

        # Forward pass following the original model's flow
        out = self.model1(x)  # Conv1d layers -> output shape [batch_size, 96]

        # Adapt dimension if needed
        if self.dimension_adapter is not None:
            out = self.dimension_adapter(out)  # -> output shape [batch_size, 224]

        out = self.model2(out)  # Linear layer -> output shape [batch_size, 9]
        out = torch.flatten(out, 1)  # Ensure flat
        out = self.fc1(out)  # fc1 expects input of size 9
        out = self.fc(out)
        return out


class ModelWrapper:
    """
    Wrapper class to handle model loading and provide a consistent interface.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model(x)

    def eval(self):
        self.model.eval()

    def load_state_dict(self, state_dict):
        # Load with strict=False to handle missing dimension_adapter
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Note: Missing keys (will be randomly initialized): {missing_keys}")
        if unexpected_keys:
            print(f"Note: Unexpected keys (will be ignored): {unexpected_keys}")


def load_model_safely(model_path):
    """
    Safely load a model and extract its architecture information.

    Args:
        model_path (str): Path to the model file

    Returns:
        dict: Model state dictionary
    """
    try:
        model_state = torch.load(model_path, map_location='cpu')
        return model_state
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def get_dataset_reshape_dim(dataset_name):
    """
    Get the correct reshape dimension for different datasets.

    Based on the model's forward method, different datasets use different reshape dimensions:
    - CICIDS2017: 65 -> should be reshaped to (-1, 1, 65) but model uses 34
    - IOT-23: 25
    - TON-IoT: 31
    - Car_Hacking: 11
    - CAN-FD: 34
    - HCRL: 10

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        int: Reshape dimension
    """
    reshape_dims = {
        "CICIDS2017Feature3": 34,  # Based on model code line 196
        "CICIDS2017": 34,
        "IOT-23": 25,
        "TON-IoT": 31,
        "Car_Hacking": 11,
        "CAN-FD": 34,
        "HCRL": 10
    }

    return reshape_dims.get(dataset_name, 34)  # Default to 34


def infer_model_architecture(state_dict, dataset_name):
    """
    Infer model architecture from state dictionary and dataset.

    Args:
        state_dict (dict): Model state dictionary
        dataset_name (str): Name of the dataset

    Returns:
        dict: Architecture parameters
    """
    # Get reshape dimension for this dataset
    reshape_dim = get_dataset_reshape_dim(dataset_name)

    # Look for key patterns to infer architecture
    arch_info = {
        'in_channels': 1,           # Always 1 for Conv1d
        'num_classes': 9,           # Default
        'dim': 224,                 # Default based on inspection
        'reshape_dim': reshape_dim  # Dataset-specific reshape dimension
    }

    # Try to infer from state dict keys
    for key, tensor in state_dict.items():
        if 'model2.0.weight' in key:  # model2 layer weight
            arch_info['num_classes'] = tensor.shape[0]  # Output classes
            arch_info['dim'] = tensor.shape[1]  # Input to model2 (this is the dim parameter!)
        elif 'fc1.0.weight' in key:  # FC layer weight
            # fc1 input comes from model2 output, so this should match num_classes
            pass

    return arch_info


def run(args, x, y, num1, num2, num3, dataset_name):
    """
    BS-Agg: Best Similarity Aggregation method for pre-labeling.

    This function implements the core BS-Agg algorithm that:
    1. Loads the three most similar client models based on CT-AE similarity
    2. Aggregates them using simple averaging
    3. Uses the aggregated model to predict labels for the test-support set

    Args:
        args: Argument parser containing model configuration
        x (torch.Tensor): Input features for prediction
        y (numpy.array): True labels for evaluation
        num1, num2, num3 (int): Indices of the three most similar client models
        dataset_name (str): Name of the dataset being used

    Returns:
        tuple: (predictions, test_accuracy, test_samples, precision, recall, f1, confusion_matrix)
    """
    test_acc = 0
    test_num = 0
    y_really = []
    y_predict = []

    # Load the three most similar client models (BS-Agg step 1)
    print(f"Loading and aggregating models: {num1}, {num2}, {num3}")

    model_paths = [
        f'../dataforsimi/{dataset_name}/client models/PerAvg_client {num1}.pt',
        f'../dataforsimi/{dataset_name}/client models/PerAvg_client {num2}.pt',
        f'../dataforsimi/{dataset_name}/client models/PerAvg_client {num3}.pt'
    ]

    # Load models safely
    models = []
    for path in model_paths:
        model_state = load_model_safely(path)
        if model_state is None:
            print(f"Failed to load model from {path}")
            return None, 0, 0, 0, 0, 0, None
        models.append(model_state)

    model1, model2, model3 = models

    # Infer architecture from the first model
    arch_info = infer_model_architecture(model1, dataset_name)
    print(f"Inferred architecture: {arch_info}")

    # Aggregate models using simple averaging (BS-Agg step 2)
    new_state_dict = {}
    for key in model1.keys():
        try:
            new_state_dict[key] = (model1[key] + model2[key] + model3[key]) / 3
        except Exception as e:
            print(f"Error aggregating parameter {key}: {e}")
            # Use the first model's parameter if aggregation fails
            new_state_dict[key] = model1[key]

    # Perform inference with aggregated model
    with torch.no_grad():
        try:
            # Initialize dynamic model with inferred parameters
            base_model = DynamicFedAvgCNN_conv1(
                in_channels=arch_info['in_channels'],
                num_classes=arch_info['num_classes'],
                dim=arch_info['dim'],
                model2_input_dim=arch_info['dim'],  # Use dim for model2 input
                reshape_dim=arch_info['reshape_dim']
            )

            # Wrap model
            args.model = ModelWrapper(base_model)
            args.model.load_state_dict(new_state_dict)
            args.model.eval()

            # Make predictions
            output = args.model(x)
            max_indices = (torch.argmax(output, dim=1)).numpy()

            # Calculate accuracy
            test_acc += np.sum(max_indices == y)
            test_num += y.shape[0]
            y_predict.append(max_indices)
            y_really.append(y)

        except Exception as e:
            print(f"Error during model inference: {e}")
            print("Model state dict keys:", list(new_state_dict.keys()))

            # Try alternative model architectures based on inspection
            alternative_configs = [
                {'in_channels': 1, 'num_classes': 9, 'dim': 224},  # Most likely correct
                {'in_channels': 1, 'num_classes': 9, 'dim': 96},   # Alternative
                {'in_channels': 1, 'num_classes': 9, 'dim': 128},  # Alternative
                {'in_channels': 65, 'num_classes': 9, 'dim': 224}, # Fallback
            ]

            for config in alternative_configs:
                try:
                    print(f"Trying alternative config: {config}")
                    base_model = DynamicFedAvgCNN_conv1(
                        in_channels=config['in_channels'],
                        num_classes=config['num_classes'],
                        dim=config['dim'],
                        model2_input_dim=config['dim'],
                        reshape_dim=arch_info['reshape_dim']
                    )
                    args.model = ModelWrapper(base_model)
                    args.model.load_state_dict(new_state_dict)
                    args.model.eval()

                    output = args.model(x)
                    max_indices = (torch.argmax(output, dim=1)).numpy()

                    test_acc += np.sum(max_indices == y)
                    test_num += y.shape[0]
                    y_predict.append(max_indices)
                    y_really.append(y)

                    print(f"Success with config: {config}")
                    break

                except Exception as e2:
                    print(f"Failed with config {config}: {e2}")
                    continue
            else:
                print("All model configurations failed!")
                return None, 0, 0, 0, 0, 0, None

    # Store global results
    predict.extend(y_predict[0])
    true.extend(y_really[0])

    print("Accuracy:", test_acc / test_num)
    precision, recall, f1, matrix = matrix_test(y_really[0], y_predict[0])

    return y_predict, test_acc, test_num, precision, recall, f1, matrix


# ================================
# ARGUMENT PARSER SETUP
# ================================

def setup_args():
    """Setup command line arguments for the SOH-FL pre-labeling system."""
    parser = argparse.ArgumentParser(description="SOH-FL Pre-labeling with BS-Agg")

    # General parameters
    parser.add_argument('-go', "--goal", type=str, default="test", help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="CICIDS2017")
    parser.add_argument('-nb', "--num_classes", type=int, default=num_classes)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=40)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-ls', "--local_epochs", type=int, default=3, help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="PerAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.2, help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False, help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=client_number, help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0, help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False, help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)

    # Practical parameters
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0, help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0, help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False, help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="The threthold for droping slow clients")

    # Algorithm-specific parameters
    parser.add_argument('-bt', "--beta", type=float, default=0.001)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0, help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5, help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01, help="personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument('-M', "--M", type=int, default=5, help="Server only sends M client models to one client at each round")
    parser.add_argument('-itk', "--itk", type=int, default=4000, help="The iterations for solving quadratic subproblems")
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2, help="More fine-graind than its original paper.")
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    return parser.parse_args()
# ================================
# PRE-LABELING SIMILARITY MATRICES
# ================================

def get_prelabel_matrix(dataset_name):
    """
    Get the pre-computed similarity matrix for model selection.

    These matrices contain the indices of the three most similar client models
    for each client, computed using CT-AE (Cosine similarity-based sparse autoencoder).
    Each row represents a client, and the three values are the indices of the
    most similar client models to aggregate for pre-labeling.

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        list: Matrix where each row contains [model1_idx, model2_idx, model3_idx]
    """

    if dataset_name == "CICIDS2017Feature3":
        # Pre-computed similarity matrix for CICIDS2017 (AE+cosine Quantity)
        return [
            [5, 8, 1],  # Client 0: most similar to clients 5, 8, 1
            [1, 3, 6],  # Client 1: most similar to clients 1, 3, 6
            [3, 1, 6],  # Client 2: most similar to clients 3, 1, 6
            [8, 3, 5],  # Client 3: most similar to clients 8, 3, 5
            [8, 3, 1],  # Client 4: most similar to clients 8, 3, 1
            [3, 8, 5],  # Client 5: most similar to clients 3, 8, 5
            [8, 3, 6],  # Client 6: most similar to clients 8, 3, 6
            [3, 8, 6],  # Client 7: most similar to clients 3, 8, 6
            [3, 8, 5]   # Client 8: most similar to clients 3, 8, 5
        ]

    elif dataset_name == "IOT-23":
        # Pre-computed similarity matrix for IoT-23 (cosine+sparse)
        return [
            [8, 5, 4],   # Client 0
            [3, 6, 4],   # Client 1
            [8, 9, 5],   # Client 2
            [11, 9, 8],  # Client 3
            [3, 6, 7],   # Client 4
            [0, 1, 2],   # Client 5
            [8, 9, 11],  # Client 6
            [0, 2, 1],   # Client 7
            [0, 1, 2],   # Client 8
            [11, 9, 8],  # Client 9
            [11, 5, 4],  # Client 10
            [11, 2, 5]   # Client 11
        ]

    elif dataset_name == "TON-IoT":
        # Pre-computed similarity matrix for TON-IoT (cosine)
        return [
            [7, 5, 4],   # Client 0
            [7, 6, 11],  # Client 1
            [7, 11, 8],  # Client 2
            [7, 5, 4],   # Client 3
            [3, 7, 6],   # Client 4
            [7, 5, 9],   # Client 5
            [11, 10, 8], # Client 6
            [11, 10, 8], # Client 7
            [7, 10, 11], # Client 8
            [11, 10, 8], # Client 9
            [10, 11, 8], # Client 10
            [7, 9, 5]    # Client 11
        ]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Get the similarity matrix for current dataset
prelabel = get_prelabel_matrix(DATASET_NAME)

# ================================
# MAIN EXECUTION FUNCTION
# ================================

def main():
    """
    Main function to execute SOH-FL pre-labeling using BS-Agg method.

    This function:
    1. Sets up arguments and loads test-support data
    2. For each client, aggregates the most similar models using BS-Agg
    3. Uses aggregated models to pre-label test-support sets
    4. Saves pre-labeled data and displays overall performance metrics
    """
    print("=" * 60)
    print("SOH-FL: Self-labeled Personalized Federated Learning")
    print("Pre-labeling with BS-Agg (Best Similarity Aggregation)")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Number of clients: {client_number}")
    print(f"Number of classes: {num_classes}")
    print("=" * 60)

    # Setup arguments
    args = setup_args()

    # Start timing
    start_time = time.time()
    test_Sdata = []

    # Process each client
    for i in range(client_number):
        print(f"\n=== Processing Client {i} ===")

        # Load test-support data for current client
        xfile_name = f"../dataforsimi/{DATASET_NAME}/Stest/{i}x.csv"
        yfile_name = f"../dataforsimi/{DATASET_NAME}/Stest/{i}y.csv"

        try:
            x_np = np.loadtxt(xfile_name, delimiter=',')
            y = np.loadtxt(yfile_name, delimiter=',')

            # Convert to tensor - the model will handle reshaping internally
            # Based on FedAvgCNN_conv1.forward(), it expects [batch_size, features]
            # and will reshape to [batch_size, 1, reshape_dim] internally
            if x_np.ndim == 1:
                x_np = x_np.reshape(1, -1)

            x = torch.from_numpy(x_np).float()
            print(f"Data shape for client {i}: {x.shape}")

        except FileNotFoundError:
            print(f"Warning: Data files not found for client {i}")
            continue
        except Exception as e:
            print(f"Error loading data for client {i}: {e}")
            continue

        # Store support data
        support_listX.append(x)
        support_listY.append(y)

        # Get the three most similar client models for current client
        num1, num2, num3 = prelabel[i][0], prelabel[i][1], prelabel[i][2]
        print(f"Using models from clients: {num1}, {num2}, {num3}")

        # Apply BS-Agg and get predictions
        y_predict, _, _, _, _, _, _ = run(
            args, x, y, num1, num2, num3, DATASET_NAME
        )

        # Store results
        y_np = y_predict[0]
        data_dict = {'x': x_np, 'y': y_np}

        # Analyze prediction differences
        differences = np.where(y_np != y)
        print(f"Prediction differences: {len(differences[0])} out of {len(y)} samples")
        if len(differences[0]) > 0:
            print(f"Different indices: {differences[0][:10]}...")  # Show first 10
            print(f"Predicted: {y_np[differences][:10]}")  # Show first 10
            print(f"True: {y[differences][:10]}")  # Show first 10

        test_Sdata.append(data_dict)

    # Save pre-labeled data
    print(f"\n=== Saving Pre-labeled Data ===")
    output_dir = f'../dataforsimi/{DATASET_NAME}/Stest/simiPrelabel/'
    os.makedirs(output_dir, exist_ok=True)

    for idx, test_Sdict in enumerate(test_Sdata):
        output_file = os.path.join(output_dir, f'{idx}.npz')
        with open(output_file, 'wb') as f:
            np.savez_compressed(f, data=test_Sdict)
    print(f"Saved pre-labeled data for {len(test_Sdata)} clients")

    # Calculate total runtime
    end_time = time.time()

    # Display overall performance
    print(f"\n=== Overall Performance ===")
    if len(true) > 0 and len(predict) > 0:
        matrix_test(true, predict)

        # Create and display confusion matrix
        cm = confusion_matrix(true, predict)
        plt.figure(figsize=(10, 8))

        # Generate class labels dynamically
        class_labels = [f'Class {i}' for i in range(num_classes)]

        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
                    xticklabels=class_labels,
                    yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'SOH-FL Pre-labeling Results - {DATASET_NAME}\nUsing BS-Agg (Best Similarity Aggregation)')
        plt.tight_layout()
        plt.show()

    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print("=" * 60)
    print("SOH-FL Pre-labeling completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()