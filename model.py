"""
Contains all functions and classes related to the model.
"""
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
import torch.nn.functional as F
import re
import spacy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('dark')

w2v = spacy.load('en_core_web_md')


# Network Class
class Net(nn.Module):
    """
    Class for the DeepNet.
    """
    def __init__(self, nodes_in=300, nodes_out=3, nodes_hidden=[64, 32],
                 drop_prob=.5, batch_norm=True):
        super(Net, self).__init__()

        # External w2v model
        global w2v
        self.w2v = w2v
        # Model architecture
        self.batch_norm = batch_norm
        self.nodes_in = nodes_in
        self.nodes_hidden = nodes_hidden
        self.nodes_out = nodes_out
        self.drop_prob = drop_prob

        self.dropout = nn.Dropout(drop_prob)

        # Input Layer
        self.fc1 = nn.Linear(nodes_in, nodes_hidden[0])

        # Hidden Layers
        for i in range(len(nodes_hidden) - 1):
            setattr(self, 'fc' + str(i + 2),
                    nn.Linear(nodes_hidden[i], nodes_hidden[i + 1]))
            setattr(self, 'bn' + str(i + 1),
                    nn.BatchNorm1d(nodes_hidden[i]))
        # Output Layer
        setattr(self, 'fc' + str(len(nodes_hidden) + 1),
                nn.Linear(nodes_hidden[-1], nodes_out))

    def forward(self, x):
        # Forward pass
        x = F.relu(self.fc1(x))
        for i in range(len(self.nodes_hidden) - 1):
            # If using batch norm
            if self.batch_norm:
                x = getattr(self, 'bn' + str(i + 1))(x)
            x = self.dropout(x)
            x = F.relu(getattr(self, 'fc' + str(i + 2))(x))

        x = self.dropout(x)
        x = getattr(self, 'fc' + str(len(self.nodes_hidden) + 1))(x)
        return x


class HateClassifierDataset(Dataset):
    """
    Class that creates the Dataset for the input files."""

    def __init__(self, X_csv_file, y_csv_file, normalized=True):
        """
        Args:
            X_csv_file (string): Path to the features csv file.
            y_csv_file (string): Path to the target csv file.
            normalized (bool): If the features vector will be normalized.
        """
        # Load data
        self.features = pd.read_csv(X_csv_file, index_col=0)
        self.target = pd.read_csv(y_csv_file, index_col=0)
        self.normalized = normalized

        # Normalize features
        self.feat_mean = self.features.mean()
        self.feat_std = self.features.std()
        self.normalized_features = (self.features -
                                    self.feat_mean) / self.feat_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        idx = int(idx)

        # Normalization
        if self.normalized:
            vectors = self.normalized_features.iloc[idx, :].values
        else:
            vectors = self.features.iloc[idx, :].values

        labels = self.target.iloc[idx, :].values

        # Convert to tensors
        return (torch.from_numpy(vectors).float(),
                torch.from_numpy(labels).float())


def predict_probs(model, inputs):
    """
    Convert model outputs to prediction probabilities.

    Arg:
        model: Model to be used.
        inputs: inputs in format

    Return:
        prob: Tensor (1, 3) that represents probabilities for the 3 classes.
    """
    # Forward pass
    with torch.no_grad():
        logits = model(inputs)

    # Probabilities
    probs = torch.sigmoid(logits)

    return probs.cpu().numpy()


def load_model(file_name):
    """
    Return saved model in file_name.

    Args:
        file_name: Name of the file to load the model from.

    Return:
        model: the loaded model.
    """
    # Load data
    checkpoint = torch.load(file_name, map_location='cpu')

    # Initialize model and optimizer
    model = Net(checkpoint['nodes_in'], checkpoint['nodes_out'],
                checkpoint['nodes_hidden'], checkpoint['prop_prob'],
                checkpoint['batch_norm'])
    criterion = nn.BCEWithLogitsLoss()  # Loss for multilabel problem
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    # Recover states
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def predict(model, text_input, dataset):
    """
    Predicts the text using the input model.

    Args:
        model: model used for prediction.
        text_inputs: string with the text input.
    """
    # Cleans input
    clean_text = re.sub(r'[^\w\s]', '', text_input)  # Removes punctuation

    # Applies word to vector embedding
    vector = model.w2v(clean_text).vector
    normed_vector = (vector - dataset.feat_mean) / dataset.feat_std
    inputs = torch.unsqueeze(torch.tensor(normed_vector), dim=0)

    # Uses model
    model.eval()   # Evaluation mode
    model.cpu()    # Move to cpu

    # Probabilities
    probs = predict_probs(model, inputs).squeeze()

    return probs


def print_probs(model, text, dataset):
    """
    Plots the results.

    Args:
        model: Model used.
        text: text to generate the probabilities.
        dataset: dataset class.
    """
    # Calculate Probs
    values = predict(model, text, dataset)
    x = ["Agressiveness", "Direct Attack", "Toxicity"]

    # plt.bar(x, values)

    threshold = 0.5

    # split it up
    above_threshold = np.maximum(values - threshold, 0)
    below_threshold = np.minimum(values, threshold)

    # and plot it
    fig, ax = plt.subplots()
    ax.bar(x, below_threshold, 0.35, color="blue")
    ax.bar(x, above_threshold, 0.35, color="red",
           bottom=below_threshold)

    # horizontal line indicating the threshold
    ax.plot([0., len(x) - 1], [threshold, threshold], "k--")
    ax.set_ylim([0, 1])
    ax.set_title(text)

    return ax