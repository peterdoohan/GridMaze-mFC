"""
Library for custom pytorch implemention of sklearn.neural_network.MLPClassifier to run on GPU.
sklearn version is too slow to run for many permutation tests, run on GPU to speed up.
@peterdoohan
"""

# %% Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from . import allocentric_goal_decoding as agd

# %% Global Variables

# %% Dev


def test():
    decoder = MLPClassifier()
    return


# %% MLPClassifier Class


class MLPClassifier:
    """
    A PyTorch implementation of an MLP classifier with sklearn-like API.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
    activation : {'relu', 'tanh', 'logistic'}, default='relu'
        Activation function for the hidden layers.
    batch_size : int, default=64
        Size of minibatches for stochastic optimizers.
    lr : float, default=1e-3
        Learning rate.
    max_epochs : int, default=20
        Maximum number of training epochs.
    device : torch.device or str, optional
        Device to run the model on ('cpu' or 'cuda'). If None, will use GPU if available.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_layer_sizes=(50,),
        activation="relu",
        batch_size=64,
        lr=1e-3,
        max_epochs=20,
        device=None,
        random_state=None,
    ):
        # Reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build network
        activations = {"relu": nn.ReLU(inplace=True), "tanh": nn.Tanh(), "logistic": nn.Sigmoid()}
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activations.get(activation, nn.ReLU(inplace=True))

        # Training hyperparameters
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs

        # Placeholders; will initialize in fit()
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def _build_model(self, input_dim, output_dim):
        layers = []
        prev_dim = input_dim
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(self.activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        model = nn.Sequential(*layers)
        return model.to(self.device)

    def fit(self, X, y):
        """
        Fit the MLP classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values (class labels).
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert to numpy
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Initialize model and optimizer
        self.model = self._build_model(n_features, n_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Create DataLoader
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

            epoch_loss = total_loss / n_samples
            # print(f"Epoch {epoch+1}/{self.max_epochs} - loss: {epoch_loss:.4f}")

        return self

    def predict(self, X):
        """
        Predict using the trained MLP classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device)
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds
