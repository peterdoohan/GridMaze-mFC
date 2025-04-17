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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from . import allocentric_goal_decoding as agd
import pandas as pd

# %% Global Variables

# %% Dev


def test(activity_df, validation_folds_df, classifier="mlp_torch", verbose=True):
    """Returns decoding accuracy for each timepoint and fold"""
    timepoints = activity_df.firing_rate.columns
    folds = validation_folds_df.columns.get_level_values(0).unique()
    if isinstance(timepoints[0], tuple):  # event-aligned case
        results_index = pd.MultiIndex.from_tuples(
            [(*col, new_level) for col in timepoints for new_level in ["test", "train"]]
        )
    else:  # trial-aligned case
        results_index = pd.MultiIndex.from_tuples(
            [(col, new_level) for col in timepoints for new_level in ["test", "train"]]
        )
    results_df = pd.DataFrame(index=results_index, columns=folds)
    for fold in folds:
        if verbose:
            print(fold)
        # get test, train data for logistic regression
        fold_df = validation_folds_df[fold]
        if len(fold_df.test.columns[0][0]) > 1:  # remove empty index when combining data across sessions
            test_df = fold_df.test.droplevel(0, axis=1)
        else:
            test_df = fold_df.test
        test_X, test_y = agd.get_synthetic_activity_matrix(
            activity_df, test_df
        )  # [n_goals, n_session_clusters, n_timepoints], [n_goals]
        training_df = fold_df.training
        training_Xs, training_ys = [], []
        for training_set in training_df.columns.get_level_values(0).unique():
            training_set_df = training_df[training_set]
            X, y = agd.get_synthetic_activity_matrix(activity_df, training_set_df)
            training_Xs.append(X)
            training_ys.append(y)
        training_X = np.vstack(training_Xs)  # [n_training_sets x n_goals, n_session_clusters, n_timepoints]
        training_y = np.hstack(training_ys)  # [n_training_sets x n_goals]
        # set up decoder based on speified inputs
        if classifier == "mlp_torch":
            decoder = MLPtorchClassifier()
        else:
            raise NotImplementedError

        # run decoding at each timepoint
        for i in range(len(timepoints)):
            test_activity = test_X[:, :, i]  # [n_goals, n_session_clusters]
            training_activity = training_X[:, :, i]  # [n_training_sets x n_goals, n_session_clusters]
            decoder.fit(training_activity, training_y)
            test_predictions = decoder.predict(test_activity)
            test_accuracy = (test_predictions == test_y).mean()
            train_predictions = decoder.predict(training_activity)
            train_accuracy = (train_predictions == training_y).mean()
            rtest_loc = (*timepoints[i], "test") if isinstance(timepoints[i], tuple) else (timepoints[i], "test")
            rtrain_loc = (*timepoints[i], "train") if isinstance(timepoints[i], tuple) else (timepoints[i], "train")
            results_df.loc[rtest_loc, fold] = test_accuracy
            results_df.loc[rtrain_loc, fold] = train_accuracy
    return results_df


# %% MLPClassifier Class


class MLPtorchClassifier:
    """
    A PyTorch implementation of an MLP classifier with sklearn-like API,
    supporting automatic one-hot encoding of categorical input features,
    label encoding of categorical targets, L2 weight decay, and early stopping.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        Sizes of each hidden layer.
    activation : {'relu', 'tanh', 'logistic'}, default='relu'
        Activation function for hidden layers.
    batch_size : int, default=64
        Mini-batch size for training.
    lr : float, default=1e-3
        Learning rate.
    max_epochs : int, default=20
        Maximum number of training epochs.
    alpha : float, default=0.0
        L2 penalty (weight decay) coefficient. Equivalent to sklearn's alpha.
    tol : float, default=1e-4
        Tolerance for early stopping. Training stops if loss change over the last
        10 epochs is less than tol.
    verbose : bool, default=False
        If True, prints loss at each epoch and early stopping notification.
    device : torch.device or str, optional
        Device for computation ('cpu' or 'cuda'). Defaults to CUDA if available.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        batch_size=64,
        lr=1e-3,
        max_epochs=500,
        alpha=0.0,
        tol=1e-4,
        verbose=True,
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
        if verbose:
            print(f"Using device: {self.device}")

        # Activation mapping
        activations = {"relu": nn.ReLU(inplace=True), "tanh": nn.Tanh(), "logistic": nn.Sigmoid()}
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activations.get(activation, nn.ReLU(inplace=True))

        # Training parameters
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.alpha = alpha
        self.tol = tol
        self.verbose = verbose

        # Placeholders for encoders and model
        self.encoder = None
        self.label_encoder = None
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
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):
        """
        Fit the MLP classifier, encoding categorical features and targets,
        with L2 weight decay and early stopping based on tol.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            Training input. Non-numeric columns will be one-hot encoded.
        y : array-like, shape (n_samples,)
            Target labels. Categorical labels will be label-encoded.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert to numpy and reshape
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)

        # Encode target if categorical
        if y.dtype.kind in ("U", "S", "O"):
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y.ravel())
        else:
            y_enc = y.ravel().astype(np.int64)

        # Encode input features if categorical
        if X.dtype.kind in ("U", "S", "O"):
            self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            X_enc = self.encoder.fit_transform(X)
        else:
            X_enc = X.astype(np.float32)
        X_enc = X_enc.astype(np.float32)

        # Prepare dimensions
        n_samples, n_features = X_enc.shape
        n_classes = len(np.unique(y_enc))

        # Build model and optimizer with weight decay
        self.model = self._build_model(n_features, n_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.alpha)

        # Create DataLoader
        dataset = TensorDataset(torch.from_numpy(X_enc), torch.from_numpy(y_enc))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop with early stopping
        epoch_losses = []
        self.model.train()
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * X_batch.size(0)
            epoch_loss = total_loss / n_samples
            epoch_losses.append(epoch_loss)

            if self.verbose:
                print(f"Epoch {epoch}/{self.max_epochs} - loss: {epoch_loss:.6f}")

            # Check early stopping: loss variation over last 10 epochs
            if len(epoch_losses) >= 10:
                recent_losses = epoch_losses[-10:]
                if max(recent_losses) - min(recent_losses) < self.tol:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch} (loss change < tol={self.tol})")
                    break

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            Input data. Must match encoding from fit().

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (original dtype if categorical).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Transform features
        if self.encoder is not None:
            X_enc = self.encoder.transform(X)
        else:
            X_enc = X.astype(np.float32)
        X_enc = X_enc.astype(np.float32)

        # Inference
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_enc).to(self.device)
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Inverse transform target labels
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(preds)
        return preds
