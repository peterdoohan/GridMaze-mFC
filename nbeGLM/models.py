import time
import torch
import numpy as np
from nbeGLM import utils

# @title Specify the model


class Encoder(torch.nn.Module):
    """pytorch model for learning task embeddings"""

    def __init__(
        self,
        Nhid=[100, 50],
        Nlat=15,
        beta_act=1e-1,
        beta_weight=1e-1,
        partition=None,
        latent_nonlin=None,
    ):
        """
        input_streams (list of int arrays): indices for each input stream. Use [np.arange(Nin)] to treat everything as one stream.
        Nhid (list of int): list of hidden layer dimensions (for each input stream)
        Nlat (int): latent dimension (output of the embedding streams)
        Nout (int): output dimension (total number of neurons)
        beta_act, beta_weight (floats): regularization parameters
        partition (list of int lists): optional partitioning of inputs into things that only combine linearly at the latent.
            each list specifies a set of input streams to be embedded together. Default: a single embedding including all input streams not in 'latent_inputs'
        """
        super(Encoder, self).__init__()

        # convert input_streams to tensor
        assert latent_nonlin in {None, "relu"}

        # how do we partition the input data
        self.partition = partition

        # set the parameters
        self.Nhid, self.Nlat = Nhid, Nlat
        self.beta_act, self.beta_weight = beta_act, beta_weight

        # set latent nonlinearity
        self.latent_nonlin = latent_nonlin

    def set_input_groups(self, train_data):

        self.input_group_indices = [torch.tensor(x) for x in train_data[0]["input_group_indices"]]
        self.input_group_names = train_data[0]["input_group_names"]

        # potentially partition inputs into non-interacting streams
        # self.parition is plain text indicating the names of the input groups in each partition
        # self.partition_input_indices are tensors of the corresponding indices of the input data
        if self.partition is None:
            self.partition_input_indices = None
        else:
            self.partition_input_indices = []
            for module in self.partition:  # get input indices for each block
                module_inds = []
                for group in module:
                    if isinstance(group, str):

                        if group not in self.input_group_names:
                            raise ValueError(
                                f"You tried to partition your data, but {group} does not exist in the input data."
                            )

                        module_inds.append(self.input_group_names.index(group))
                    else:
                        module.inds.append(group)

                new_partition = [self.input_group_indices[ind] for ind in module_inds]

                assert len(new_partition) > 0

                self.partition_input_indices.append(np.concatenate(new_partition))

        # specify number of embedding input dimensions
        if self.partition is None:  # no partitioning
            self.Nin = np.sum([len(inds) for inds in self.input_group_indices])
        else:
            self.Nin = np.sum([len(inds) for inds in self.partition_input_indices])

    def initialise_weights(self, train_data):

        self.num_neurons_per_session = [s["spikes"].shape[0] for s in train_data]
        self.Nout = sum(self.num_neurons_per_session)  # total number of output neurons
        cum_neurons_per_session = np.cumsum([0] + self.num_neurons_per_session)
        self.neuron_indices_by_session = [
            torch.arange(cum_neurons_per_session[i], cum_neurons_per_session[i + 1]) for i in range(len(train_data) - 1)
        ]

        self.enc = []
        if self.partition is None:  # simple; just run all the inputs through the model
            Nhid = [self.Nin] + self.Nhid + [self.Nlat]  # concatenate stuff so our loop works
            for n, nhid in enumerate(Nhid[1:]):
                # weights for each layer
                setattr(
                    self,
                    f"Win_{n}",
                    torch.nn.Parameter(torch.randn(nhid, Nhid[n]) / np.sqrt(Nhid[n]), requires_grad=True),
                )
                # bias for each layer
                setattr(self, f"bin_{n}", torch.nn.Parameter(torch.zeros(nhid), requires_grad=True))
                self.enc.append((getattr(self, f"Win_{n}"), getattr(self, f"bin_{n}")))
        else:
            for ip, inds in enumerate(self.partition_input_indices):
                self.enc.append([])
                Nhid_p = [len(inds)] + self.Nhid + [self.Nlat]  # concatenate stuff so our loop works
                for n, nhid in enumerate(Nhid_p[1:]):
                    # weights for each layer
                    setattr(
                        self,
                        f"Win_{n}_{ip}",
                        torch.nn.Parameter(torch.randn(nhid, Nhid_p[n]) / np.sqrt(Nhid_p[n]), requires_grad=True),
                    )
                    # bias for each layer
                    setattr(self, f"bin_{n}_{ip}", torch.nn.Parameter(torch.zeros(nhid), requires_grad=True))
                    self.enc[ip].append((getattr(self, f"Win_{n}_{ip}"), getattr(self, f"bin_{n}_{ip}")))

        # define output parameters
        self.Wout = torch.nn.Parameter(torch.randn(self.Nout, self.Nlat) / np.sqrt(self.Nlat), requires_grad=True)
        # bias
        self.bout = torch.nn.Parameter(torch.zeros(self.Nout), requires_grad=True)

    def encode_channel(self, x, encoder):
        z = x
        for n, params in enumerate(encoder):
            # multiply by weight, add bias
            z = params[0] @ z + params[1][:, None]
            if n != len(encoder) - 1:  # currently ReLU for all non-terminal layers
                z = torch.relu(z)
        return z

    def encode(self, x):
        """
        x: input data
        """

        if self.partition is None:
            # no partitioning of input data, just pass everything through the encoder
            self.z = self.encode_channel(x, self.enc)
        else:
            # separately pass each input component through the appropriate embedding stream, then compute the (normalized) sum
            self.zs = torch.stack(
                [
                    self.encode_channel(x[inds, ...], self.enc[n])
                    for (n, inds) in enumerate(self.partition_input_indices)
                ]
            )

            self.z = torch.sum(
                self.zs,
                axis=0,
            ) / np.sqrt(len(self.partition))

        if self.latent_nonlin == "relu":
            z = torch.relu(z)

        return self.z

    def decode(self, z, neuron_inds=None):
        """
        z: latent representation
        neuron_inds: which neurons to decode

        Returns:
        yhat estimated firing rates
        """

        if neuron_inds is None:
            neuron_inds = np.arange(self.Nout)  # default to all

        # output matrix and bias for these neurons (NxM) or (#partitions x N x M)
        Weff, beff = self.Wout[..., neuron_inds, :], self.bout[..., neuron_inds]

        # predicted mean activity, passed through inv link function
        yhat = torch.exp(Weff @ z + beff[..., None])

        return yhat

    def loss(self, y, yhat, z, neuron_inds=None):
        """
        y: spike counts
        yhat: predicted spike counts
        z: latent representation
        neuron_inds: which neurons where decoded
        """
        if neuron_inds is None:
            neuron_inds = np.arange(self.Nout)  # default to all

        pred_loss = -torch.distributions.Poisson(yhat).log_prob(y).mean()  # predictive loss

        activity_loss = (z**2).mean()  # mean squared activity regularizer

        # weight regularization loss
        weight_loss = (self.Wout[..., neuron_inds, :] ** 2).mean() / (len(self.enc) + 1)
        if self.partition is None:
            for ps in self.enc:
                weight_loss += (ps[0] ** 2).mean() / (len(self.enc) + 1)
        else:
            for enc in self.enc:
                for ps in enc:
                    weight_loss += (ps[0] ** 2).mean() / (len(self.enc) + 1)

        # set losses for future access if needed
        self.pred_loss, self.activity_loss, self.weight_loss = (
            pred_loss,
            self.beta_act * activity_loss,
            self.beta_weight * weight_loss,
        )

        loss = self.pred_loss + self.activity_loss + self.weight_loss
        return loss

    def forward(self, x, y, neuron_inds=None):
        """
        compute full loss from a dataset
        x: input data
        y: spike counts
        neuron_inds: which neurons the spikes correspond to
        """
        self.neuron_inds = neuron_inds
        self.z = self.encode(x)  # calculate embedding
        self.yhat = self.decode(self.z, neuron_inds=neuron_inds)  # predict spikes from embedding
        self.tot_loss = self.loss(y, self.yhat, self.z, neuron_inds=neuron_inds)  # compute loss
        return self.tot_loss

    def score(self, x, y, **kwargs):
        # ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

        z = self.encode(x).detach().cpu().numpy()  # T x D

        # ensure y is a numpy array
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        return utils.eval_representation(z.T, y, **kwargs)

    def train(
        self,
        train_sessions,
        test_session=None,
        device=None,
        test_freq=5,
        lr=1e-3,
        nepochs=300,
        eval_alpha=1e-3,
        verbose=True,
    ):
        """training function"""
        # update X, spikes to torch

        self.set_input_groups(train_sessions)
        self.initialise_weights(train_sessions)

        # move below to model init?
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.device = device
        if verbose:
            print("device:", self.device)

        optim = torch.optim.Adam(self.parameters(), lr=lr)

        self.train_losses = []
        self.test_perfs = []
        self.train_perfs = []
        test_perf = np.nan

        train_Xs = [torch.tensor(session["X"], dtype=torch.float32).to(self.device) for session in train_sessions]
        train_spikes = [
            torch.tensor(session["spikes"], dtype=torch.float32).to(self.device) for session in train_sessions
        ]

        if test_session is not None:
            test_X, test_spikes = (
                torch.tensor(test_session["X"], dtype=torch.float32).to(self.device),
                test_session["spikes"],
            )

        t0 = time.time()
        for epoch in range(nepochs):

            epoch_losses = []
            for sesh_ind, neuron_inds in enumerate(self.neuron_indices_by_session):
                optim.zero_grad()
                loss = self.forward(train_Xs[sesh_ind], train_spikes[sesh_ind], neuron_inds=neuron_inds)
                loss.backward()
                optim.step()
                epoch_losses.append(loss.detach().cpu().numpy())

            if epoch % test_freq == 0:
                test_train_sesh = np.random.choice(len(train_sessions))
                if test_session is not None:
                    test_perf = np.nanmean(self.score(test_X, test_spikes))

                train_perf = np.nanmean(
                    self.score(train_Xs[test_train_sesh], train_spikes[test_train_sesh], alpha=eval_alpha)
                )

                self.train_losses.append(np.mean(epoch_losses))
                self.test_perfs.append(test_perf)
                self.train_perfs.append(train_perf)

                if verbose:
                    rate_loss, weight_loss = (
                        self.activity_loss.detach().cpu().numpy(),
                        self.weight_loss.detach().cpu().numpy(),
                    )
                    back = int(round(test_freq / 2))
                    test_perf, train_perf = np.mean(self.test_perfs[-back:]), np.mean(self.train_perfs[-back:])
                    print(
                        f"\nEpoch {epoch:>3} │ "
                        f"Test Perf:   {test_perf:>10.4g} │ "
                        f"Train Perf:  {train_perf:>10.4g} │ "
                        f"Train Loss:  {self.train_losses[-1]:>10.4g} │\n"
                        f"{'':>9}  │ "
                        f"Rate Loss:   {rate_loss:>10.4g} │ "
                        f"Weight Loss: {weight_loss:>10.4g} │ "
                        f"Time:        {(time.time() - t0):>10.4g}s│"
                    )
