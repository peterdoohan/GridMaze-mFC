import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from sklearn.metrics import mean_poisson_deviance
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from pyglmnet import GLM, simulate_glm
from .additive_multiplicative_regression import optimize, pred

# @title Specify the model


class Encoder(torch.nn.Module):
    """pytorch model for learning task embeddings"""

    def __init__(
        self,
        input_streams,
        Nhid,
        Nlat,
        Nout,
        beta_act=5e-2,
        beta_weight=5e-2,
        partition=None,
        latent_inputs=None,
        latent_nonlin=None,
        input_stream_names=None,
        inv_link="exp",
        noise_function="Poisson",
        sqrt_counts=None,
        combine_frs = False,
    ):
        """
        input_streams (list of int arrays): indices for each input stream. Use [np.arange(Nin)] to treat everything as one stream.
        Nhid (list of int): list of hidden layer dimensions (for each input stream)
        Nlat (int): latent dimension (output of the embedding streams)
        Nout (int): output dimension (total number of neurons)
        beta_act, beta_weight (floats): regularization parameters
        partition (list of int lists): optional partitioning of inputs into things that only combine linearly at the latent.
            each list specifies a set of input streams to be embedded together. Default: a single embedding including all input streams not in 'latent_inputs'
        latent_inputs (list of ints): input streams that should be appended directly to the latent instead of being embedded.
            if 'partition' is None, the latent_inputs are not included in the embedding by default. Specify both 'latent_inputs' and 'partition' to include an input at both stages.
            the total latent size is (Nlat + dim(latent_inputs))
        sqrt_counts: optionally take the sqrt of the spike counts -> this tends to improve performance for Gaussian models
        combine_frs: combine input streams in firing rate space instead of latent space (one of: [None, "additive", "multiplicative"])
        """
        super(Encoder, self).__init__()

        # convert input_streams to tensor
        input_streams = [torch.tensor(x) for x in input_streams]

        assert inv_link in {"exp", "identity", "softplus"}
        assert latent_nonlin in {None, "relu"}
        assert noise_function in {"Gaussian", "Poisson"}

        if sqrt_counts is None:
            self.sqrt_counts = noise_function == "Gaussian"
        else:
            self.sqrt_counts = sqrt_counts
        assert (not self.sqrt_counts) or (noise_function == "Gaussian")  # can only use sqrt counts with Gaussian noise

        # set the parameters
        self.Nhid, self.Nlat, self.Nout = Nhid, Nlat, Nout
        self.beta_act, self.beta_weight = beta_act, beta_weight
        self.input_stream_names = input_stream_names

        # set latent nonlinearity
        self.latent_nonlin = latent_nonlin
        
        # optionally combine streams at FR level (instead of latents)
        assert combine_frs in [False, "additive", "multiplicative"]
        self.combine_frs = combine_frs

        # potentially add some inputs directly to the latents
        if latent_inputs is None:
            self.latent_inputs = None
            self.Nlat_out = Nlat
        else:
            if type(latent_inputs) == int:
                latent_inputs = (latent_inputs,)
            elif all([type(inp_) == str for inp_ in latent_inputs]):
                # convert from strings to indices
                latent_inputs = [input_stream_names.index(inp_) for inp_ in latent_inputs]

            self.latent_inputs = np.concatenate([input_streams[ind] for ind in latent_inputs])
            self.Nlat_out = self.Nlat + len(self.latent_inputs)  # number of _output_ latent dimensions

        # potentially partition inputs into non-interacting streams
        if partition is None:
            if latent_inputs is None:  # everything just gets embedded together
                self.partition = None  # no partitioning
            else:  # default to everything not provided as direct latents
                self.partition = [
                    np.concatenate(
                        [input_streams[ind] for ind in range(len(input_streams)) if ind not in latent_inputs]
                    )
                ]
        else:
            self.partition = []
            for module in partition:  # get input indices for each block
                # when explicitly specified, an input stream can both be embedded and in latent inputs
                new_partition = [input_streams[ind] for ind in module]  # if ind not in latent_inputs]
                if len(new_partition) > 0:
                    self.partition.append(np.concatenate(new_partition))

        # specify number of embedding input dimensions
        if self.partition is None:  # no partitioning
            self.Nin = np.sum([len(inds) for inds in input_streams])
        else:
            self.Nin = np.sum([len(inds) for inds in self.partition])

        # initialize the parameters for the encoding part of the model

        self.enc = []
        if self.partition is None:  # simple; just run all the inputs through the model
            Nhid = [self.Nin] + Nhid + [Nlat]  # concatenate stuff so our loop works
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
            for ip, inds in enumerate(self.partition):
                self.enc.append([])
                Nhid_p = [len(inds)] + Nhid + [Nlat]  # concatenate stuff so our loop works
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

        if self.combine_frs:
            self.Wout = torch.nn.Parameter(
                torch.randn(len(self.partition), self.Nout, self.Nlat_out) / np.sqrt(self.Nlat_out), requires_grad=True
            )
            # maybe initialize this from data?
            self.bout = torch.nn.Parameter(torch.zeros(len(self.partition), self.Nout), requires_grad=True)
        else:
            # define output parameters
            self.Wout = torch.nn.Parameter(
                torch.randn(self.Nout, self.Nlat_out) / np.sqrt(self.Nlat_out), requires_grad=True
            )
            # maybe initialize this from data?
            self.bout = torch.nn.Parameter(torch.zeros(self.Nout), requires_grad=True)

        # inv link function
        if inv_link == "exp":
            self.inv_link = torch.exp
        elif inv_link == "identity":
            self.inv_link = lambda x: x
        elif inv_link == "softplus":
            self.inv_link = torch.nn.functional.softplus

        self.noise_function = noise_function
        if self.noise_function == "Gaussian":
            # consider learning the output noise parameters (but first check that this works)
            self.sigma_out = torch.nn.Parameter(torch.ones(self.Nout), requires_grad=False)
            assert inv_link == "identity"
            self.eval_function = self.eval_linear_gaussian
            self.eval_cv_function = self.eval_linear_gaussian
        elif self.noise_function == "Poisson":
            if inv_link == "exp":
                self.eval_function = self.eval_exp_poisson
                self.eval_cv_function = self.eval_exp_poisson_cv
            elif inv_link == "softplus":
                self.eval_function = self.eval_softplus_poisson
                self.eval_cv_function = self.eval_softplus_poisson
            else:
                raise NotImplementedError

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
            z = self.encode_channel(x, self.enc)
        else:
            # separately pass each input component through the appropriate embedding stream, then compute the (normalized) sum
            zs = torch.stack(
                    [self.encode_channel(x[inds, ...], self.enc[n]) for (n, inds) in enumerate(self.partition)]
                )
            
            if self.combine_frs: # combine in fr space!
                z = zs # maintain separate streams
            else:
                z = torch.sum(zs, axis=0,) / np.sqrt(len(self.partition))

        if self.latent_nonlin == "relu":
            z = torch.relu(z)

        if self.latent_inputs is not None:  # directly add some inputs to the latents
            if self.combined_frs:
                z = torch.concatenate([z,
                    x[None,self.latent_inputs, ...]+torch.zeros([len(z), len(self.latent_inputs), x.shape[-1]])], dim=1)
            else:
                z = torch.concatenate([z, x[self.latent_inputs, ...]], dim=0)

        return z

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
        yhat = self.inv_link(Weff @ z + beff[..., None])
        
        if self.combine_frs == "additive":
            yhat = yhat.sum(0)
        elif self.combine_frs == "multiplicative":
            yhat = yhat.prod(0)

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

        if self.sqrt_counts:
            y = y.sqrt()

        if self.noise_function == "Poisson":
            pred_loss = -torch.distributions.Poisson(yhat).log_prob(y).mean()  # predictive loss
        elif self.noise_function == "Gaussian":
            pred_loss = (
                -torch.distributions.Normal(yhat, self.sigma_out[neuron_inds, None]).log_prob(y).mean()
            )  # predictive loss

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
    
    def eval_comb(self, z_train, y_train, z_test, y_test, alpha = 1e-3, cv = None):
        Ws_f = optimize(z_train, y_train, type_ = self.combine_frs, seed = None, tol = 1e-4, verbose = False, alpha = alpha)
        yhat = pred(z_test, Ws_f, type_ = self.combine_frs)
        perfs_test = [calc_poisson_deviance(yhat[n, :], y_test[n, :]) for n in range(yhat.shape[0])]
        return perfs_test
    
    def eval_exp_poisson_cv(self, z_train, y_train, z_test, y_test, alphas = 10.0**np.arange(2,-5, -1), cv = 5, alpha = None):
        Tf = len(y_train)
        splits = np.round(np.linspace(0, Tf, cv + 1)).astype(int)
        inds = [np.arange(splits[i], splits[i + 1]) for i in range(cv)]  # contiguous split
        
        perfs = np.zeros((cv, len(alphas)))
        for fold in range(cv):
            test, train = inds[fold], np.concatenate([inds[f] for f in range(cv) if f != fold])
            zf_train, yf_train, zf_test, yf_test = z_train[train], y_train[train], z_train[test], y_train[test]
            for ialpha, alpha in enumerate(alphas):
                
                if ialpha == 0: # instantiate to begin with
                    clf = linear_model.PoissonRegressor(alpha=alpha, warm_start = True) # then continue from warm start
                else:
                    clf.alpha = alpha # just overwrite alpha

                clf.fit(zf_train, yf_train)
                perfs[fold, ialpha] = clf.score(zf_test, yf_test)

        mean_perfs = perfs.mean(0) # mean across folds for each alpha
        best_alpha = alphas[np.argmax(mean_perfs)] # best regularization strength
        #print(best_alpha)
        
        return self.eval_exp_poisson(z_train, y_train, z_test, y_test, alpha = best_alpha)

    def eval_exp_poisson(self, z_train, y_train, z_test, y_test, alpha = 1e-3, cv = None):
        clf = linear_model.PoissonRegressor(alpha=alpha)
        clf.fit(z_train, y_train)
        return clf.score(z_test, y_test)  # evaluate the goodness of the fit

    def eval_linear_gaussian(self, z_train, y_train, z_test, y_test, alpha = 1e-3, cv = None):
        clf = linear_model.Ridge(alpha=alpha)
        clf.fit(z_train, y_train)
        return clf.score(z_test, y_test)  # evaluate the goodness of the fit

    def eval_softplus_poisson(self, z_train, y_train, z_test, y_test, alpha = 1e-3, cv = None):

        clf = GLM(distr="softplus", score_metric="pseudo_R2", reg_lambda=alpha)  # define model
        clf.fit(z_train, y_train)  # fit model

        # their eval metric is slightly different from sklearn, so we compute the sklearn one ourselves instead
        yhat_test = clf.predict(z_test)  # prediction at test time
        return calc_poisson_deviance(yhat_test, y_test)  # goodness of prediction

    def eval_representation(self, x, y, cv=None, return_keep=False, alpha=1e-10, embed=True, trials = None, split = None):
        """
        function for evaluating the utility of the learned embedding on some dataset
        x: input data to be embedded
        y: output data to regress embedding onto
        """

        # first embed the input data
        if embed:
            z = self.encode(x).detach().cpu().numpy()  # T x D
        else:  # use the raw inputs
            # this won't work with combine_frs yet
            z = x.detach().cpu().numpy()

        if self.sqrt_counts:
            y = y.sqrt()
        y = y.cpu().numpy()

        if split is not None: cv = len(split) # optionally provide a split
        N, T = y.shape
        scores = np.zeros(N) if cv is None else np.zeros((N, cv))
        
        # for each neuron in the test data
        # y ~ Poisson( lambda = exp(W z) )
        
        if cv is None:
            keep = np.ones(N).astype(bool)
        if cv is not None:
            if split is not None: # provided splits
                inds = split
            elif (trials is None): # just generate contiguous splits
                splits = np.round(np.linspace(0, T, cv + 1)).astype(int)
                inds = [np.arange(splits[i], splits[i + 1]) for i in range(cv)]  # contiguous split
            else: # split by trials
                inds = split_trials(trials, cv)
            # require spikes in all splits
            keep = np.array([(np.amin([y[n, :][ind].sum() for ind in inds]) > 0) for n in range(N)]) 
        
        if self.combine_frs:
            if cv is None:
                scores[:] = np.array(self.eval_comb(z, y, z, y, alpha = alpha, cv = cv))
            else:
                y = y[keep, :]
                for fold in range(cv):
                    test, train = inds[fold], np.concatenate([inds[f] for f in range(cv) if f != fold])
                    scores[keep, fold] = self.eval_comb(z[..., train], y[:, train], z[..., test], y[:, test], alpha = alpha, cv = cv)
        else:
            z = z.T
            for n in range(N):
                y_n = y[n, :]  # target spike counts
                # fit a Poisson regression model from the embeddings
                if cv is None:  # no crossvalidation; just test representation on the whole thing
                    scores[n] = self.eval_function(z, y_n, z, y_n, alpha = alpha)
                elif keep[n]:
                    for fold in range(cv):
                        test, train = inds[fold], np.concatenate([inds[f] for f in range(cv) if f != fold])
                        scores[n, fold] = self.eval_cv_function(z[..., train, :], y_n[train], z[..., test, :], y_n[test], alpha = alpha, cv = cv)

        if return_keep:
            return scores[keep], keep
        else:
            return scores[keep]

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

def split_trials(trials, cv):
    unique_trials = np.unique(trials)
    trial_splits = [[] for _ in range(cv)]
    for trial in unique_trials:
        trial_splits[int(trial) % cv].append(trial)
    inds = [np.concatenate([np.where(trials == trial_id)[0] for trial_id in trial_split]) for trial_split in trial_splits]
    return inds        

def calc_poisson_deviance(yhat, y):
    test_dev = mean_poisson_deviance(y, yhat)
    ref_dev = mean_poisson_deviance(y, np.ones(len(y)) * np.mean(y))
    return 1 - test_dev / ref_dev


# just check that all of the class methods we've implemented vaguely work


def sanity_checks(sessions):
    Nin = sessions[0]["X"].shape[0]
    Ntot = sum([sesh["spikes"].shape[0] for sesh in sessions])
    Nhid = [100, 50]
    Nlat = 20
    print(f"Nin: {Nin}, Nhid: {Nhid}, Nlat: {Nlat}, Ntot: {Ntot}")

    model = Encoder(Nin, Nhid, Nlat, Ntot)
    session = sessions[1]

    print("\n\nModel params:")
    for ip, param in enumerate(model.parameters()):
        print(ip, param.shape)

    print(model.Win_0.shape, (model.Win_0 @ sessions[0]["X"]).shape, model.bin_0.shape)

    z = model.encode(session["X"])
    print(z.shape, z.min(), z.max())

    yhat = model.decode(z, neuron_inds=session["cluster_inds"])
    print(yhat.shape, yhat.min(), yhat.max())

    print(torch.distributions.Poisson(yhat).log_prob(session["spikes"]).shape)
    print((z**2).shape)
    print((model.Win_0**2).shape)

    loss = model.loss(session["spikes"], yhat, z, neuron_inds=session["cluster_inds"])

    print(loss, model.pred_loss, model.activity_loss, model.weight_loss)

    print(model.forward(session["X"], session["spikes"], neuron_inds=session["cluster_inds"]))

    print(np.mean(model.eval_representation(session["X"], session["spikes"])))


# @title Define training function
def train_model(
    model,
    train_sessions,
    test_session=None,
    device=None,
    test_freq=10,
    Print=True,
    lr=1e-2,
    nepochs=300,
    eval_alpha=1e-10,
):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_perfs = []
    train_perfs = []
    test_perf = np.nan

    train_Xs = [session["X"].to(device) for session in train_sessions]
    train_spikes = [session["spikes"].to(device) for session in train_sessions]

    if test_session is not None:
        test_X, test_spikes = test_session["X"].to(device), test_session["spikes"].to(device)

    t0 = time.time()
    for epoch in range(nepochs):

        epoch_losses = []
        for sesh_ind, session in enumerate(train_sessions):
            optim.zero_grad()
            loss = model(train_Xs[sesh_ind], train_spikes[sesh_ind], neuron_inds=session["cluster_inds"])
            loss.backward()
            optim.step()
            epoch_losses.append(loss.detach().cpu().numpy())

        if epoch % test_freq == 0:
            test_train_sesh = np.random.choice(len(train_sessions))
            if test_session is not None:
                test_perf = np.mean(model.eval_representation(test_X, test_spikes))

            train_perf = np.mean(
                model.eval_representation(train_Xs[test_train_sesh], train_spikes[test_train_sesh], alpha=eval_alpha)
            )
            train_losses.append(np.mean(epoch_losses))
            test_perfs.append(test_perf)
            train_perfs.append(train_perf)

            if Print:
                print(
                    "\n",
                    epoch,
                    train_losses[-1],
                    model.activity_loss.detach().cpu().numpy(),
                    model.weight_loss.detach().cpu().numpy(),
                )
                back = int(round(test_freq / 2))
                print(np.mean(test_perfs[-back:]), np.mean(train_perfs[-back:]), time.time() - t0)

    return model, train_losses, test_perfs, train_perfs
