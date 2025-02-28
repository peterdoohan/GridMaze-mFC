
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from sklearn.metrics import mean_poisson_deviance
from scipy.stats import pearsonr
from embedding_utils import Encoder, sanity_checks, train_model


#@title Generate synthetic data

N, T = 100, 5000 # number of neurons and timepoints per session
num_sesh = 10 # number of sessions
num_dist_bins = 10 # binning resolution for distance-to-goal
side_length = 7 # side length of the arena
num_states = side_length**2 # number of states in the arena (this would be number of state-actions)


sessions = [] # list of session dicts
first_ind = 0 # first neuron index in this session
for seshnum in range(num_sesh): # for each session

  # sample locations and distances to goal randomly
  locs = torch.tensor(np.random.choice(torch.arange(num_states), T, replace = True)) # locations
  dists = torch.tensor(np.random.choice(torch.arange(num_dist_bins), T, replace = True)) # distance to goals
  # generate model input data (concatenated 1hot representations of locations and dist-to-goal)
  X = torch.cat([torch.nn.functional.one_hot(locs, num_classes=num_states), torch.nn.functional.one_hot(dists, num_classes=num_dist_bins)], axis = -1).float().T

  # convert locations from indices to actual locations in Euclidean space
  euclid_locs = torch.stack([locs % side_length, locs // side_length]).T # T x 2

  # preferred location and distance-to-goal of each neuron (assume Gaussian tunings in both spaces)
  pref_locs = torch.rand(N, 2)*(side_length-1) # location (sample uniformly in [0, side_length-1])
  pref_dists = torch.rand(N)*(num_dist_bins-1) # distance to goal

  # compute firing rates
  frs = torch.exp(-0.5*torch.sum((pref_locs[None, ...] - euclid_locs[:, None, :])**2, axis = -1))
  frs *= torch.exp(- 0.5*(pref_dists[None, :]-dists[:, None])**2)
  frs *= 4 # scale factor (increasing this increases signal-to-noise)

  # sample spike counts
  spikes = torch.poisson(frs).T

  # print some summary stuff
  print(f"{seshnum} optimal:", -torch.distributions.Poisson(frs.T).log_prob(spikes).mean())
  print(frs.shape, torch.amin(frs), torch.amax(frs), X.shape, first_ind)

  # what are the indices in the global set of neurons for this session?
  inds = np.arange(spikes.shape[0])+first_ind
  first_ind = inds[-1]+1 # first index for the next session

  # store session data
  sessions.append({"X": X, "spikes": spikes, "frs": frs, "locs": locs, "dists": dists, "pref_locs": pref_locs,
                   "pref_dists": pref_dists, "inds": inds, "euclid_locs": euclid_locs})

Nin = sessions[0]["X"].shape[0]
Ntot = sum([sesh["spikes"].shape[0] for sesh in sessions])


#@title Sanity check synthetic data

# generate some empirical tuning curves
tunings = torch.zeros(num_sesh, num_dist_bins, side_length, side_length, N)+torch.nan
for seshnum in range(num_sesh): # for each session
  session = sessions[seshnum]
  for d in range(num_dist_bins): # for each location/dist bin
    for x in range(side_length):
      for y in range(side_length):
        # find time points where we were in that bin
        inds = np.where((session["dists"] == d) & (session["euclid_locs"][:, 0] == x) & (session["euclid_locs"][:,1] == y))[0]
        if len(inds) > 0: # if there are some
          ind_frs = session["spikes"][:, inds] # compute mean FR across these bins
          assert len(ind_frs.shape) == 2
          tunings[seshnum, d, x, y, :] = torch.mean(ind_frs, axis = -1)

# plot tuning curve for an example neuron
neuron_num = 33
sesh_num = 5

# we plot a spatial heatmap for each distance-to-goal
fig, axs = plt.subplots(2,5, figsize = (10, 5))
for d in range(num_dist_bins):
  ax = axs[d//5, d%5]
  ax.imshow(tunings[sesh_num, d, :, :, neuron_num], vmin = 0, vmax = 4, cmap = "viridis")
plt.tight_layout()
plt.savefig("./figs/test.png", bbox_inches = "tight")
plt.close()


sanity_checks()


#@title Evaluate generalization performance

Nlat = 8
all_losses = []
all_test_perfs = []
all_train_perfs = []
import warnings
warnings.filterwarnings("ignore")
Nhid = [100, 50]
for sesh in range(num_sesh):
  print("\n\n")
  print("------------------------------------------------------")
  print(f"NEW SESSION: {sesh+1} of {num_sesh}\n")
  np.random.seed(sesh)
  torch.manual_seed(sesh)

  model = None
  torch.cuda.empty_cache()

  model = Encoder(Nin, Nhid, Nlat, Ntot, beta_act = 1e-2, beta_weight = 1e-2)
  train_sessions = sessions[:sesh]+sessions[sesh+1:]
  test_sesssion = sessions[sesh]
  model, train_losses, test_perfs, train_perfs = train_model(model, train_sessions, test_sesssion, nepochs = 100, test_freq = 10)

  all_losses.append(train_losses)
  all_test_perfs.append(test_perfs)
  all_train_perfs.append(train_perfs)


#@title Plot generalization performance

np_losses = np.array(all_losses)
np_perfs = np.array(all_test_perfs)
np_train_perfs = np.array(all_train_perfs)

datas = [np_losses, np_train_perfs, np_perfs]
labels = ["train loss", "train data", "test data"]

ideal_scores = []
for session in sessions:
  ideal_scores.append(calc_poisson_deviance(session["frs"], session["spikes"]))
print(ideal_scores)

plt.figure(figsize = (3,2))
for idata, data in enumerate(datas):
  m, s = np.mean(data, axis = 0), np.std(data, axis = 0)/np.sqrt(data.shape[0])
  xs = np.arange(len(m))
  plt.plot(xs, m, label = labels[idata])
  plt.fill_between(xs, m-s, m+s, alpha = 0.2)
m, s = np.ones(2)*np.mean(ideal_scores), np.ones(2)*np.std(ideal_scores)/np.sqrt(len(ideal_scores))
x = [xs[0], xs[-1]]
plt.plot(x, m, color = "k", label = "optimal")
plt.fill_between(x, m-s, m+s, alpha = 0.2, color = "k")

plt.xlim(xs[0], xs[-1])
plt.ylim(0, 0.8)
plt.xlabel("training epoch")
plt.ylabel("value")
plt.legend()
plt.savefig("./test_gen.png", bbox_inches = "tight")
plt.close()


#@title Fit a model to the full dataset

Nhid = [100, 50]
Nlat = 10
Nin = sessions[0]["X"].shape[0]
Ntot = sum([sesh["spikes"].shape[0] for sesh in sessions])

t = time.time()
np.random.seed(0)
torch.manual_seed(0)
print(len(sessions))
model_full = Encoder(Nin, Nhid, Nlat, Ntot, beta_act = 1e-2, beta_weight = 1e-2)
model_full, train_losses, test_perfs, train_perfs = train_model(model_full, sessions, test_session = None, nepochs = 800,
                                                           lr = 1e-2, test_freq = np.nan)
device = model_full.Wout.device
print(time.time() - t)
ind = 4

test_perf = np.mean(model.eval_representation(sessions[ind]["X"].to(device), sessions[ind]["spikes"].to(device)))
opt_perf = calc_poisson_deviance(sessions[ind]["frs"], sessions[ind]["spikes"])
print(test_perf, opt_perf)

#@title Plot the latents

all_locs = torch.arange(num_states)
all_dists = torch.arange(num_dist_bins)

all_X = torch.zeros(Nin, num_states*num_dist_bins)
all_loc_dists = torch.zeros(all_X.shape[-1], 2)
for loc in all_locs:
  for dist in all_dists:
    ind = loc*num_dist_bins+dist
    all_X[loc, ind] = 1.
    all_X[num_states+dist, ind] = 1.
    all_loc_dists[ind, :] = torch.tensor([loc, dist])

print(all_X.shape)

all_z = model_full.encode(all_X.to(model_full.Wout.device)).detach().cpu().numpy()
print(all_z.shape)
print(all_z.min(), all_z.max(), all_z.std(-1).mean())

for n in range(all_z.shape[0]):
  fig, axs = plt.subplots(2,5, figsize = (10, 4))
  for d in range(num_dist_bins):
    ax = axs[d//5, d%5]
    ax.imshow(all_z[n, d::num_dist_bins].reshape(side_length, side_length), cmap = "viridis")
    ax.set_xticks([])
    ax.set_yticks([])
  plt.tight_layout()
  plt.savefig(f"./figs/latents/test{n}.png", bbox_inches = "tight")
  plt.close()
  print()



sims = np.zeros((Nlat, Nlat)) + np.nan
for i in range(Nlat):
  for j in range(i+1, Nlat):
    cor = pearsonr(all_z[i, :], all_z[j, :])[0]
    sims[i, j] = cor
    sims[j, i] = cor

plt.figure()
plt.imshow(sims, vmin = -1, vmax = 1)
plt.xticks([])
plt.yticks([])
plt.savefig("./figs/test_cor.png", bbox_inches = "tight")
plt.close()



