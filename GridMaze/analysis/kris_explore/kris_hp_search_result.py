
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

base_dir = "/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/grid_search/"

params = pickle.load(open(f"{base_dir}hp_search_params.p", "rb"))


all_Nhids = set([tuple(p["Nhid"]) for p in params])
all_betas = set([tuple(p["betas"]) for p in params])
all_lrs = np.unique([p["lr"] for p in params])
all_Nlats = np.unique([p["Nlat"] for p in params])

Nhids = {Nhid: i for (i, Nhid) in enumerate(all_Nhids)}
betas = {beta: i for (i, beta) in enumerate(all_betas)}
lrs = {lr: i for (i, lr) in enumerate(all_lrs)}
Nlats = {Nlat: i for (i, Nlat) in enumerate(all_Nlats)}

cv_perfs = np.zeros((len(all_Nhids), len(all_betas), len(all_lrs), len(all_Nlats), 10, 10)) + np.nan # model fit sessions by test sessions
tt_perfs = np.zeros((len(all_Nhids), len(all_betas), len(all_lrs), len(all_Nlats), 10, 5)) + np.nan

results = [f for f in os.listdir(base_dir) if "result" in f]

for f in results:
    result = pickle.load(open(f"{base_dir}{f}", "rb"))
    ps = result["param"]
    
    inds = [Nhids[tuple(ps["Nhid"])], betas[tuple(ps["betas"])], lrs[ps["lr"]], Nlats[ps["Nlat"]]]
    cv_perfs[inds[0], inds[1], inds[2], inds[3], ...] = np.array(result["cv_perf"])
    
    tt_perfs[inds[0], inds[1], inds[2], inds[3], ...] = np.array(result["test_perfs"])
    

meancvs = np.mean(cv_perfs, axis = -2) # average over test sessions
cols = [np.array(plt.get_cmap("tab10")(i)[:-1]) for i in range(3)]
scales = [0.5, 0.7, 1.0]

# plot final crossvalidated test performance
for idata, data in enumerate([meancvs[..., 0], meancvs[..., 1:].mean(-1)]):

    ymin, ymax = np.nanmin(data), np.nanmax(data)*1.02
    
    fig, axs = plt.subplots(1,data.shape[0], figsize = (8, 2.5)) # hidden architectures
    xs = all_Nlats
    for iax, ax in enumerate(axs):
        for ibeta in range(data.shape[1]):
            for ilr in range(data.shape[2]):
                if ibeta == 2 and iax == 0:
                    label = all_lrs[ilr]
                elif ilr == 0 and iax == 1:
                    label = [str(beta) for (beta, i) in betas.items() if i == ibeta][0]
                else:
                    label = None
                    
                ax.plot(xs, data[iax, ibeta, ilr, :], color = scales[ibeta]*cols[ilr], ls = "-", label = label)

        ax.set_xlabel("# latents")
        if iax == 0:
            ax.set_ylabel("cv performance")
        else:
            ax.set_yticks([])
        ax.set_ylim(ymin, ymax)
        ax.set_title([Nhid for (Nhid, i) in Nhids.items() if i == iax][0])
        
        if iax != 2:
            ax.legend()
        
    plt.tight_layout()
    plt.savefig(f"{base_dir}visual_summary{idata}.png", bbox_inches = "tight")
    plt.close()

# plot training curve of non-crossvalidated test representation
    
fig, axs = plt.subplots(tt_perfs.shape[3], tt_perfs.shape[0], figsize = (8, 15)) # hidden architectures
for ilat in range(tt_perfs.shape[3]):
    data = tt_perfs[..., ilat, :, :].mean(-2)
    ymin, ymax = np.nanmin(data), np.nanmax(data)*1.02
    
    xs = [0, 500, 1000, 1500, 2000]
    for iax in range(len(Nhids)):
        ax = axs[ilat, iax]
        for ibeta in range(data.shape[1]):
            for ilr in range(data.shape[2]):
                if ibeta == 2 and iax == 0:
                    label = all_lrs[ilr]
                elif ilr == 0 and iax == 1:
                    label = [str(beta) for (beta, i) in betas.items() if i == ibeta][0]
                else:
                    label = None
                    
                ax.plot(xs, data[iax, ibeta, ilr, :], color = scales[ibeta]*cols[ilr], ls = "-", label = label)

        ax.set_xlabel("train epoch")
        if iax == 0:
            ax.set_ylabel("test representation")
        else:
            ax.set_yticks([])
        ax.set_ylim(ymin, ymax)
        
        ax.set_title(str([Nhid for (Nhid, i) in Nhids.items() if i == iax][0]) + " | " + str(all_Nlats[ilat]))
        
        if (ilat == 0) and (iax != 2):
            ax.legend()
        
plt.tight_layout()
plt.savefig(f"{base_dir}visual_train_summary.png", bbox_inches = "tight")
plt.close()


        