### ADD A BIAS ###

import numpy as np
import copy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import pickle

def pred_add(Xs, Ws):
    """run the additive model forward pass"""
    f = np.sum(np.array([np.exp(Ws[i]@Xs[i]) for i in range(len(Xs))]), axis = 0)
    return f

def pred_mul(Xs, Ws):
    """run the multiplicative model forward pass"""
    f = np.exp(np.sum(np.array([Ws[i]@Xs[i] for i in range(len(Xs))]), axis = 0))
    return f

def pred(Xs, Ws, type_):
    if type_ == "additive":
        return pred_add(Xs, Ws)
    elif type_ == "multiplicative":
        return pred_mul(Xs, Ws)
    else:
        raise NotImplementedError
    

def calc_loss(f, y, Ws = None, alpha = None):
    """
    compute the model loss for a given set of predictions and targets
    f are predictions, y are counts
    alpha is reg strength, Ws are parameters
    """
    
    # loss = -np.mean(y*np.log(f) - f) # negative log likelihood
    loss = -np.sum(y*np.log(f) - f)/y.shape[-1] # negative log likelihood

    if alpha is not None: # optional regularization
        #loss += 0.5 * alpha * np.sum([np.mean(W**2) for W in Ws])
        loss += 0.5 * alpha * np.sum([np.sum(W**2) for W in Ws])
        
    return loss

def grad_pred_add(Xs, Ws, y):
    
    fis = [np.exp(Ws[i]@Xs[i]) for i in range(len(Xs))] # each term
    
    f = np.sum(np.array(fis), axis = 0) # (N x T)
    #g = (1 - y/f) / y.size # dLpred / df (N x T)
    g = (1 - y/f) / y.shape[-1] # dLpred / df (N x T)
    
    gs = [ (g * fis[i]) @ Xs[i].T for i in range(len(Xs))]
    
    return gs

def grad_pred_mul(Xs, Ws, y):
    
    f = pred_mul(Xs, Ws) # (N x T)
    #g = (f-y) / y.size
    g = (f-y) / y.shape[-1]
    gs = [ g @ Xs[i].T for i in range(len(Xs))]
    
    return gs


def grad_pred(Xs, Ws, y, type_ = "additive"):
    """compute gradients of the predictive loss"""
    
    # first compute the predictions
    if type_ == "additive":
        return grad_pred_add(Xs, Ws, y)
        #f = pred_add(Xs, Ws) # (N x T)
    elif type_ == "multiplicative":
        return grad_pred_mul(Xs, Ws, y)
        #f = pred_mul(Xs, Ws) # (N x T)
    else:
        raise NotImplementedError
    
    return gs

def flatten(Ws):
    return np.concatenate([W.flatten() for W in Ws])

def unflatten(theta, Nneuron, Nins):
    Ws = []
    ind = 0
    for Nin in Nins:
        Ws.append(theta[ind:ind+Nin*Nneuron].reshape(Nneuron, Nin))
        ind += Nin*Nneuron
        
    return Ws


def forward(Xs, theta, y, type_ = "additive", alpha = None):
    Nneuron, Nins = y.shape[0], [X.shape[0] for X in Xs]
    Ws = unflatten(theta, Nneuron, Nins)
    
    f = pred(Xs, Ws, type_)
    loss = calc_loss(f, y, Ws = Ws, alpha = alpha)
    
    return loss

def grad(Xs, theta, y, type_ = "additive", alpha = None):
    Nneuron, Nins = y.shape[0], [X.shape[0] for X in Xs]
    Ws = unflatten(theta, Nneuron, Nins)
    gs = grad_pred(Xs, Ws, y, type_ = type_)
    
    if alpha is not None:
        for i in range(len(gs)):
            #gs[i] += alpha/Ws[i].size*Ws[i]
            gs[i] += alpha*Ws[i]
    
    return flatten(gs)


def cb(intermediate_result = None):
    print(intermediate_result.fun)

def optimize(Xs, y, seed = None, type_ = "additive", alpha = 1e-3, tol = 1e-4, verbose = True, method = "L-BFGS-B"):
    if seed is not None:
        np.random.seed(seed)
    
    Nneuron, Nins = y.shape[0], [X.shape[0] for X in Xs]
    Ws0 = [np.random.normal(0, 1, (Nneuron, X.shape[0]))/np.sqrt(X.shape[0])/2 for X in Xs]
    
    fclos = lambda theta : forward(Xs, theta, y, type_ = type_, alpha = alpha)
    jac = lambda theta : grad(Xs, theta, y, type_ = type_, alpha = alpha)
    
    opt_res = minimize(fclos, flatten(Ws0), callback = (cb if verbose else None), method = method, jac = jac, tol = 1e-4)

    Ws_f = unflatten(opt_res.x, Nneuron, Nins)
    
    return Ws_f

if __name__ == "__main__":
    
    import GridMaze.analysis.embedding_model.get_input_data
    from GridMaze.analysis.embedding_model.get_input_data import get_input_data
    from GridMaze.analysis.embedding_model.embedding_utils import split_trials, calc_poisson_deviance

    Nin, Nneuron, T = 100, 50, 1000

    X1, X2 = [np.random.normal(0, 1, (Nin, T)) for _ in range(2)]
    W1, W2 = [np.random.normal(0, 1, (Nneuron, Nin))/np.sqrt(Nin)/2 for _ in range(2)]

    F1, F2 = np.exp(W1 @ X1), np.exp(W2 @ X2)

    Ya = np.random.poisson(F1+F2)
    Yp = np.random.poisson(F1*F2)
    
    
    for pred_f in [pred_add, pred_mul]:
        f = pred_f([X1, X2], [W1, W2])
        for ytest in [Ya, Yp]:
            print(calc_loss(f, ytest))
        
    print()
    Ws_f = optimize([X1, X2], Yp, type_ = "multiplicative", seed = 0)

    for iy, y in enumerate([Ya, Yp]):
        for type_ in ["additive", "multiplicative"]:
            print("\n", type_, ["Ya", "Yp"][iy])
            optimize([X1, X2], y, type_ = type_, seed = 0)
            
            
    # compare how models perform for different mixing ratios
    ratios = np.linspace(0, 1, 21)
    reps = 10
    ress = np.zeros((2, len(ratios), reps))
    Nneuron, Nins, T = 70, [100, 20], 5000
    train_inds, test_inds = np.arange(int(T/2)), np.arange(int(T/2), T)
    for itype, type_ in enumerate(["additive", "multiplicative"]):
        for irat, ratio in enumerate(ratios):
            for rep in range(reps):
                Xs = [np.zeros((Nin, T)) for Nin in Nins]
                for i1 in range(2):
                    for i2 in range(T):
                        Xs[i1][np.random.choice(Nins[i1]), i2] = 1 # random one-hots
                Ws = [np.random.normal(0, 1, (Nneuron, Nin))/2 for Nin in Nins]
                        
                # Xs = [np.random.normal(0, 1, (Nin, T)) for Nin in Nins]
                # Ws = [np.random.normal(0, 1, (Nneuron, Nin))/np.sqrt(Nin)/2 for Nin in Nins]
                
                F1, F2 = np.exp(Ws[0] @ (Xs[0]+np.random.normal(0, 1e-10, Xs[0].shape))), np.exp(Ws[1] @ (Xs[1]+np.random.normal(0, 1e-10, Xs[1].shape)))
                F = ratio*(F1+F2)*6/3 + (1-ratio)*(F1*F2)*0.80
                F /= 1
                
                y = np.random.poisson( F )
                if rep == 0: print()
                
                Ws_f = optimize([X[:, train_inds] for X in Xs], y[:, train_inds], type_ = type_, seed = itype*10000+irat*100+rep, verbose = False, alpha = 1e-3)
                yhat = pred(Xs, Ws_f, type_)
                
                perfs_test = [calc_poisson_deviance(yhat[n, test_inds], y[n, test_inds]) for n in range(Nneuron)]
                
                ress[itype, irat, rep] = np.mean(perfs_test)
            print(type_, ratio, np.mean(ress[itype, irat, :]), np.mean(F), np.std(F))
            
    m, s = ress.mean(-1), ress.std(-1)/np.sqrt(reps)
    plt.figure()
    for i in range(2):
        plt.plot(ratios, m[i], label = ["additive", "multiplicative"][i])
        plt.fill_between(ratios, (m-s)[i], (m+s)[i], alpha = 0.2)
    #plt.ylim(np.amin(m-s), np.amax(m+s))
    plt.legend()
    plt.xlabel("% additive")
    plt.ylabel("Poisson deviance")
    plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/additive_vs_mult_synthetic.png", bbox_inches = "tight")
    plt.close()


    ### okay now try on some actual mouse data ###

    input_features=["distance", "place_direction"]#, "tower_bridge"]
    cv = 4
    alphas = 10.0**np.arange(-5,1, 1)
    t0 = time.time()

    subjects = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
    for mouse in subjects:

        sessions = get_input_data(subject_IDs=[mouse],
            maze_name="maze_1",
            input_features=input_features,
            distance_metrics=("distance_to_goal", "geodesic"),
            include_multi_unit=False,
            navigation_only=True,
            moving_only=True,
            resolution=0.1,  # s
            max_distance=1.8,  # m
            n_distance_bins=20,
            min_spike_count = 300,)


        mouse_scores = []

        for isesh, session in enumerate(sessions):

            input_streams = session["X_type_inds"]
            input_stream_names = session["input_feature_names"]

            Xs = [session["X"][stream].numpy() for stream in input_streams]
            Xs = [np.concatenate([X, np.ones((1, X.shape[-1]))], axis = 0) for X in Xs]

            y = session["spikes"].numpy()
            trials = session["trial_ids"]

            inds = split_trials(trials, cv)

            minspikes = np.amin(np.array([np.sum(y[:, ind], axis = -1) for ind in inds]), axis = 0)
            y = y[minspikes > 5, :]
            Nins, Nout, T = [X.shape[0] for X in Xs], y.shape[0], y.shape[1]

            scores = np.zeros((2,len(alphas),  Nout, cv))

            for ialpha, alpha in enumerate(alphas):
                for itype, type_ in enumerate(["additive", "multiplicative"]):
                    for fold in range(cv):
                        test, train = inds[fold], np.concatenate([inds[f] for f in range(cv) if f != fold])
                        Ws_f = optimize([X[:, train] for X in Xs], y[:, train], type_ = type_, seed = fold, verbose = False, alpha = alpha)
                        
                        
                        yhat = pred(Xs, Ws_f, type_)
                        
                        perfs_test = [calc_poisson_deviance(yhat[n, test], y[n, test]) for n in range(Nout)]
                        perfs_train = [calc_poisson_deviance(yhat[n, train], y[n, train]) for n in range(Nout)]
                        
                        scores[itype, ialpha, :, fold] = np.array(perfs_test)
                        
                    print(f"-----{mouse} {isesh} {alpha} {type_}: {np.mean(scores[itype, ialpha, ...])}, {np.round((time.time()-t0)/60, 1)}------")
                    
            mouse_scores.append(scores)
            
        pickle.dump({"scores": mouse_scores, "alphas": alphas}, open(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/add_vs_mul/{mouse}.p", "wb"))

    subjects = ['m2', 'm3', 'm4', 'm6', 'm7', 'm8']
    all_scores = [pickle.load(open(f"/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/add_vs_mul/{mouse}.p", "rb")) for mouse in subjects]
    alphas = all_scores[0]["alphas"]
    # avg across neurons+folds+sessions
    scores = np.array([np.array([sesh.mean((-1, -2)) for sesh in score["scores"]]).mean(0) for score in all_scores]) # (subjects x model_type x alpha)

    m, s = scores.mean(0), scores.std(0) / np.sqrt(scores.shape[0])

    mean_per_animal = scores.mean(1)[:, None, :]
    s = (scores - mean_per_animal).std(0) / np.sqrt(scores.shape[0])

    xs = np.log10(alphas)
    plt.figure()
    for i in range(2):
        plt.plot(xs, m[i], label = ["additive", "multiplicative"][i])
        plt.fill_between(xs, (m-s)[i], (m+s)[i], alpha = 0.2)
    plt.ylim(0.02, np.amax(m+s))
    plt.legend()
    plt.xlabel("log10 reg strength")
    plt.ylabel("Poisson deviance")
    plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/additive_vs_mult_all_animals.png", bbox_inches = "tight")
    plt.close()


    ### plot result ###

    # mean_scores = scores.mean(-1) # mean across folds
    # ms, ss = mean_scores.mean(-1), mean_scores.std(-1) / np.sqrt(mean_scores.shape[-1])

    # xs = [0, 1]
    # plt.figure(figsize = (2,3))
    # plt.bar(xs, ms, yerr = ss)
    # plt.xticks(xs, ["additive", "multiplicative"], rotation = 45, ha = "right")
    # plt.ylabel("accuracy")
    # plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/additive_vs_multiplicative.png", bbox_inches = "tight")
    # plt.close()

    # diffs = mean_scores[1] - mean_scores[0]
    # m, s = np.mean(diffs), np.std(diffs)/np.sqrt(len(diffs))
    # xs = [0]
    # plt.figure(figsize = (1,3))
    # plt.bar(xs, m, yerr = s)
    # plt.xticks([])
    # plt.ylabel("delta")
    # plt.savefig("/ceph/behrens/peter_doohan/goalNav_mFC_refactor/results/nn_dim_red/misc/figs/additive_vs_multiplicative_diff.png", bbox_inches = "tight")
    # plt.close()


    #### compare to scipy poisson regression ###


    # load some data

    sessions = get_input_data(subject_IDs=["m2"],
            maze_name="maze_1",
            input_features=["distance", "place_direction"],
            distance_metrics=("distance_to_goal", "geodesic"),
            include_multi_unit=False,
            navigation_only=True,
            moving_only=True,
            resolution=0.1,  # s
            max_distance=1.8,  # m
            n_distance_bins=20,
            min_spike_count = 300,)
    session = sessions[4]

    # parse into correct form
    input_streams = session["X_type_inds"]
    input_stream_names = session["input_feature_names"]

    Xs = [session["X"][stream].numpy() for stream in input_streams]
    Xs2 = [np.concatenate([X, np.ones((1, X.shape[-1]))], axis = 0) for X in Xs]
    Xsp = np.concatenate(Xs).T

    cv = 5
    y = session["spikes"].numpy()
    trials = session["trial_ids"]
    inds = split_trials(trials, cv)
                        
    # fit my home-brew implementation

    from sklearn import linear_model
    fold = 3
    alpha = 1e-3
    tol = 1e-4
    test, train = inds[fold], np.concatenate([inds[f] for f in range(cv) if f != fold])

    t0 = time.time()
    type_ = "multiplicative"
    Ws_f = optimize([X[:, train] for X in Xs2], y[:, train], type_ = type_, seed = None, tol = tol, verbose = True, alpha = alpha)
    yhat = pred(Xs2, Ws_f, type_ = type_)
    perfs_test = [calc_poisson_deviance(yhat[n, test], y[n, test]) for n in range(y.shape[0])]
    t1 = time.time()

    # fit scipy
    scores = []
    for n in range(y.shape[0]):
        clf = linear_model.PoissonRegressor(alpha=alpha)
        clf.fit(Xsp[train, :], y[n, train])
        scores.append(clf.score(Xsp[test, :], y[n, test]))
    t2 = time.time()

    # compare results
    from scipy.stats import pearsonr

    print(pearsonr(perfs_test, scores))
    print(np.mean(perfs_test), np.mean(scores))
    print(t1-t0, t2-t1)


    # try repeating the fit to see how internally consistent it is

    Ws_f2 = optimize([X[:, train] for X in Xs2], y[:, train], type_ = type_, seed = None, tol = tol, verbose = False, alpha = alpha)
    yhat2 = pred(Xs2, Ws_f2, type_ = type_)
    perfs_test2 = [calc_poisson_deviance(yhat2[n, test], y[n, test]) for n in range(y.shape[0])]
    print(pearsonr(perfs_test, perfs_test2))
    print(np.mean(perfs_test), np.mean(perfs_test2))


