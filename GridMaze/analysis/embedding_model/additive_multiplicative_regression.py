import numpy as np
import copy
from scipy.optimize import minimize

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
