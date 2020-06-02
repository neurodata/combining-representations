import numpy as np
import graspy
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

from combining_representations import combining_representations as cr

def generate_latent_positions(n=50, d=2, acorn=None):
    """
    A function to generate an adjacency matrix.
    
    Input
    n - int
        If P is None then n is the number of latent positions to randomly generate.
    d - int
        If P is None the d is the dimension of the latent positions.
    acorn - int
        Random seed.
        
    Return
    X - np.array (shape=(n,d))
        An array of uniformly distributed points in the positive unit sphere
    """
    
    if acorn is not None:
        np.random.seed(acorn)
        
    mvns = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)

    mvns = abs(mvns) / np.linalg.norm(mvns, axis=1)[:, None]

    unis = np.random.uniform(size=n)**(1/d)
    X = mvns * unis[:, None]
                
    return X
  

def generate_distance_matrix(A, embedding_functions, covariance_functions, ind=0,acorn=None):
    """
    A function to generate a distance matrix given an adjacency matrix. The ith column of the
    distance matrix is the Euclidean distance from the first node in the graph to the other
    nodes in the ith embedding.
    
    Input
    A - np.array (shape=(n,n))
        Probability matrix or adjacency matrix.
    emebedding_functions - list of functions (length=J)
        Embedding methods for A.
    covariance_functions - list of np.arrays (length=J)
        Functions estimate the covariance used to calculate distances.
    ind - int
        Index of v star.
    acorn - int
        Random seed.
        
    Return
    dist_matrix - np.array (shape=(n, J))
        Distance matrix where the ith column is the Euclidean distance from v_star
        to all of the other nodes in the graph in the ith embedding.
    """
    
    if acorn is not None:
        np.random.seed(acorn)
        
    n = A.shape[0]
        
    J = len(covariance_functions)
            
    dist_matrix = np.zeros((n, J))
    
    for j, embed in enumerate(embedding_functions):
        if j == 0:
            temp_X = embed(A)
        else:
            temp_X = np.sqrt(n) * embed(A)
            
        if isinstance(covariance_functions[j], np.ndarray):
            temp_cov = covariance_functions[j]
        else:
            temp_cov = covariance_functions[j](temp_X, ind)
            
        diffs = np.array([temp_X[ind] - temp_X[i] for i in range(n)])
        dist_matrix[:, j] = np.sqrt(np.array([diffs[i].T @ temp_cov @ diffs[i] for i in range(n)]))
       
        if np.sum(dist_matrix[:, j] < 0) > 0:
            print("i broke on distance %i"%(j))
    return dist_matrix


def generate_S_indices(dist_matrix, alpha, n_inds=1, return_new_dists=False, beta=0.5):
    """
    A function to generate the nodes of interest.
    
    Input
    dist_matrix - np.array (shape=(n,J))
        Array containing the distances between the vertex of interest and the other n - 1
        vertices. It is assumed that the vertex of interest is indexed by 0.
    alpha - float or array-like
        Coefficients of the distances to generate the ground truth. alpha can only be an int
        if J == 2. If alpha is array-like then the sum of alpha must be 1.
    n_inds - int or func
        Number of vertices in the vertex set of interest. If n_inds is a function then the
        ceiling of n_inds(n) is used.
        
    Return
    S - np.array (length=n_inds)
        A list of indices of length n_inds in range(1,n) corresponding to vertices of interest.
    """
    
    n, J = dist_matrix.shape
    
    if isinstance(alpha, float) or alpha == 0 or alpha == 1:
        if J == 2:
            alpha = [alpha, 1-alpha]
        elif J == 4:
            alpha = np.array([beta*alpha, beta*(1-alpha), (1-beta)*alpha, (1-beta)*(1-alpha)])
    
    if not np.isclose(np.sum(alpha), 1):
        print(np.sum(alpha))
        assert np.sum(alpha) == 1
    
    if not isinstance(n_inds, int):
        n_inds = int(np.math.ceil(n_inds(n)))
    
    new_distances = np.average(dist_matrix, axis=1, weights=alpha)
    
    new_nearest_neighbors = np.argsort(new_distances)
    
    S = new_nearest_neighbors[1:n_inds+1]
    
    if return_new_dists:
        return S, new_distances
    else:
        return S

def generate_2d_rotation(theta=0, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
    
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    
    return R

def identity(X, ind):
    _, d = X.shape
    return np.eye(d)


def rank_distance(true_ranks, estimated_ranks):
    return np.mean(abs(true_ranks - estimated_ranks))


def mean_reciprocal_rank(rank_lists, inds):
    """
    Calculates mean reciprocal rank given a rank list and set of indices of interest.
    
    Input
    rank_list - array-like (length=n-1)
        A ranked array-like of objects (assumed to be integers).
    inds - array-like (length<=n-1)
        A array-like of objects (assumed to be integers).
        
    Return
    mrr - float
        Mean reciprocal rank of the objects in inds.
    """
    mrrs = np.zeros(len(rank_lists))
    
    for i, r in enumerate(rank_lists):
        mrrs[i] = np.mean(1 / (np.array([np.where(r == s)[0][0] for s in inds])))
        
    return mrrs


def mean_rank(rank_list, inds):
    ranks = np.array([np.where(rank_list == i)[0][0] for i in inds]) + 1
#     print(ranks)
    return np.mean(ranks)


def remove_S_indices(rank_lists, S_indices):
    new_rank_lists = []
    for i, r in enumerate(rank_lists):
        idx = np.array([np.where(r == s)[0][0] for s in S_indices])
        new_rank_lists.append(np.delete(r, idx))
        
    return new_rank_lists


def average_rank(rank_lists, s_star_indices):
    
    average_ranks = np.zeros(len(rank_lists))
    
    for i, r in enumerate(rank_lists):
        average_ranks[i] = np.mean(np.array([np.where(r == s)[0][0] for s in s_star_indices]))
    
    return average_ranks


def monte_carlo(mc_its, P, embedding_functions, covariance_functions, 
                alpha, n_inds, s_stars, latent=True, poc=False, 
                metric='rank', beta=0.25, k=1,
                acorn=None):
    """
    Function to run a bunch of simulations.
    
    See documentation for above functions.
    
    Return
    mmrs - np.array (shape=(mc_its, J+1))
        Matrix of mean reciprocal ranks for each of the embedding functions and the estimated
        optimal distance.
    """
    if acorn is None:
        np.random.seed(acorn)
        
    J = len(embedding_functions)
    
    if J == 2:
        c = 1
    elif J == 4:
        c = 3
    else:
        c=1
        
    errors = np.zeros((mc_its, 3))
    
    if J in [2, 4]:
        alpha_hats = np.zeros(mc_its)

    true_dist = generate_distance_matrix(P, embedding_functions=embedding_functions, 
                                         covariance_functions=covariance_functions)
    
    all_S_inds, new_dists = generate_S_indices(true_dist, alpha, n_inds=n_inds, return_new_dists=True, beta=beta)
    true_rankings = np.argsort(new_dists)
        
    if isinstance(s_stars, np.ndarray):
        for i,s in enumerate(s_stars):
            if s < 0:
                s_stars[i] = len(all_S_inds) + s
        S_star_inds = all_S_inds[s_stars]
    elif isinstance(s_stars, int):
        S_star_inds = np.random.choice(all_S_inds, size=s_stars)
    else:
        raise ValueError('bad')
    S_inds = np.array([i for i in all_S_inds if i not in S_star_inds])
        
    for i in range(mc_its):
        A_bar = np.zeros(P.shape)
        for j in range(k):
            connected = False
            while not connected:
                A_temp = graspy.simulations.rdpg(P)
                connected = graspy.utils.is_fully_connected(A_temp)
                A_bar = (1/(j+1)) * A_temp + (j/(j+1))*A_bar
            
        if poc:
            if J == 2:
                dist_matrix = true_dist
            elif J == 4:
                dist_matrix = generate_distance_matrix(P, embedding_functions[2:], 
                                                   covariance_functions=covariance_functions[2:])
        else:
            if J == 2:
                dist_matrix = generate_distance_matrix(A_bar, embedding_functions, 
                                                   covariance_functions=covariance_functions)
            elif J == 4:
                dist_matrix = generate_distance_matrix(A_bar, embedding_functions[2:], 
                                                   covariance_functions=covariance_functions[2:])
                    
        alpha_hat_vec, opt_dists = cr.combine_representations(dist_matrix, 0, S_inds)
        
        if J in [2, 4]:
            alpha_hats[i] = alpha_hat_vec[0]
            
        if J == 2:
            ranked_lists = [np.argsort(dist_matrix[:, j]) for j in range(J)]
        elif J == 4:
            ranked_lists = [np.argsort(dist_matrix[:, j]) for j in range(J-2)]
        ranked_lists.append(np.argsort(opt_dists))
                
        if metric == 'error':
            errors[i] = np.array([rank_distance(S_star_inds, np.where(R == true_rankings[S_star_inds])) for R in ranked_lists])
        elif metric == 'rank':
            ranked_lists = remove_S_indices(ranked_lists, S_inds)
            errors[i] = average_rank(ranked_lists, S_star_inds)
        elif metric == 'mrr':
            ranked_lists = remove_S_indices(ranked_lists, S_inds)
            errors[i] = mean_reciprocal_rank(ranked_lists, S_star_inds)
            
    if J in [2, 4]:
        return alpha_hats, errors
    else:
        return errors


def plot_kde_differences(all_data, figsize=(8,6), h=1, num=1000, suptitle=None, filename=None):
    J, n = all_data.shape
    colors = sns.color_palette('Set1', n_colors=J)
    
    min_ = min(np.array([min(data) for data in all_data]))
    max_ = max(np.array([max(data) for data in all_data]))
    
    linspace = np.linspace(min_ - h, max_ + h, num)
    
    fig, ax = plt.subplots(J, J, sharex=True, sharey=True)
        
    for i, di in enumerate(all_data):
        for j, dj in enumerate(all_data):
            if i == j:
                temp_kdei = gaussian_kde(di)
                ax[i,i].plot(linspace, temp_kdei.pdf(linspace), c=colors[-i-1])
            else:
                temp_kdei = gaussian_kde(di)
                temp_kdej = gaussian_kde(dj)
                
                ax[i,j].plot(linspace, temp_kdei.pdf(linspace) - temp_kdej.pdf(linspace), ls="--")
            ax[i,j].axhline(y=0, c='k', lw=1)
    if suptitle is not None:
        fig.suptitle(suptitle)
        
    if filename:
        plt.savefig(filename)
        

def find_max_alpha(dist_matrix, S_indices, min_alpha, h=0.0001, max_its=50):
    n, J = dist_matrix.shape
    
    assert J == 2
    
    min_alpha_dist = np.average(dist_matrix, axis=1, weights = (1 - min_alpha, min_alpha))
    min_alpha_ranks = np.argsort(min_alpha_dist)
    
    obj_func_value = max([np.where(min_alpha_ranks == s)[0][0] for s in S_indices])
    
    interval = np.linspace(min_alpha, 1, num=(1 - min_alpha) / h)
    
    found = False
    prev = min_alpha
    current = (1 + min_alpha)/2
    i=0
    while i < max_its:
        # find closest point in our mesh
        temp_alpha = interval[np.argmin(abs(interval - current))]
        
        # find distance corresponding to closest point in our mesh
        temp_dist = np.average(dist_matrix, axis=1, weights = (1 - temp_alpha, temp_alpha))
        
        # find objective function value correpsonding to closest point in our mesh
        temp_obj_func_value = max([np.where(min_alpha_ranks == s)[0][0] for s in S_indices])
        
        # went too far
        if temp_obj_func_value > obj_func_value:
            prev = current
            current = (prev + min_alpha)/2
        elif temp_obj_func_value == obj_func_value:
            prev = current
            current = (prev + 1)/2
        
        if abs(prev - current) < h:
            return current
            
        i+=1
    return current