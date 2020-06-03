import pulp
import numpy as np


def combine_representations(dist_matrix, voi_index, S_indices, return_new_dists=True):
    """
    A function to find the weights of optimal linear combination of representations.
    
    Input
    dist_matrix - np.array (shape=(n, J))
        Array containing the distances between the vertex of interest and the other n - 1
        vertices.
    voi_index - int
        Index of vertex of interest.
    S_indices - array-like
        Indices of the vertices that should be at the top of the
        nomination list for the vertex of interest.
    return_new_dists - bool
        If true, returns both the weights and the corresponding distance matrix
        
    Return
    weights - np.array (length=J)
        Array containing the coefficients for the optimal distance function.
    """
    
    n, J = dist_matrix.shape
    M = np.sum(abs(dist_matrix))
    
    S = len(S_indices)
    Q_indices = np.array([int(i) for i in np.concatenate((range(0, voi_index), range(voi_index+1, n))) if i not in S_indices])

    Q = len(Q_indices)
    
    M = np.sum(abs(dist_matrix))

    model=pulp.LpProblem(sense=pulp.LpMinimize)
    ind = pulp.LpVariable.dicts("indicators for elements not in S", 
                                (q for q in Q_indices),
                                cat='Integer',
                               upBound=1,
                               lowBound=0)

    weights = pulp.LpVariable.dicts("weights for representations",
                                   (j for j in range(J)),
                                   cat='Continuous',
                                   upBound=1,
                                   lowBound=0
                                   )

    model += (
        pulp.lpSum(
            [ind[(q)] for q in Q_indices]
        )
    )

    model += (
        pulp.lpSum(
            [weights[(j)] for j in range(J)]
        ) == 1
    )

    for s in S_indices:
        for q in Q_indices:
            model += (
                pulp.lpSum(
                    [weights[(j)] * dist_matrix[s, j] for j in range(J)]
                )
                <=
                pulp.lpSum(
                    [weights[(j)] * dist_matrix[q, j] for j in range(J)]
                ) 
                + 
                pulp.lpSum(ind[(q)] * M)
            )

    model.solve()
    alpha_hat = np.array([w.varValue for w in weights.values()])

    if return_new_dists:
        return alpha_hat, np.average(dist_matrix, axis=1, weights=alpha_hat)
    
    return alpha_hat