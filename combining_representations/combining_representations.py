# if gurobi is available then import it

try:
    import gurobipy as gp
    from gurobipy import GRB
except:
    pass

# import the open-source apis (pulp, mip)
import pulp
import mip

import numpy as np

# import some utility functions
from .utils import *


def combine_representations(dist_matrix, voi_index, S_indices, return_new_dists=True, threshold=None, solver='coin_cmd', api='py-mip', variable_num_tol=0.001, max_seconds=np.inf):
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
        If true, returns both the weights and the corresponding distance matrix.
    threshold - maximum value of the rank of an element of S_indices.
    solver - solver to use. in {'coin_cmd'}
    api - api to use. in {'gurobi', 'py-mip', 'pulp'}
    variable_num_tol - float in (0, 1]. resolution of the approximation of the continuous weights.
        If None, continuous weights are found.
    max_seconds - float in (0, inf)
        
    Return
    weights - np.array (length=J)
        Array containing the coefficients for the optimal distance function.
    -- optional -- new_dists - np.array (length=n)
        Array containing the distances after applying the learned weight vector. 
    """
    
    # Grab the shape of the data that was passed int
    n, J = dist_matrix.shape

    # Pre-process the data so that there are no elements of S_indices that are above threshold
    if threshold is not None:
        ranks = evaluate_best_vertices(dist_matrix, vertices=np.arange(J), s_star=S_indices)
        dist_matrix = edit_dist_matrices(dist_matrix, S_indices, ranks, threshold)

    # Grab the maximum value of dist_matrix (to be used as an upper bound later)
    M = np.max(abs(dist_matrix))
    
    # Grab the number of elements known to be similar to voi_index
    S = len(S_indices)
    
    # Define an array of integers corresponding to elements not in S_indices
    Q_indices = np.array([int(i) for i in np.concatenate((range(0, voi_index), range(voi_index+1, n))) if i not in S_indices])

    # Grab the number of elements not known to be similar to voi_index
    Q = len(Q_indices)

    # We can either use continuous weights for the representations or use discrete approximation
    if variable_num_tol is not None:
        # Here, we are using a discrete approximation
        
        # variable_num_tol is in the interval (0, 1]. If variable_num_tol is close to 0 that means 
        # we want our approximation to be of a high resolution. To achieve this, we define 1 / variable_num_tol
        # to be the maximum value that a particular weight can take on. 
        # i.e. with variable_num_tol = 0.1, the weights can take on values 0.0, 0.1, 0.2, 0.3, .., 1.
        # We normalize later.
        up_bound = int(np.math.ceil(1 / variable_num_tol))
        variable_num_tol = 1 / up_bound
        cat='Integer'
    else:
        # Here, we let the weights be continuous
        up_bound=1
        cat='Continuous'
        
       
    # We've implemented the ILP in 3 different APIs: pulp, py-mip and gurobi.
    # Gurobi seems to be the industry standard but licensing is a bit prohibitive.
    
    # Each API is relatively similar. First, you define a set of variables. 
    # Then, using these variables, you define an objective function and a set of constraints that you assign to a model object. 
 
    if api == 'pulp':
        # Define a model.
        model=pulp.LpProblem(sense=pulp.LpMinimize)
        
        # Define indicator variables (as defined in section 1.2 of https://arxiv.org/pdf/2005.10700.pdf) for every vertex
        # That is not in S_indices.
        # NB: i am more than happy to walk through the paper if that would be helpful!
        inds = pulp.LpVariable.dicts("indicators for elements not in S", 
                                    (q for q in Q_indices),
                                    cat='Integer',
                                    upBound=1,
                                    lowBound=0
                                    )

        # Define non-negative weight variables for each of the representations.
        weights = pulp.LpVariable.dicts("weights for representations",
                                       (j for j in range(J)),
                                       cat=cat,
                                       upBound=up_bound,
                                       lowBound=0
                                       )

        # Set the objective function.
        model += (
            pulp.lpSum(
                [inds[(q)] for q in Q_indices]
            )
        )

        # Add constraint that the weights must sum to the upper bound defined by variable_num_tol.
        model += (
            pulp.lpSum(
                [weights[(j)] for j in range(J)]
            ) == up_bound
        )
        
        # Add constraint that elements of S_indices should be closer than elements not in S_indices (or, the Q_indices)
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
                    pulp.lpSum(inds[(q)] * M * up_bound)
                )

        # Different solvers for api == 'pulp'
        try:
            if solver == 'pulp':
                model.solve()
            elif solver == 'coin_cmd':
                model.solve(solver=pulp.COIN_CMD())

            alpha_hat = np.array([w.varValue for w in weights.values()])
        except Exception as e: 
            print(e)
            return None
            
        alpha_hat = np.array([w.varValue for w in weights.values()])
    elif api=='gurobi':
        model= gp.Model()
        
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', max_seconds)

        ind = model.addVars(Q, vtype=GRB.BINARY, name='ind')
        model.setObjective(gp.quicksum(ind), GRB.MINIMIZE)

        w = model.addVars(J, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='w')
        model.addConstr(w.sum() == 1)

        for s in S_indices:
            temp_s = gp.tupledict([((i), dist_matrix[s, i]) for i in range(J)])
            for i, q in enumerate(Q_indices):
                temp_q = gp.tupledict([((i), dist_matrix[q, i]) for i in range(J)])
                model.addConstr(w.prod(temp_s) <= w.prod(temp_q) + ind[i]*M)
            
        model.optimize()
        alpha_hat = np.array([i.X for i in list(w.values())])

    elif api=='py-mip':
        model = mip.Model(sense=mip.MINIMIZE)

        inds = [model.add_var(name='inds', var_type=mip.BINARY) for q in range(Q)]

#         if variable_num_tol is None:
        weights = [model.add_var(name='weights', lb=0.0, ub=up_bound, var_type=cat) for j in range(J)]
        model += mip.xsum(w for w in weights) == 1
#         else:
#             weights = [model.add_var(name='weights', lb=0, ub=up_bound, var_type='I') for j in range(J)]
#             model += mip.xsum(w for w in weights) == up_bound
        
        for s in S_indices:
            for i, q in enumerate(Q_indices):
                model += mip.xsum(weights[j] * dist_matrix[s, j] for j in range(J)) <= mip.xsum(weights[j] * dist_matrix[q, j] for j in range(J)) + inds[i]*M*up_bound

        model.objective = mip.xsum(ind for ind in inds)
        model.optimize(max_seconds=max_seconds)

        alpha_hat = np.array([w.x for w in weights])
    else:
        raise ValueError("api %s not implemented"%(api))
    
    # Return the new distances if told to do so.
    if return_new_dists:
        if np.sum(alpha_hat == 0) == J:
            return alpha_hat, dist_matrix
        else:
            return alpha_hat, np.average(dist_matrix, axis=1, weights=alpha_hat)

    if alpha_hat[0] is None:
        return None
    else:
        # Normalize
        return alpha_hat / np.sum(alpha_hat)
 
def multiple_pairs(dist_matrices, voi_ind_to_S_sets, threshold=None, api='py-mip', solver='pulp', variable_num_tol=0.001, max_seconds=np.inf):
    """
    Doc string to be written.
    """
    voi_indices = list(voi_ind_to_S_sets.keys())
    n_voi = len(voi_indices)
    S_sets = list(voi_ind_to_S_sets.values())
    
    M = np.max(np.max(abs(dist_matrices), axis=0))

    a, b, c = dist_matrices.shape

    if b == c:
        J = a
        n = b
        dist_matrices = dist_matrices.transpose((1, 2, 0))[voi_indices]
    else:
        n = b
        J = c

    voi_to_Q_indices = {voi: np.array([int(i) for i in np.concatenate((range(0, voi), range(voi+1, n))) if i not in voi_ind_to_S_sets[voi]]) for voi in voi_indices}

    if variable_num_tol is not None:
        up_bound = int(np.math.ceil(1 / variable_num_tol))
        variable_num_tol = 1 / up_bound
        cat='Integer'
    else:
        up_bound=1
        cat='Continuous'
    
    if threshold is not None:
        if b == c:
            raise ValueError('dist_matrices not shaped correctly')
        temp_dist_matrices = np.zeros((n_voi, n, J))
        for i, voi in enumerate(voi_indices):
            temp_ranks = evaluate_best_vertices(dist_matrices[i], np.arange(J), S_sets[i])
            temp_dist_matrices[i] = edit_dist_matrices(dist_matrices[i], S_sets[i], temp_ranks, threshold)

    if api == 'pulp':
        model=pulp.LpProblem(sense=pulp.LpMinimize)

        inds = pulp.LpVariable.dicts("indicators for elements not in S", 
                                    ('voi' + str(voi) + str(q) for voi in voi_indices for q in voi_to_Q_indices[voi]),
                                    cat='Integer',
                                    upBound=1,
                                    lowBound=0
                                    )

        weights = pulp.LpVariable.dicts("weights for representations",
                                       (j for j in range(J)),
                                       cat=cat,
                                       upBound=up_bound,
                                       lowBound=0
                                       )

        model += (
            pulp.lpSum(
                [inds[('voi' + str(voi) + str(q))] for voi in voi_indices for q in voi_to_Q_indices[voi]]
            )

        )

        model += (
            pulp.lpSum(
                [weights[(j)] for j in range(J)]
            ) == up_bound
        )

        for i, voi in enumerate(voi_indices):
            dist_matrix = dist_matrices[i, :, :]
            for s in voi_ind_to_S_sets[voi]:
                for q in voi_to_Q_indices[voi]:
                    model += (
                        pulp.lpSum(
                            [weights[(j)] * dist_matrix[s, j] for j in range(J)]
                        )
                        <=
                        pulp.lpSum(
                            [weights[(j)] * dist_matrix[q, j] for j in range(J)]
                        ) 
                        + 
                        pulp.lpSum(inds[('voi' + str(voi) + str(q))] * M * up_bound)
                    )
        try:
            if solver == 'pulp':
                model.solve()
            elif solver == 'coin_cmd':
                model.solve(solver=pulp.COIN_CMD())

            alpha_hat = np.array([w.varValue for w in weights.values()])
        except Exception as e: 
            print(e)
            return None
    elif api=='py-mip':
        model = mip.Model(sense=mip.MINIMIZE)

        # Need to fix inds
        inds = [[model.add_var(name='inds', var_type=mip.BINARY) for q in voi_to_Q_indices[voi]] for voi in voi_indices]

#         if variable_num_tol is None:
        weights = [model.add_var(name='weights', lb=0.0, ub=up_bound, var_type=cat) for j in range(J)]
        model += mip.xsum(w for w in weights) == up_bound
#         else:
#             weights = [model.add_var(name='weights', lb=0, ub=up_bound, var_type='I') for j in range(J)]
#             model += mip.xsum(w for w in weights) == up_bound
        
        for i, voi in enumerate(voi_indices):
            dist_matrix = dist_matrices[i, :, :]
            for s in voi_ind_to_S_sets[voi]:
                for k, q in enumerate(voi_to_Q_indices[voi]):
                    model += mip.xsum(weights[j] * dist_matrix[s, j] for j in range(J)) <= mip.xsum(weights[j] * dist_matrix[q, j] for j in range(J)) + inds[i][k]*M*up_bound

        model.objective = mip.xsum(mip.xsum(i for i in ind) for ind in inds)
        model.optimize(max_seconds=max_seconds)

        alpha_hat = np.array([w.x for w in weights])

    if alpha_hat[0] is None:
        return None
    else:
        return alpha_hat / np.sum(alpha_hat)
