import torch


def SVD(X, r):
    """
    Standard SVD for low-rank approximation.
    
    Args:
        X: Input matrix (m x n)
        r: Target rank
    
    Returns:
        V: Right singular vectors (r x n)
        U: Left singular vectors with singular values absorbed (m x r)
    """
    U, S, V = torch.linalg.svd(X.to(torch.float32), full_matrices=False)
    U, S, V = U[:, :r], S[:r], V[:r, :]
    U @= torch.diag(S)
    return V, U
