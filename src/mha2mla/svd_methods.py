import torch


def SVD(X, r):
    """
    Standard SVD for low-rank approximation.
    
    Args:
        X: Input matrix (m x n)
        r: Target rank
    
    Returns:
        down: Matrix (n x r) for down projection (V.T)
        up: Matrix (m x r) for up projection with singular values absorbed (U @ S)
    
    Usage for PyTorch Linear layers:
        X ≈ up @ down.T
        down_weight = down.T  # shape (r, n) for Linear(n, r)
        up_weight = up.T      # shape (m, r) for Linear(r, m)
    """
    U, S, V = torch.linalg.svd(X.to(torch.float32), full_matrices=False)
    U, S, V = U[:, :r], S[:r], V[:r, :]
    U @= torch.diag(S)
    # Return (down=V.T, up=U) so that X ≈ U @ V = up @ down.T
    return V.T, U  # (n, r), (m, r)


def rSVD(X, r, oversampling=10, n_iter=2):
    """
    Randomized SVD with power iteration for faster low-rank approximation.
    
    This implementation uses random projection to approximate the range of X,
    followed by power iteration to improve accuracy, and finally computes
    SVD on a smaller matrix.
    
    Args:
        X: Input matrix (m x n)
        r: Target rank
        oversampling: Additional samples for accuracy (default: 10)
        n_iter: Number of power iterations for improved accuracy (default: 2)
    
    Returns:
        down: Matrix (n x r) for down projection (V.T)
        up: Matrix (m x r) for up projection with singular values absorbed (U @ S)
    
    Usage for PyTorch Linear layers:
        X ≈ up @ down.T
        down_weight = down.T  # shape (r, n) for Linear(n, r)
        up_weight = up.T      # shape (m, r) for Linear(r, m)
    """
    # Convert to float32 for numerical stability
    X = X.to(torch.float32)
    m, n = X.shape
    
    # Determine the size of the random matrix
    k = min(r + oversampling, min(m, n))
    
    # Step 1: Generate random Gaussian matrix
    Omega = torch.randn(n, k, device=X.device, dtype=X.dtype)
    
    # Step 2: Form Y = X * Omega
    Y = X @ Omega
    
    # Step 3: Power iteration to improve accuracy
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
    
    # Step 4: Orthonormalize Y using QR decomposition
    Q, _ = torch.linalg.qr(Y)
    
    # Step 5: Form B = Q^T * X (smaller matrix)
    B = Q.T @ X
    
    # Step 6: Compute SVD of the smaller matrix B
    U_tilde, S, V = torch.linalg.svd(B, full_matrices=False)
    
    # Step 7: Recover U = Q * U_tilde
    U = Q @ U_tilde
    
    # Step 8: Truncate to rank r
    U, S, V = U[:, :r], S[:r], V[:r, :]
    
    # Step 9: Absorb singular values into U (to match SVD behavior)
    U = U @ torch.diag(S)
    
    # Return (down=V.T, up=U) to match SVD format
    return V.T, U  # (n, r), (m, r)


def low_rank_decomposition(X, r, method="svd", oversampling=10, n_iter=2):
    """
    Unified interface for low-rank decomposition methods.
    
    Automatically selects and applies the specified decomposition method
    with appropriate parameters.
    
    Args:
        X: Input matrix (m x n)
        r: Target rank
        method: Decomposition method - "svd" or "rsvd" (default: "svd")
        oversampling: Additional samples for rSVD accuracy (default: 10)
        n_iter: Number of power iterations for rSVD (default: 2)
    
    Returns:
        V: Right singular vectors (r x n)
        U: Left singular vectors with singular values absorbed (m x r)
    
    Raises:
        ValueError: If an unknown method is specified
    """
    if method == "svd":
        return SVD(X, r)
    elif method == "rsvd":
        return rSVD(X, r, oversampling=oversampling, n_iter=n_iter)
    else:
        raise ValueError(f"Unknown decomposition method: '{method}'. Choose 'svd' or 'rsvd'")
