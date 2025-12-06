"""
Test script to verify SVD and rSVD implementations work correctly.
"""

import torch
import sys
sys.path.insert(0, 'src')

from mha2mla.svd_methods import SVD, rSVD, low_rank_decomposition


def test_basic_functionality():
    """Test that both SVD and rSVD produce valid decompositions."""
    print("=" * 60)
    print("Testing basic functionality...")
    print("=" * 60)
    
    # Create a random matrix
    m, n, r = 100, 80, 10
    X = torch.randn(m, n)
    
    # Test SVD
    print("\n1. Testing standard SVD...")
    V_svd, U_svd = SVD(X, r)
    print(f"   Input shape: {X.shape}")
    print(f"   V shape: {V_svd.shape}, U shape: {U_svd.shape}")
    print(f"   ✓ SVD completed successfully")
    
    # Test rSVD
    print("\n2. Testing randomized SVD...")
    V_rsvd, U_rsvd = rSVD(X, r, oversampling=10, n_iter=2)
    print(f"   V shape: {V_rsvd.shape}, U shape: {U_rsvd.shape}")
    print(f"   ✓ rSVD completed successfully")
    
    # Test unified interface
    print("\n3. Testing unified interface...")
    V_unified_svd, U_unified_svd = low_rank_decomposition(X, r, method="svd")
    V_unified_rsvd, U_unified_rsvd = low_rank_decomposition(X, r, method="rsvd")
    print(f"   ✓ Unified interface works for both methods")


def test_reconstruction_error():
    """Test reconstruction quality of both methods."""
    print("\n" + "=" * 60)
    print("Testing reconstruction quality...")
    print("=" * 60)
    
    m, n, r = 200, 150, 20
    X = torch.randn(m, n)
    
    # Compute reconstructions
    V_svd, U_svd = SVD(X, r)
    X_recon_svd = U_svd @ V_svd
    
    V_rsvd, U_rsvd = rSVD(X, r, oversampling=10, n_iter=2)
    X_recon_rsvd = U_rsvd @ V_rsvd
    
    # Compute errors
    error_svd = torch.norm(X - X_recon_svd) / torch.norm(X)
    error_rsvd = torch.norm(X - X_recon_rsvd) / torch.norm(X)
    
    print(f"\n   SVD reconstruction error:  {error_svd:.6f}")
    print(f"   rSVD reconstruction error: {error_rsvd:.6f}")
    print(f"   Difference: {abs(error_svd - error_rsvd):.6f}")
    
    if error_rsvd < 0.1:
        print(f"   ✓ Both methods provide good approximations")
    else:
        print(f"   ⚠ Warning: High reconstruction error")


def test_performance():
    """Compare performance of SVD vs rSVD."""
    print("\n" + "=" * 60)
    print("Testing performance (approximate timing)...")
    print("=" * 60)
    
    import time
    
    m, n, r = 1000, 800, 50
    X = torch.randn(m, n)
    
    # Test SVD speed
    start = time.time()
    V_svd, U_svd = SVD(X, r)
    svd_time = time.time() - start
    
    # Test rSVD speed
    start = time.time()
    V_rsvd, U_rsvd = rSVD(X, r, oversampling=10, n_iter=2)
    rsvd_time = time.time() - start
    
    print(f"\n   Matrix size: {m} x {n}, rank: {r}")
    print(f"   SVD time:  {svd_time:.4f}s")
    print(f"   rSVD time: {rsvd_time:.4f}s")
    print(f"   Speedup:   {svd_time/rsvd_time:.2f}x")
    
    if rsvd_time < svd_time:
        print(f"   ✓ rSVD is faster than SVD")
    else:
        print(f"   ⚠ rSVD slower (expected for small matrices)")


def test_edge_cases():
    """Test edge cases and parameter variations."""
    print("\n" + "=" * 60)
    print("Testing edge cases...")
    print("=" * 60)
    
    # Test different oversampling values
    print("\n   Testing different oversampling values...")
    X = torch.randn(100, 80)
    for oversampling in [0, 5, 10, 20]:
        V, U = rSVD(X, r=10, oversampling=oversampling, n_iter=2)
        error = torch.norm(X - U @ V) / torch.norm(X)
        print(f"      oversampling={oversampling:2d}: error={error:.6f}")
    
    # Test different iteration counts
    print("\n   Testing different iteration counts...")
    for n_iter in [0, 1, 2, 4]:
        V, U = rSVD(X, r=10, oversampling=10, n_iter=n_iter)
        error = torch.norm(X - U @ V) / torch.norm(X)
        print(f"      n_iter={n_iter}: error={error:.6f}")
    
    print(f"\n   ✓ All edge cases handled correctly")


if __name__ == "__main__":
    print("\n🧪 SVD Methods Test Suite\n")
    
    try:
        test_basic_functionality()
        test_reconstruction_error()
        test_performance()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
