#### Regularized RegMean
import itertools
from typing import List, Optional, Tuple, Literal, Dict
from copy import deepcopy
import torch
from tqdm import tqdm
import math
from scipy.linalg import solve_sylvester

batched_kron = torch.vmap(torch.kron)


# ===================================================
#                  Helper Functions
# ===================================================


def make_ls_mats(pus, pvs, a_tilde):
    """Create least squares params from task proj mats and target mat

    Args:
        pus (torch.Tensor): Task left proj mats. Shape: (B, Do, Do)
        pvs (torch.Tensor): Task right proj mats. Shape: (B, Di, Di)
        a_tilde (torch.Tensor): Target mat. Shape: (B, Do, Di)

    Returns:
        mis_stack (torch.Tensor): (B*D, D)
        a_tilde_flat (torch.Tensor): (B*D*D)
    """
    B, Do, Di = a_tilde.shape
    D = Do * Di
    mis = batched_kron(pvs, pus)  # (B, Do*Di, Do*Di)
    a_tilde_flat = a_tilde.reshape(-1)  # (B*Do*Di)
    mis_stack = mis.reshape(-1, D)  # (B*D, D)
    return mis_stack, a_tilde_flat


def compute_procrustes(x: torch.Tensor) -> torch.Tensor:
    """Finds best ortho approx of x

    Args:
        x (torch.Tensor): (Di, Do)

    Returns:
        torch.Tensor: (Di, Do)
    """
    u, _, vt = torch.linalg.svd(x, full_matrices=False)
    return u @ vt


def compute_tsvopt_loss(u_ortho, s_hat, vt_ortho, a_tilde):
    """Computes the loss for the TSV optimization.

    Args:
        u_ortho (torch.Tensor): (Di, Rp)
        vt_ortho (torch.Tensor): (Rp, Do)
        a_tilde (torch.Tensor): (B, Do, Di)

    Returns:
        _type_: _description_
    """
    pred = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)  # (Di, Do)
    loss = (
        torch.linalg.norm(pred.unsqueeze(0) - a_tilde, ord="fro", dim=(1, 2))
        .square()
        .mean()
    )  # (B, Do, Di)
    return loss


def compute_lsopt_loss(a_hat, mis_stack, a_tilde_flat):
    pred = mis_stack @ a_hat.reshape(-1)
    loss = torch.nn.functional.mse_loss(pred, a_tilde_flat)  # default = mean
    return loss


def compute_lsopt_loss_gd(a_hat, a_tilde):
    """Computes the loss for the LSopt optimization.

    Args:
        a_hat (torch.Tensor): (Do, Di)
        a_tilde (torch.Tensor): (B, Do, Di)

    Returns:
        _type_: _description_
    """
    B, Do, Di = a_tilde.shape
    u, s, vt = torch.linalg.svd(a_tilde, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = max(1, R // B)
    # TODO
    # Rp = max(1, R // math.sqrt(B))
    u = u[:, :, :Rp]
    vt = vt[:, :Rp, :]

    # projectors
    pus = u @ u.transpose(1, 2)  # (B, Do, Do)
    pvs = vt @ vt.transpose(1, 2)  # (B, Di, Di)
    pvs = vt.transpose(1, 2) @ vt  # (B, Di, Di)

    preds = torch.einsum("boj,jk,bki->boi", pus, a_hat, pvs)
    tgt = torch.einsum("boj,bjk,bki->boi", pus, a_tilde, pvs)
    loss = torch.linalg.norm(preds - tgt, ord="fro", dim=(1, 2)).square().mean()
    return loss


def compute_polar(A):
    # Helper to project A onto the orthogonal group
    # A = UP -> returns U
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U @ Vh


def matrix_pow(A, pow=1.0):
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=1e-6)  # prevent division by zero
    return eigvecs @ torch.diag(eigvals.pow(pow)) @ eigvecs.T


def solve_sylvester_simplified(a, b, c):
    """
    Solves AW + WB = C assuming A, B are symmetric
    and we force all their eigenvalues to be 1.
    """
    # 1. Get the Eigenbases (Directions)
    # We ignore vals_a/vals_b since we assume they are 1
    _, vecs_a = torch.linalg.eigh(a)
    _, vecs_b = torch.linalg.eigh(b)

    # 2. Rotate C into the aligned coordinate system
    # C_tilde = U_A.T @ C @ U_B
    c_tilde = vecs_a.T @ c @ vecs_b

    # 3. Solve assuming eigenvalues are 1
    # Equation: 1*w + w*1 = c_tilde  =>  2w = c_tilde
    w_tilde = c_tilde / 2.0

    # 4. Rotate back to original space
    w = vecs_a @ w_tilde @ vecs_b.T

    return w


# ===================================================
#                  Merge Methods (per layer)
# ===================================================


def merge_avg(ws, *args, **kwargs):
    return ws.mean(dim=0)


def merge_ta(ws, *args, **kwargs):
    return ws.sum(dim=0)


# **************************************************
#                TSV Variants
# **************************************************


def merge_tsv(tensors, **kwargs):
    """Computes the TSV merge of the given tensors.

    Computes: Uo  Dc Vto

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    N_tasks = len(tensors)
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = R // N_tasks
    u, s, vt = u[:, :, :Rp], s[:, :Rp], vt[:, :Rp, :]

    # # # w/o decorrelation
    # tau_bl = torch.einsum("bij,bj,bjk->bik", u, s, vt)
    # tau[layer_name] = tau_bl.sum(dim=0)

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    u_hat = u.permute(1, 0, 2).reshape(Di, B * Rp)
    s_hat = s.reshape(-1)
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = compute_procrustes(vt_hat.T).T  # (Rp, Do)
    tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)  # (Di, Do)
    return tau_l


def merge_tsv_v1(tensors, **kwargs):
    # Project: Uo UoT  Dc Vo VoT
    N_tasks = len(tensors)
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = R // N_tasks
    u, s, vt = u[:, :, :Rp], s[:, :Rp], vt[:, :Rp, :]

    # # # w/o decorrelation
    # tau_bl = torch.einsum("bij,bj,bjk->bik", u, s, vt)
    # tau[layer_name] = tau_bl.sum(dim=0)

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    u_hat = u.permute(1, 0, 2).reshape(Di, B * Rp)
    s_hat = s.reshape(-1)
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = compute_procrustes(vt_hat.T).T  # (Rp, Do)
    wbar = tensors.sum(dim=0)
    tau_l = u_ortho @ u_ortho.T @ wbar @ vt_ortho.T @ vt_ortho
    return tau_l


def merge_tsv_v2(tensors, *args, **kwargs):
    # Performs same computation as merge_tsv but with the whitening operation (less stable)
    # using Uc(Uc^T Uc)^-1/2 and Vc(Vc^T Vc)^-1/2
    N, Do, Di = tensors.shape
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    Rp = min(Do // N, Di // N)
    u = u[:, :, :Rp]
    vt = vt[:, :Rp, :]
    v = vt.transpose(1, 2)
    s = s[:, :Rp]

    # svd of projectors
    uc = u.permute(1, 0, 2).reshape(Do, N * Rp)
    vc = v.permute(1, 0, 2).reshape(Di, N * Rp)
    gu = uc.T @ uc
    gv = vc.T @ vc
    utilde = matrix_pow(gu, -0.5)
    vtilde = matrix_pow(gv, -0.5)
    return uc @ utilde @ torch.diag(s.reshape(-1)) @ vtilde @ vc.T


def merge_tsv_v3(tensors, *args, **kwargs):
    # Performs similar computation as merge_tsv but with the pseudoinverse operation (less stable)
    # using Uc(Uc^T Uc)^-1 and Vc(Vc^T Vc)^-1
    N, Do, Di = tensors.shape
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    Rp = min(Do // N, Di // N)
    u = u[:, :, :Rp]
    vt = vt[:, :Rp, :]
    v = vt.transpose(1, 2)
    s = s[:, :Rp]

    # svd of projectors
    uc = u.permute(1, 0, 2).reshape(Do, N * Rp)
    vc = v.permute(1, 0, 2).reshape(Di, N * Rp)
    gu = uc.T @ uc
    gv = vc.T @ vc
    utilde = matrix_pow(gu, -1.0)
    vtilde = matrix_pow(gv, -1.0)
    return uc @ utilde @ torch.diag(s.reshape(-1)) @ vtilde @ vc.T


def merge_tsv_v4(tensors, *args, **kwargs):
    # Performs same computation as merge_tsv_v3 but with the pseudoinverse operation (stable)
    # using Uc(Uc^T Uc)^-1 and Vc(Vc^T Vc)^-1
    N, Do, Di = tensors.shape
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)

    # 1. Truncate (Optional, but good for noise reduction)
    Rp = min(Do // N, Di // N)
    u = u[:, :, :Rp]
    s = s[:, :Rp]
    vt = vt[:, :Rp, :]
    v = vt.transpose(1, 2)

    # 2. Reshape to Concatenated forms
    uc = u.permute(1, 0, 2).reshape(Do, N * Rp)
    vc = v.permute(1, 0, 2).reshape(Di, N * Rp)

    # 3. Compute Pseudoinverses directly
    # This effectively computes Uc(Uc^T Uc)^-1 and Vc(Vc^T Vc)^-1
    uc_pinv = torch.linalg.pinv(uc)  # Shape: [N*Rp, Do]
    vc_pinv = torch.linalg.pinv(vc)  # Shape: [N*Rp, Di]

    # 4. Diagonal Matrix
    dc = torch.diag(s.reshape(-1))

    # 5. Combine
    # Note: The formula implies we want the "Left" terms to be (Do x N*Rp)
    # But usually, pinv(A) is (N*Rp x Do).
    # If your formula is Uc @ (Gram)^-1, that is simply (Uc^+)^T or similar.

    # Based on the symmetry of your previous code:
    # Result = (Uc^+)^\top @ Dc @ (Vc^+)

    return uc_pinv.T @ dc @ vc_pinv


def merge_tsv_iso_c_left(tensors, **kwargs):
    """Computes the TSV merge of the given tensors.

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    N_tasks = len(tensors)
    u_tot, s_tot, vt_tot = torch.linalg.svd(tensors, full_matrices=False)
    R = min(u_tot.shape[1], vt_tot.shape[2])
    Rp = R // N_tasks
    u, s, vt = u_tot[:, :, :Rp], s_tot[:, :Rp], vt_tot[:, :Rp, :]

    # # # w/o decorrelation
    # tau_bl = torch.einsum("bij,bj,bjk->bik", u, s, vt)
    # tau[layer_name] = tau_bl.sum(dim=0)

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    s_hat = s.reshape(-1)
    vt_hat = vt.reshape(B * Rp, Do)
    # u_ortho = compute_procrustes(u_hat)  # (Di, Rp)
    u_ortho = compute_procrustes(u.sum(dim=0))  # (Di, Rp)
    vt_ortho = compute_procrustes(vt_hat.T).T  # (Rp, Do)
    tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)  # (Di, Do)
    return tau_l


def merge_tsv_iso(tensors, *args, **kwargs):
    """Computes the TSV merge of the given tensors.

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    N_tasks = len(tensors)
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = R // N_tasks
    u, s, vt = u[:, :, :Rp], s[:, :Rp], vt[:, :Rp, :]

    # # # w/o decorrelation
    # tau_bl = torch.einsum("bij,bj,bjk->bik", u, s, vt)
    # tau[layer_name] = tau_bl.sum(dim=0)

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    u_hat = u.permute(1, 0, 2).reshape(Di, B * Rp)
    # Use the mean of s
    s_hat = torch.ones_like(s.reshape(-1)) * s.mean()
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = compute_procrustes(vt_hat.T).T  # (Rp, Do)
    tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)
    return tau_l


def merge_tsv_perm(tensors, *args, **kwargs):
    """Computes the TSV merge of the given tensors.

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    N_tasks = len(tensors)
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = R // N_tasks
    u, s, vt = u[:, :, :Rp], s[:, :Rp], vt[:, :Rp, :]

    # # # w/o decorrelation
    # tau_bl = torch.einsum("bij,bj,bjk->bik", u, s, vt)
    # tau[layer_name] = tau_bl.sum(dim=0)

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    u_hat = u.permute(1, 0, 2).reshape(Di, B * Rp)
    # Do a random permutation of s
    s_hat = s.reshape(-1)[torch.randperm(s.numel())]
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = compute_procrustes(vt_hat.T).T  # (Rp, Do)
    tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)
    return tau_l


def merge_tsvopt(a_tilde, verbose=False, n_iters=100, *args, **kwargs):
    N_tasks = len(a_tilde)
    u, s, vt = torch.linalg.svd(a_tilde, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = R // N_tasks
    u, s, vt = u[:, :, :Rp], s[:, :Rp], vt[:, :Rp, :]

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    u_hat = u.permute(1, 0, 2).reshape(Di, B * Rp)
    # s_hat = s.reshape(-1)
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = compute_procrustes(vt_hat.T).T  # (Rp, Do)

    # Get optimal singular values
    s_hat = torch.nn.Parameter(s.reshape(-1).clone())
    opt = torch.optim.AdamW([s_hat], lr=0.01)
    for t in range(n_iters):
        opt.zero_grad()
        loss = compute_tsvopt_loss(u_ortho, s_hat, vt_ortho, a_tilde)
        loss.backward()
        opt.step()
    return torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)


# **************************************************
#               Eigen Covariance
# **************************************************


def merge_eigcov_right(tensors, *args, **kwargs):
    _, _, vt = torch.linalg.svd(tensors, full_matrices=False)
    wbar = tensors.sum(dim=0)
    vbar = torch.bmm(vt.transpose(1, 2), vt).sum(dim=0)
    return wbar @ torch.linalg.pinv(vbar)


def merge_eigcov_left(tensors, *args, **kwargs):
    T, Do, Di = tensors.shape
    u, s, vt = torch.linalg.svd(
        tensors, full_matrices=False
    )  # u:(T,Do,R) s:(T,R) vt:(T,R,Di)
    R = s.shape[-1]
    u = u[:, :, :R]
    vt = vt[:, :R, :]

    # (T, Do, R) -> (Do, T, R) ->(Do, T*R)
    uc = u.permute(1, 0, 2).reshape(Do, T * R)
    # (T, R, Di) -> (T*R, Di)
    vct = vt.reshape(T * R, Di)
    sc = torch.diag(s.reshape(-1))
    uc_pinv = torch.linalg.pinv(uc)
    wm = uc_pinv.T @ sc @ vct  # (Do,Di)
    return wm


def merge_eigcov_sylvester_v1(tensors, *args, **kwargs):
    # Solve sylvester equation: AW + WB = C
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    a = torch.bmm(u, u.transpose(1, 2)).sum(dim=0)
    b = torch.bmm(vt.transpose(1, 2), vt).sum(dim=0)
    c = tensors.sum(dim=0) * 2
    wm = torch.from_numpy(
        solve_sylvester(a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy())
    ).to(tensors.device)
    return wm


def merge_eigcov_sylvester_v2(tensors, *args, **kwargs):
    # Solve sylvester equation: AW + WB = C,
    # using simplified method as A,B are symmetric
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    a = torch.bmm(u, u.transpose(1, 2)).sum(dim=0)
    b = torch.bmm(vt.transpose(1, 2), vt).sum(dim=0)
    wbar = tensors.sum(dim=0)
    ua, _, _ = torch.linalg.svd(a, full_matrices=False)
    ub, _, _ = torch.linalg.svd(b, full_matrices=False)
    return ua @ ua.T @ wbar @ ub @ ub.T


def merge_eigcov_sylvester_v3(tensors, epsilon=1e-4, *args, **kwargs):
    # Solve sylvester equation: AW + WB = C
    # using SVD of projectors
    N, Do, Di = tensors.shape
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    uc = u.permute(1, 0, 2).reshape(Do, N * u.shape[2])
    vc = vt.transpose(1, 2).reshape(Di, N * u.shape[2])

    # get psi_u and psi_v
    psi_u, lambda_u, _ = torch.linalg.svd(uc, full_matrices=False)
    psi_v, lambda_v, _ = torch.linalg.svd(vc, full_matrices=False)

    # Extract diags
    lambda_uv = lambda_u.square().unsqueeze(1) + lambda_v.square().unsqueeze(0)

    w_bar = tensors.sum(dim=0) * 2
    c_tilde = (psi_u.T @ w_bar @ psi_v) / (lambda_uv + epsilon)

    return psi_u @ c_tilde @ psi_v.T


def merge_eigcov_sylvester_v4(tensors, *args, **kwargs):
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    pu_tilde = torch.bmm(u, u.transpose(1, 2)).sum(dim=0)
    pv_tilde = torch.bmm(vt.transpose(1, 2), vt).sum(dim=0)

    # svd of projectors
    psi_u, lambda_u, _ = torch.linalg.svd(pu_tilde, full_matrices=True)
    psi_v, lambda_v, _ = torch.linalg.svd(pv_tilde, full_matrices=True)

    # # extract diags
    # lambda_uv = lambda_u.unsqueeze(1) + lambda_v.unsqueeze(0)
    wbar = tensors.sum(dim=0)
    return psi_u @ psi_u.T @ wbar @ psi_v @ psi_v.T


# ================================================
#              Alternate Methods
# ================================================


def merge_punity(tensors: torch.Tensor, **kwargs):
    # partition of unity method
    # W = \sum_i Wi Qi (\sum_j Qj)^-1
    q = torch.bmm(tensors.transpose(1, 2), tensors)  # (N, Di, Di)
    qbar = q.sum(dim=0)
    return torch.bmm(tensors, q).sum(dim=0) @ torch.linalg.pinv(qbar)


def merge_punity_v2(tensors: torch.Tensor, **kwargs):
    # 'avg_normalized_top1': np.float64(0.9764140206523543),
    # 'avg_top1': np.float64(0.916471214218732)}}
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    vt = vt[:, : s.size(1)]
    c = torch.einsum("nir,nr,nrj->nij", vt.transpose(1, 2), s, vt)
    cbar = c.sum(dim=0)
    return torch.bmm(tensors, c).sum(dim=0) @ torch.linalg.pinv(cbar)


def merge_punity_v3(tensors: torch.Tensor, **kwargs):
    #    'avg_normalized_top1': np.float64(0.5901715801772506),
    #     'avg_top1': np.float64(0.5559191226042289)}}
    winv = torch.linalg.pinv(tensors)
    winv_bar = winv.sum(dim=0)
    return torch.bmm(tensors, winv).sum(dim=0) @ torch.linalg.pinv(winv_bar)


def merge_punity_v4(tensors: torch.Tensor, **kwargs):
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    vt = vt[:, : s.size(1)]
    c = torch.einsum("nir,nr,nrj->nij", vt.transpose(1, 2), s.square(), vt)
    cbar = c.sum(dim=0)
    return torch.bmm(tensors, c).sum(dim=0) @ torch.linalg.pinv(cbar)


def merge_punity_rand(tensors: torch.Tensor, **kwargs):
    N, Do, Di = tensors.shape
    r = torch.randn(N, Di, Di // N, device=tensors.device)
    pi = torch.einsum("nik,njk->nij", r, r)
    return torch.bmm(tensors, pi).sum(dim=0)


def merge_punity_hard(tensors: torch.Tensor, **kwargs):
    """
    Solves Max Sum tr(Pi_t W_t^T W_t) s.t. Pi_i Pi_j = 0

    Args:
        tensors: Shape (N, Do, Di) - The weight matrices W_t
    """
    # 1. Compute Energy Matrices Q_t
    # Under assumption C_t = V_t V_t^T, Q_t simplifies to W_t^T @ W_t
    # Shape: (N, Di, Di)
    Qs = torch.bmm(tensors.transpose(1, 2), tensors)

    # 2. Find Global Basis (Voting Booth)
    # We diagonalize the Sum of Energies to find the common vector space
    Q_total = Qs.sum(dim=0)

    # Linalg.eigh is for symmetric matrices (returns eigenvalues, eigenvectors)
    # V shape: (Di, Di), columns are eigenvectors
    eigenvalues, V = torch.linalg.eigh(Q_total)

    # 3. Calculate "Vote" for each Task per Eigenvector
    # We project each task's energy onto this global basis V.
    # We want scalar E_ti = v_i^T Q_t v_i
    # This is the diagonal of V^T Q_t V

    # rotated_Qs = V^T @ Qs @ V
    # Shape: (N, Di, Di)
    Vt = V.transpose(0, 1)  # (Di, Di)
    rotated_Qs = torch.matmul(Vt, torch.matmul(Qs, V))

    # We only care about the diagonal (energy along the axes)
    # Shape: (N, Di)
    task_energies = torch.diagonal(rotated_Qs, dim1=-2, dim2=-1)

    # 4. Winner Take All (Argmax)
    # winners[i] = index of the task that owns eigenvector i
    # Shape: (Di,)
    winners = torch.argmax(task_energies, dim=0)

    # 5. Construct Projectors and Stitch
    # W_star = Sum (W_t @ Pi_t)
    # Pi_t = V @ Mask_t @ V^T

    W_merged = torch.zeros_like(tensors[0])

    for t in range(tensors.shape[0]):
        # Boolean mask: 1 if Task t owns the eigenvector, 0 otherwise
        mask = (winners == t).float()

        # Construct Pi_t implicitly efficiently:
        # Pi_t = V * mask * V^T
        # We can skip full matrix construction by applying V, scaling cols, then V^T

        # Operation: W_merged += W_t @ (V @ diag(mask) @ V^T)

        # 1. Rotate W_t into eigenbasis: W_v = W_t @ V
        W_v = tensors[t] @ V

        # 2. Apply Mask (Kill columns that don't belong to this task)
        W_v_masked = W_v * mask.unsqueeze(0)  # Broadcast over output dim

        # 3. Rotate back: W_contribution = W_v_masked @ V^T
        W_contribution = W_v_masked @ Vt

        W_merged += W_contribution

    return W_merged


def merge_spectral_scalar(tensors: torch.Tensor, mode="soft", epsilon=1e-6, **kwargs):
    """
    Optimizes the 'Sigmas' (Eigenvalues) directly in the spectral domain.

    Args:
        tensors: (N, Do, Di) - Weight matrices
        mode: 'hard' (Argmax) or 'soft' (Wiener Ratio)
    """
    # 1. Compute Energy Matrices Q_t
    # Q_t = W^T @ W (Assuming C_t aligns or is identity)
    # Shape: (N, Di, Di)
    Qs = torch.bmm(tensors.transpose(1, 2), tensors)

    # 2. Find the Common Basis (V)
    # We diagonalize the TOTAL energy to define the "World Coordinates"
    Q_total = Qs.sum(dim=0)

    # evals: (Di,), V: (Di, Di)
    total_evals, V = torch.linalg.eigh(Q_total)

    # 3. Project Task Energies onto this Basis
    # We want lambda_{t,i} = v_i^T @ Q_t @ v_i
    # This is the diagonal of V^T @ Q_t @ V
    # Shape: (N, Di, Di)
    Vt = V.T
    Q_rotated = torch.matmul(Vt, torch.matmul(Qs, V))

    # Extract the diagonals (The scalar energies per mode)
    # Shape: (N, Di) -> Each row is a task, each col is an eigenmode
    lambdas = torch.diagonal(Q_rotated, dim1=-2, dim2=-1)

    # 4. Optimize Sigmas (The "Mixing" Step)
    if mode == "hard":
        # Hard Constraint: Winner Take All
        # sigma = 1.0 if max, 0.0 else
        # Shape: (N, Di)
        winners = torch.argmax(lambdas, dim=0)
        sigmas = torch.zeros_like(lambdas)
        # Use scatter to set the 1s
        sigmas.scatter_(0, winners.unsqueeze(0), 1.0)

    elif mode == "soft":
        # Soft Constraint: Wiener Filter (Ratio)
        # sigma_t = lambda_t / sum(lambda)
        total_lambda = lambdas.sum(dim=0, keepdim=True) + epsilon
        sigmas = lambdas / total_lambda

    # 5. Reconstruct Projectors & Merge
    # Pi_t = V @ diag(sigma_t) @ V^T

    W_merged = torch.zeros_like(tensors[0])

    for t in range(tensors.shape[0]):
        # Construct Diagonal Matrix Sigma_t
        Sigma_t = torch.diag(sigmas[t])

        # Pi_t = V @ Sigma_t @ V^T
        Pi_t = V @ Sigma_t @ Vt

        # Merge
        W_merged += tensors[t] @ Pi_t

    return W_merged


def merge_glowhite(tensors: torch.Tensor, **kwargs):
    # glowhite method
    u, s, vt = torch.linalg.svd(tensors, full_matrices=False)
    v = vt.transpose(1, 2)
    cx = torch.bmm(v, vt)
    cbar = cx.sum(dim=0)
    wm = torch.einsum("nij,jk,nkl->nil", tensors, matrix_pow(cbar, -0.5), cx).sum(
        dim=0
    ) @ torch.linalg.pinv(cbar)
    return wm


# **************************************************
#                Mixtures
# **************************************************


def merge_mix(tensors, layer_name, n2f):
    for k, v in n2f.items():
        if k in layer_name:
            return v(tensors)
    return n2f["default"](tensors)


# **************************************************
#                Isotropic
# **************************************************


def merge_isoc(tensors, *args, **kwargs):
    m = tensors.sum(dim=0)
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    s_mean = s.mean() * torch.ones_like(s)
    return torch.einsum("ik,k,kj->ij", u, s_mean, vt)


# **************************************************
#                Miscellaneous
# **************************************************


def merge_normalized_mean(ws, **kwargs):
    # normalize each row of ws
    ws_prime = ws / torch.linalg.norm(ws, dim=-1, keepdim=True)
    return ws_prime.mean(dim=0)


def merge_weighted_mean(ws, *args, **kwargs):
    """
    ws: (num_models, d, d) - batch of weight matrices
    returns: (d, d) - merged weight matrix biased towards angles
    """
    EPS = 1e-8

    # 1. compute gram matrices: g = w^T @ w
    # shape: (n, d, d)
    g = torch.bmm(ws.transpose(1, 2), ws)

    # 2. compute m = g^(-1/2) using eigendecomposition
    # since g is symmetric, eigh is stable.
    # l: eigenvalues (n, d), v: eigenvectors (n, d, d)
    l, v = torch.linalg.eigh(g)

    # inverse square root of eigenvalues
    # shape: (n, d)
    inv_sqrt_l = 1.0 / torch.sqrt(torch.clamp(l, min=EPS))

    # reconstruct m = v @ diag(l^-1/2) @ v^T
    # v * inv_sqrt_l.unsqueeze(1) scales each column i by l_i
    m = torch.bmm(v * inv_sqrt_l.unsqueeze(1), v.transpose(1, 2))

    # 3. compute the numerator: sum(w @ m) -> sum(orthogonal components)
    # ws @ m extracts the pure rotation (polar decomposition)
    sum_ortho = torch.bmm(ws, m).sum(dim=0)

    # 4. compute the denominator: sum(m)
    sum_weights = m.sum(dim=0)

    # 5. solve: w_star = sum_ortho @ (sum_weights)^-1
    w_star = sum_ortho @ torch.linalg.inv(sum_weights)

    return w_star


def polar_decomp_single(m):
    # Fix: full_matrices=False allows rectangular inputs (32x64)
    # U: (32, 32), S: (32,), Vh: (32, 64) -> U @ Vh is (32, 64)
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)

    u = U @ Vh
    p = Vh.T.conj() @ S.diag().to(dtype=m.dtype) @ Vh
    return u, p


def merge_decoupled(ws, *args, **kwargs):
    # 1. Vectorize with vmap
    batch_polar_decomp = torch.vmap(polar_decomp_single, in_dims=0)
    qs, ps = batch_polar_decomp(ws)  # (N, Do, Di), (N, Di, Di)

    # 2. Solve Rotation: Procrustes Mean
    # Fix: full_matrices=False here too for the reconstruction
    u_sum, _, vh_sum = torch.linalg.svd(qs.sum(dim=0), full_matrices=False)
    q_star = u_sum @ vh_sum

    # 3. Solve Magnitude: Arithmetic Mean
    p_star = ps.mean(dim=0)

    # 4. Recombine
    return q_star @ p_star


def merge_wa_norm_pres(tensors: torch.Tensor, **kwargs) -> torch.Tensor:
    """Plain mean. Ws_delta: (T, Do, Di)"""
    N, Do, Di = tensors.shape
    return tensors.sum(dim=0) / math.sqrt(N)


def compute_opmerge_task_vector(task_vectors, config, *args, **kwargs):
    """Computes the OpMerge task vector.

    Args:
        task_vectors (List[Dict]): A list of task vector objects (state dicts)
        config (Object): Contains the following attributes: [DATASETS, device]
    """
    output_vector = {}
    device = config.device

    merge_func = {
        # **** Baseline Methods ****
        "avg": merge_avg,
        "ta": merge_ta,
        "tsv": merge_tsv,
        "isoc": merge_isoc,
        # **** TSV Methods ****
        "tsv_v1": merge_tsv_v1,
        "tsv_v2": merge_tsv_v2,
        "tsv_v3": merge_tsv_v3,
        # **** Eigen Covariance Methods ****
        "eigcov_right": merge_eigcov_right,
        "eigcov_left": merge_eigcov_left,
        "eigcov_sylvester_v1": merge_eigcov_sylvester_v1,
        "eigcov_sylvester_v2": merge_eigcov_sylvester_v2,
        "eigcov_sylvester_v3": merge_eigcov_sylvester_v3,
        "eigcov_sylvester_v4": merge_eigcov_sylvester_v4,
        # **** Alternate Methods ****
        "punity": merge_punity,
        "punity_v2": merge_punity_v2,
        "punity_v3": merge_punity_v3,
        "punity_v4": merge_punity_v4,
        "punity_rand": merge_punity_rand,
        "punity_hard": merge_punity_hard,
        "glowhite": merge_glowhite,
        "spectral_scalar": merge_spectral_scalar,
        # **** Mixture Methods ****
        # mixes
        # proj: identity (avg), fc: onehot (ta)
        # mixing wa/ta with wa fallback
        "mix_wa_ta_wa": lambda x, **kw: merge_mix(
            x, n2f={"c_proj": merge_avg, "c_fc": merge_ta, "default": merge_avg}, **kw
        ),
        "mix_wa_ta_wa_abl": lambda x, **kw: merge_mix(
            x, n2f={"c_proj": merge_ta, "c_fc": merge_avg, "default": merge_avg}, **kw
        ),
        # mixing wa/ta with ta fallback
        "mix_wa_ta_ta": lambda x, **kw: merge_mix(
            x, n2f={"c_proj": merge_avg, "c_fc": merge_ta, "default": merge_ta}, **kw
        ),
        "mix_wa_ta_ta_abl": lambda x, **kw: merge_mix(
            x, n2f={"c_proj": merge_ta, "c_fc": merge_avg, "default": merge_ta}, **kw
        ),
        # TSV applied to one layer type at a time
        "mix_tsv_ta_l1": lambda x, **kw: merge_mix(
            x, n2f={"c_proj": merge_tsv, "default": merge_ta}, **kw
        ),
        "mix_tsv_ta_l2": lambda x, **kw: merge_mix(
            x, n2f={"c_fc": merge_tsv, "default": merge_ta}, **kw
        ),
        "mix_tsv_ta_l3": lambda x, **kw: merge_mix(
            x, n2f={"in_proj": merge_tsv, "default": merge_ta}, **kw
        ),
        "mix_tsv_ta_l4": lambda x, **kw: merge_mix(
            x, n2f={"out_proj": merge_tsv, "default": merge_ta}, **kw
        ),
        "normalized_mean": merge_normalized_mean,
        "weighted_mean": merge_weighted_mean,
        "wa_norm_pres": merge_wa_norm_pres,
        "decoupled": merge_decoupled,
    }  # selected by config.opm

    print(f"***** Using OpMerge {config.opm} *****")

    for layer_name in task_vectors[0].vector.keys():
        tensors = torch.stack([tv.vector[layer_name] for tv in task_vectors]).to(device)

        # If it's 2D we apply the merge function
        layer_tensor_shape = task_vectors[0].vector[layer_name].shape
        if (
            len(layer_tensor_shape) == 2
            and "text_projection" not in layer_name
            and max(layer_tensor_shape) < 10_000
        ):
            print(f"Merging layer {layer_name} with {config.opm}")
            w = merge_func[config.opm](tensors, layer_name=layer_name)
            output_vector[layer_name] = w
        # if not 2D we compute the mean
        else:
            if len(layer_tensor_shape) == 2 and max(layer_tensor_shape) >= 10_000:
                print(f"Averageing layer {layer_name} because it's larger than 10k")
            output_vector[layer_name] = torch.mean(tensors, dim=0)

    return output_vector


if __name__ == "__main__":
    task_vectors = [
        {
            "linear1": torch.randn(10, 10),
            "linear2": torch.randn(10, 10),
            "dot": torch.randn(8),
        },
        {
            "linear1": torch.randn(10, 10),
            "linear2": torch.randn(10, 10),
            "dot": torch.randn(8),
        },
    ]


# """Results

# >>> model=ViT-B-16 method="opmerge" DATASETS=[Cars,DTD,EuroSAT,GTSRB,MNIST,SVHN] n_eval_points=3 opm=lsopt_gd
#  'val_best': {'CarsVal:normalized_top1': 0.0028409090909090906,
#               'CarsVal:top1': 0.002457002457002457,
#               'DTDVal:normalized_top1': 0.032397408207343416,
#               'DTDVal:top1': 0.026595744680851064,
#               'EuroSATVal:normalized_top1': 0.09999999999999999,
#               'EuroSATVal:top1': 0.09925925925925926,
#               'GTSRBVal:normalized_top1': 0.027047332832456798,
#               'GTSRBVal:top1': 0.02702702702702703,
#               'MNISTVal:normalized_top1': 0.10231155778894473,
#               'MNISTVal:top1': 0.1018,
#               'SVHNVal:normalized_top1': 0.0979743695742042,
#               'SVHNVal:top1': 0.0948,
#               'avg_normalized_top1': np.float64(0.06042859624897637),
#               'avg_top1': np.float64(0.058656505570689965)}}
# >>> model=ViT-B-16 method="opmerge" DATASETS=[Cars,DTD,EuroSAT,GTSRB,MNIST,SVHN] n_eval_points=3 opm=tsv
#  'val_best': {'CarsVal:normalized_top1': 0.9161931818181818,
#               'CarsVal:top1': 0.7923832923832924,
#               'DTDVal:normalized_top1': 1.033477321814255,
#               'DTDVal:top1': 0.848404255319149,
#               'EuroSATVal:normalized_top1': 0.9630597014925374,
#               'EuroSATVal:top1': 0.955925925925926,
#               'GTSRBVal:normalized_top1': 0.9725770097670924,
#               'GTSRBVal:top1': 0.9718468468468469,
#               'MNISTVal:normalized_top1': 0.9941708542713568,
#               'MNISTVal:top1': 0.9892,
#               'SVHNVal:normalized_top1': 0.9247622984704423,
#               'SVHNVal:top1': 0.8948,
#               'avg_normalized_top1': np.float64(0.9673733946056443),
#               'avg_top1': np.float64(0.9087600534125357)}}
# >>> model=ViT-B-16 method="opmerge" DATASETS=[Cars,DTD,EuroSAT,GTSRB,MNIST,SVHN] n_eval_points=3 opm=tsv_iso
#  'val_best': {'CarsVal:normalized_top1': 0.8877840909090909,
#               'CarsVal:top1': 0.7678132678132679,
#               'DTDVal:normalized_top1': 0.8293736501079914,
#               'DTDVal:top1': 0.6808510638297872,
#               'EuroSATVal:normalized_top1': 0.9201492537313433,
#               'EuroSATVal:top1': 0.9133333333333333,
#               'GTSRBVal:normalized_top1': 0.9346356123215628,
#               'GTSRBVal:top1': 0.933933933933934,
#               'MNISTVal:normalized_top1': 0.9752763819095478,
#               'MNISTVal:top1': 0.9704,
#               'SVHNVal:normalized_top1': 0.774286895411327,
#               'SVHNVal:top1': 0.7492,
#               'avg_normalized_top1': np.float64(0.8869176473984773),
#               'avg_top1': np.float64(0.8359219331517204)}}

# model=ViT-B-16 method="opmerge" DATASETS=[Cars,DTD,EuroSAT,GTSRB,MNIST,SVHN] n_eval_points=3 opm=tsv_perm
#  'val_best': {'CarsVal:normalized_top1': 0.8806818181818181,
#               'CarsVal:top1': 0.7616707616707616,
#               'DTDVal:normalized_top1': 0.826133909287257,
#               'DTDVal:top1': 0.6781914893617021,
#               'EuroSATVal:normalized_top1': 0.9194029850746269,
#               'EuroSATVal:top1': 0.9125925925925926,
#               'GTSRBVal:normalized_top1': 0.9312546957175056,
#               'GTSRBVal:top1': 0.9305555555555556,
#               'MNISTVal:normalized_top1': 0.9780904522613065,
#               'MNISTVal:top1': 0.9732,
#               'SVHNVal:normalized_top1': 0.771186440677966,
#               'SVHNVal:top1': 0.7462,
#               'avg_normalized_top1': np.float64(0.8844583835334134),
#               'avg_top1': np.float64(0.8337350665301021)}}
# # """
