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
    q = torch.bmm(tensors.transpose(1, 2), tensors)  # (N, Di, Di)
    qbar = q.sum(dim=0)
    return torch.bmm(tensors, q).sum(dim=0) @ torch.linalg.pinv(qbar)


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


def merge_ta_wa_mix(tensors, layer_name, *args, **kwargs):
    """Computes the TSV merge of the given tensors.

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    if "proj" in layer_name:  # closer to diagonal
        return merge_avg(tensors, *args, **kwargs)
    else:
        return merge_ta(tensors, *args, **kwargs)


def merge_ta_wa_tsv_mix(tensors, layer_name, *args, **kwargs):
    """Computes the TSV merge of the given tensors.

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    if "c_proj" in layer_name:  # closer to diagonal
        return merge_tsv(tensors, *args, **kwargs)
    elif "c_fc" in layer_name:  # closer to onehot
        return merge_ta(tensors, *args, **kwargs)
    else:
        return merge_tsv(tensors, *args, **kwargs)


def merge_tsv_wa_mix(tensors, layer_name, *args, **kwargs):
    """Computes the TSV merge of the given tensors.

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    if "proj" in layer_name:  # closer to diagonal
        return merge_avg(tensors, *args, **kwargs)
    else:
        return merge_tsv(tensors, *args, **kwargs)


def merge_tsv_wa_mix_abl(tensors, layer_name, *args, **kwargs):
    """Computes the TSV merge of the given tensors.

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    if "proj" in layer_name:  # closer to diagonal
        return merge_tsv(tensors, *args, **kwargs)
    else:
        return merge_avg(tensors, *args, **kwargs)


def merge_mix(tensors, layer_name, n2f):
    for k, v in n2f.items():
        if k in layer_name:
            return v(tensors)
    return n2f["default"](tensors)


def merge_ta_wa_mix_abl(tensors, layer_name, **kwargs):
    """Computes the TSV merge of the given tensors.

    Args:
        tensors (torch.Tensor): The tensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    if "proj" in layer_name:  # closer to diagonal
        return merge_ta(tensors, **kwargs)
    else:
        return merge_avg(tensors, **kwargs)


# **************************************************
#                Isotropic
# **************************************************


def merge_isoc(tensors, *args, **kwargs):
    m = tensors.sum(dim=0)
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    s_mean = s.mean() * torch.ones_like(s)
    return torch.einsum("ik,k,kj->ij", u, s_mean, vt)


def merge_lsopt_lstsq(a_tilde, verbose=False, *args, **kwargs):
    """
    Args:
        a_tilde (torch.Tensor): Task matrices. Shape: (B, Do, Di)
        verbose (bool): Whether to print debug info

    Returns:
        a_hat (torch.Tensor): Merged task matrix. Shape: (Do, Di)
    """
    # B, Do, Di = 8, 32, 64
    # a_tilde = torch.randn(B, Do, Di)
    B, Do, Di = a_tilde.shape
    u, _, vt = torch.linalg.svd(a_tilde, full_matrices=False)
    device = a_tilde.device

    # Keep only the first Rp columns of u and vt
    R = min(u.shape[1], vt.shape[2])
    Rp = R // len(a_tilde)
    u = u[:, :, :Rp]
    vt = vt[:, :Rp, :]

    ut = u.permute(0, 2, 1)
    v = vt.permute(0, 2, 1)
    pus = torch.einsum("bij,bjk->bik", u, ut)
    pvs = torch.einsum("bij,bjk->bik", v, vt)

    mis_stack, a_tilde_flat = make_ls_mats(pus, pvs, a_tilde)
    if verbose:
        print("pus.shape", pus.shape)
        print("pvs.shape", pvs.shape)
        print("batched_kron.shape", batched_kron(pus, pvs).shape)
        print("mis_stack.shape", mis_stack.shape)
        print("a_tilde_flat.shape", a_tilde_flat.shape)

    print("Computing LS solution...")
    print("mis_stack.shape", mis_stack.shape)
    print("a_tilde_flat.shape", a_tilde_flat.shape)
    sol = torch.linalg.lstsq(mis_stack.to("cpu"), a_tilde_flat.to("cpu"))
    print("Done.")
    a_hat_vec = sol.solution
    a_hat = a_hat_vec.reshape(Do, Di)
    return a_hat.to(device)


def merge_lsopt_gd(a_tilde, verbose=True, max_iter=1000, tol=1e-4, lr=0.01):
    B, Do, Di = a_tilde.shape
    device = a_tilde.device

    # mis_stack, a_tilde_flat = make_ls_mats(pus, pvs, a_tilde)
    a_hat = torch.nn.Parameter(torch.randn(Do, Di, device=device, dtype=a_tilde.dtype))
    opt = torch.optim.AdamW([a_hat], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", patience=10, factor=0.1
    )

    loss_prev = float("inf")
    # pbar = tqdm(range(max_iter), desc="LSopt GD", leave=False)
    for it in range(max_iter):
        opt.zero_grad()
        # pred = mis_stack @ a_hat.reshape(-1)
        # loss = torch.nn.functional.mse_loss(pred, a_tilde_flat)  # default = mean
        # loss = compute_lsopt_loss(a_hat, mis_stack, a_tilde_flat)
        loss = compute_lsopt_loss_gd(a_hat, a_tilde)

        loss.backward()
        opt.step()
        sched.step(loss.item())

        lv = loss.item()
        # pbar.set_postfix(loss=lv)
        if verbose and it % 50 == 0:
            print(f"iter {it}, loss {lv}")
            # print shapes
            print(f"a_hat.shape: {a_hat.shape} | a_tilde.shape: {a_tilde.shape}")

        if abs(lv - loss_prev) < tol:
            if verbose:
                print(f"Converged in {it} iterations")
            break
        loss_prev = lv

    return a_hat.detach()


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


def merge_mci(ws, mode="right", **kwargs):
    """Merges task vectors using the min constructive interference method.

    Args:
        us (torch.Tensor): (N, Do, Di)

    Returns:
        torch.Tensor: (Do, Di)
    """
    N, Do, Di = ws.shape
    w_bar = ws.sum(dim=0)
    us, _, vts = torch.linalg.svd(ws, full_matrices=True)
    if mode == "left":
        R = Do // N
        us = us[:, :, :R]
        p_k = torch.bmm(us, us.transpose(1, 2))  # (N, Do, Do)
        p_bar = p_k.sum(dim=0)
        p_pinv = torch.linalg.pinv(p_bar)
        w_m = p_pinv @ w_bar
        print("[Left] MCI: Do =", Do, "N =", N, "R =", R)
    elif mode == "right":
        R = Di // N
        vts = vts[:, :R, :]
        p_k = torch.bmm(vts.transpose(1, 2), vts)  # (N, Do, Do)
        p_bar = p_k.sum(dim=0)
        p_pinv = torch.linalg.pinv(p_bar)
        w_m = w_bar @ p_pinv
        print("[Right] MCI: Do =", Do, "N =", N, "R =", R)
    return w_m


def merge_min_angle(ws, mode="rows", *args, **kwargs):
    """Merges task vectors using the min angle method. I.e., (11ᵀ ΣₖWₜᵀ)(Σₜ Wₜᵀ Wₜ)^†.

    Args:
        ws_raw (torch.Tensor): (N, Do, Di)

    Returns:
        torch.Tensor: (Do, Di)
    """
    # Normalize rows
    N, Do, Di = ws.shape

    if mode == "rows":
        w_tilde = ws / ws.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        w_tilde_bar = w_tilde.sum(dim=0)
        wtw_tilde_bar = torch.bmm(w_tilde.transpose(1, 2), w_tilde).sum(dim=0)
        return (
            torch.ones(Do, Do, device=ws.device)
            @ w_tilde_bar
            @ torch.linalg.pinv(wtw_tilde_bar)
        )
    else:
        w_tilde = ws / ws.norm(dim=-2, keepdim=True).clamp_min(1e-6)
        w_tilde_bar = w_tilde.sum(dim=0)
        wwt_tilde_bar = torch.bmm(w_tilde, w_tilde.transpose(1, 2)).sum(dim=0)
        return (
            torch.linalg.pinv(wwt_tilde_bar)
            @ w_tilde_bar
            @ torch.ones(Di, Di, device=ws.device)
        )


def merge_proj(tensors, mode="right", *args, **kwargs):
    """Merges task vectors using the min constructive interference method.

    Args:
        us (torch.Tensor): (N, Do, Di)

    Returns:
        torch.Tensor: (Do, Di)
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
    u_ortho = compute_procrustes(u_hat)  # (Di, B*Rp)
    vt_ortho = compute_procrustes(vt_hat.T).T  # (B*Rp, Do)
    # tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)
    p_vo = torch.einsum("ij,jk->ik", vt_ortho.T, vt_ortho)
    p_uo = torch.einsum("ij,jk->ik", u_ortho, u_ortho.T)
    if mode == "right":
        tau_l = torch.einsum("bij,bj,bjk,km->bim", u, s, vt, p_vo).sum(dim=0)
    elif mode == "left":
        tau_l = torch.einsum("ni,bij,bj,bjk->bnk", p_uo, u, s, vt).sum(dim=0)
    else:  # mode == "both"
        tau_l = torch.einsum("ni,bij,bj,bjk,km->bnm", p_uo, u, s, vt, p_vo).sum(dim=0)
    # TSV:
    # tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)
    # Proj:
    # tau_l = torch.einsum("ni,bij,bj,bjk,km->bnm", p_uo, u, s, vt, p_vo).sum(dim=0)
    return tau_l


def merge_aproptwa(ws, *args, **kwargs):
    """Approximately optimal WA."""
    N, Do, Di = ws.shape
    E = torch.einsum("pij,qji", ws, ws.transpose(1, 2))
    alpha = torch.linalg.pinv(E) @ E.sum(dim=-1)
    return 1 / N * torch.einsum("nij,n->ij", ws, alpha)


@torch.no_grad()
def merge_wa_asym(
    Ws: torch.Tensor,  # (T, Do, Di) real-valued
    k: int = 1024,  # rank of shared subspace
    n_splits: int = 4,  # number of random entrywise splits to average
    p_split: float = 0.5,  # split prob
    seed: int = 0,
    return_debug: bool = False,
    **kwargs,
):
    """
    Rectangular multi-task combining via an Asymmetric-PCA-style shared subspace.

    Ws: (T, Do, Di)

    High-level (rectangular version):
      - For each split s and task t:
          W = W1 + W2 via entrywise Bernoulli mask
          X_t = W1 @ W2.T    (Do x Do)
          Y_t = W2.T @ W1    (Di x Di)
      - Average over tasks and splits:
          Xbar = mean_{s,t} X_t,  Ybar = mean_{s,t} Y_t
      - Compute top-k eigenpairs of Xbar and Ybar
      - Average left+right eigenvectors (left from eig of A.T) for stability
      - Orthonormalize to get U (Do x k), V (Di x k)
      - Per task core: S_t = U.T @ W_t @ V   (k x k)
      - Merge core: Sbar = mean_t S_t
      - Return merged: W_merge = U @ Sbar @ V.T   (Do x Di)

    Collapses to "single W denoise" when T=1 (rank-k projection onto shared U,V).
    """

    assert Ws.ndim == 3, "Ws must be (T, Do, Di)"
    T, Do, Di = Ws.shape
    device = Ws.device
    dtype = Ws.dtype

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def _split(W: torch.Tensor):
        M = (torch.rand((Do, Di), generator=g, device=device) < p_split).to(dtype)
        return W * M, W * (1.0 - M)

    def _topk_avg_lr_evecs(A: torch.Tensor, k_use: int):
        """
        Returns Q (N x k_use) real orthonormal basis from averaged left/right eigenvectors,
        plus the selected eigenvalues (complex).
        """
        # Right eigenvectors of A
        evals_r, evecs_r = torch.linalg.eig(A)  # complex
        # "Left" eigenvectors are right eigenvectors of A.T
        evals_l, evecs_l = torch.linalg.eig(A.T)  # complex

        # Pick top-k by |lambda| (simple outlier proxy)
        idx_r = torch.topk(evals_r.abs().real, k=k_use, largest=True).indices
        lam = evals_r[idx_r]

        Z = []
        for j in idx_r.tolist():
            lr = evals_r[j]
            # match by closest eigenvalue in complex plane
            iL = torch.argmin((evals_l - lr).abs())
            vR = evecs_r[:, j]
            vL = evecs_l[:, iL]

            # Optional: crude phase/sign alignment to reduce cancellation
            ip = torch.vdot(vL, vR)  # conj(vL)^T vR
            if ip.abs() > 1e-12:
                vL = vL * torch.conj(ip / (ip.abs() + 1e-12))

            z = (vR + vL).real
            z = z / (z.norm() + 1e-12)
            Z.append(z)

        Z = torch.stack(Z, dim=1)  # (N, k_use)

        # Orthonormalize (stabilizes downstream core extraction)
        Q, _ = torch.linalg.qr(Z)
        return Q[:, :k_use], lam

    # Build averaged asymmetric matrices (Do x Do) and (Di x Di)
    Xbar = torch.zeros((Do, Do), device=device, dtype=dtype)
    Ybar = torch.zeros((Di, Di), device=device, dtype=dtype)

    for _ in range(n_splits):
        for t in range(T):
            W1, W2 = _split(Ws[t])
            Xbar += W1 @ W2.T
            Ybar += W2.T @ W1

    Xbar /= T * n_splits
    Ybar /= T * n_splits

    kU = min(k, Do)
    kV = min(k, Di)
    k_eff = min(kU, kV)

    # Shared bases
    U, lamX = _topk_avg_lr_evecs(Xbar, k_use=k_eff)  # (Do, k_eff)
    V, lamY = _topk_avg_lr_evecs(Ybar, k_use=k_eff)  # (Di, k_eff)

    # Per-task cores and merge
    cores = []
    for t in range(T):
        S_t = U.T @ Ws[t] @ V  # (k_eff, k_eff)
        cores.append(S_t)
    Sbar = torch.stack(cores, dim=0).mean(dim=0)

    Wm = U @ Sbar @ V.T  # (Do, Di)

    if return_debug:
        return Wm, {
            "U": U,
            "V": V,
            "Sbar": Sbar,
            "lamX": lamX,
            "lamY": lamY,
            "Xbar": Xbar,
            "Ybar": Ybar,
            "k_eff": k_eff,
        }
    return Wm


def merge_maxvar(ws, *args, **kwargs):
    """Merges task vectors using the max variance method.

    Args:
        ws (torch.Tensor): (N, Do, Di)

    Returns:
        torch.Tensor: (Do, Di)
    """
    N, Do, Di = ws.shape
    k = torch.einsum("ik,ok->io", ws.reshape(N, -1), ws.reshape(N, -1))  # (N, N)
    # find top eigenvector of k
    e, v = torch.linalg.eigh(k)  # (N, N)
    # v is now used as mixture weights
    alpha = v[:, 0]
    print("alpha", alpha.shape)
    print("ws", ws.shape)
    wm = torch.einsum("n,npq->pq", alpha, ws)
    return wm


def merge_3ltsv(tensors, *args, **kwargs):
    # 1. Compute SVDs: U (B, M, K), S (B, K), Vh (B, K, N)
    u, s, vh = torch.linalg.svd(tensors, full_matrices=False)

    # --- FIX START: Resolve Sign Ambiguity ---
    # We fix the first tensor as the "anchor" and align others to it.
    # Check dot product of columns. If negative, flip signs of U and Vh.
    ref_u = u[0]  # Reference basis
    ref_vh = vh[0]

    for i in range(1, u.shape[0]):
        # Calculate correlations with reference (simple column-wise dot)
        # Note: This is a heuristic. For strict alignment, use Procrustes between u[i] and ref_u
        dot_prod = (u[i] * ref_u).sum(dim=0)  # Shape (K,)
        signs = torch.sign(dot_prod)  # +1 or -1

        # Apply flips to U and Vh (S is always positive)
        u[i] = u[i] * signs.view(1, -1)
        vh[i] = vh[i] * signs.view(-1, 1)
    # --- FIX END ---

    # 2. Average the "Linear" part
    smean = s.mean(dim=0)

    # 3. Average the "Manifold" parts (Projected Mean)
    # Note: u.sum is now safe because signs are aligned
    ubar = u.sum(dim=0)
    vbar = vh.sum(dim=0)  # vh is V^T

    # 4. Project back to Manifold (Procrustes / Polar Decomposition)
    # U_orth = U (U^T U)^-1/2
    u_orth = compute_polar(ubar)
    v_orth = compute_polar(vbar.T).T

    # 5. Reconstruct
    return u_orth @ torch.diag(smean) @ v_orth


def merge_maor(tensors, magnitude_agnostic=True, **kwargs):
    if magnitude_agnostic:
        u, _, vt = torch.linalg.svd(tensors, full_matrices=False)
        q = torch.bmm(u, vt)
        wbar = q.sum(dim=0)
    else:
        wbar = tensors.sum(dim=0)
    # Orthogonalize
    u, s, vt = torch.linalg.svd(wbar, full_matrices=False)
    return u @ vt


def merge_karcher(tensors, alphas=None, max_iter=10, tol=1e-5, **kwargs):
    """
    Implements weight fusion based on the Riemannian (Karcher) mean concept.

    Args:
        tensors: List of tensors to merge
        alphas: List of weights for each tensor
        max_iter: Maximum number of iterations for the Karcher mean algorithm
        tol: Convergence tolerance

    Returns:
        Merged tensor using Karcher mean algorithm
    """
    if alphas is None:
        alphas = torch.ones(len(tensors))

    if len(tensors) == 1:
        return tensors[0]

    # Calculate norms and unit vectors
    norms = []
    units = []
    for t in tensors:
        t_float = t.float()
        n = torch.linalg.norm(t_float)
        n_val = n.item()
        if n_val == 0.0:
            norms.append(0.0)
            units.append(torch.zeros_like(t))
        else:
            norms.append(n_val)
            units.append((t / n).to(t.dtype))

    # Select non-zero weight vectors
    valid_indices = [i for i, n in enumerate(norms) if n > tol]
    if not valid_indices:
        return torch.zeros_like(tensors[0])

    valid_alphas = [alphas[i] for i in valid_indices]
    alpha_sum = sum(valid_alphas)
    normalized_alphas = [a / alpha_sum for a in valid_alphas]
    valid_units = [units[i] for i in valid_indices]

    # Initial guess: Normalized weighted arithmetic mean
    u = torch.zeros_like(valid_units[0])
    for a, ui in zip(normalized_alphas, valid_units):
        u += a * ui
    norm_u = torch.linalg.norm(u.float()).item()
    if norm_u < tol:
        u = valid_units[0].clone()
    else:
        u = (u / norm_u).to(u.dtype)

    # Iterative Karcher mean computation
    for _ in range(max_iter):
        T = torch.zeros_like(u)
        for a, ui in zip(normalized_alphas, valid_units):
            # Flatten tensor for dot product calculation
            dot = torch.clamp(torch.dot(u.flatten(), ui.flatten()), -1.0, 1.0)
            theta = torch.arccos(dot)
            theta_val = theta.item()
            if theta_val < tol:
                continue
            else:
                # Ensure tensor operations
                sin_theta = torch.sin(theta)
                T += a * (theta / sin_theta) * (ui - dot * u)

        # Convert norm_T to tensor
        norm_T = torch.linalg.norm(T.float())
        if norm_T.item() < tol:
            break

        # Use tensor for trigonometric calculations
        cos_norm_T = torch.cos(norm_T)
        sin_norm_T = torch.sin(norm_T)
        u = (cos_norm_T * u + sin_norm_T * (T / norm_T)).to(u.dtype)

        # Ensure u is a unit vector
        u_norm = torch.linalg.norm(u.float())
        if u_norm.item() > tol:
            u = (u / u_norm).to(u.dtype)

    # Global scale: Weighted sum of original tensor norms (including zero vectors)
    s = 0.0
    for a, n in zip(alphas, norms):
        s += a * n

    return s * u


# =============================================================================
# Conventions (IMPORTANT)
# =============================================================================
# In this file, Ws are *deltas* unless stated otherwise:
#   Ws_delta[t] = W_ft[t] - W_pre
#
# Most merge methods below return a merged DELTA:
#   dW_merge  (Do x Di)
#
# To recover a merged fine-tuned weight matrix, do:
#   W_merge = W_pre + dW_merge
#
# Why work with deltas?
#   - The shared "pretrained" component is factored out.
#   - The merge is trying to estimate a task-agnostic update, while suppressing
#     task-specific spiky directions that hurt OOD.
# =============================================================================


# -----------------------------
# Utilities
# -----------------------------


def _svd(W: torch.Tensor):
    # full_matrices=False is faster and enough for reconstruction
    return torch.linalg.svd(W, full_matrices=False)


def _reconstruct(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
    # U: (Do, r), S: (r,), Vh: (r, Di)
    return (U * S.unsqueeze(0)) @ Vh


def _soft_thresh(x: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)


def _geometric_median_fro(
    X: torch.Tensor, max_iter: int = 50, eps: float = 1e-7
) -> torch.Tensor:
    """
    Geometric median of matrices under Frobenius norm.
    X: (G, Do, Di)
    """
    G = X.shape[0]
    y = X.mean(dim=0)
    for _ in range(max_iter):
        diff = X - y  # (G, Do, Di)
        d = torch.linalg.norm(diff.flatten(1), dim=1).clamp_min(eps)  # (G,)
        w = 1.0 / d
        w = w / w.sum()
        y_new = (w.view(G, 1, 1) * X).sum(dim=0)
        if torch.linalg.norm((y_new - y).flatten()) <= eps * torch.linalg.norm(
            y.flatten()
        ).clamp_min(eps):
            y = y_new
            break
        y = y_new
    return y


def _median_of_means(Ws: torch.Tensor, n_groups: int = 8) -> torch.Tensor:
    """
    Median-of-means in Frobenius space:
    1) partition tasks into groups
    2) average within each group
    3) take geometric median across group-means (Weiszfeld)
    """
    T = Ws.shape[0]
    n_groups = max(1, min(n_groups, T))
    perm = torch.randperm(T, device=Ws.device)
    groups = perm.chunk(n_groups)
    means = torch.stack([Ws[g].mean(dim=0) for g in groups], dim=0)  # (G, Do, Di)
    return _geometric_median_fro(means)


def _ipr(v: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    Inverse Participation Ratio for unit vectors (localization score).
    For a unit vector, IPR = sum_j v_j^4. Larger => more localized.
    """
    v2 = v * v
    denom = v2.sum(dim=dim, keepdim=True).clamp_min(eps)
    v_unit = v / torch.sqrt(denom)
    return (v_unit**4).sum(dim=dim)


def _pick_rank_energy(S: torch.Tensor, energy: float = 0.95) -> int:
    """
    Pick smallest r such that cumulative energy (sum_{k<=r} S_k^2) covers `energy` fraction.
    """
    p = S * S
    c = torch.cumsum(p, dim=0)
    r = int(torch.searchsorted(c, energy * c[-1]).item()) + 1
    return max(1, min(r, S.numel()))


def _cv_lambda_two_split(
    Ws: torch.Tensor, candidate_lams: torch.Tensor, method: str
) -> float:
    """
    Simple 2-split CV to pick lambda based on stability:
    minimize ||f(mean(split1)) - mean(split2)||_F^2 + ||f(mean(split2)) - mean(split1)||_F^2
    method: 'svd_soft' or 'svd_hard' or 'svd_ridge'
    """
    T = Ws.shape[0]
    perm = torch.randperm(T, device=Ws.device)
    a, b = perm[: T // 2], perm[T // 2 :]
    W1 = Ws[a].mean(dim=0)
    W2 = Ws[b].mean(dim=0)

    def apply(W, lam):
        U, S, Vh = _svd(W)
        if method == "svd_soft":
            S2 = torch.clamp(S - lam, min=0.0)
        elif method == "svd_hard":
            S2 = S * (S > lam)
        elif method == "svd_ridge":
            # spectral ridge-ish: sigma -> sigma^3 / (sigma^2 + lam)
            S2 = (S * S * S) / (S * S + lam)
        else:
            raise ValueError(method)
        return _reconstruct(U, S2, Vh)

    best = None
    best_lam = float(candidate_lams[0].item())
    for lam in candidate_lams:
        lamf = float(lam.item())
        A1 = apply(W1, lamf)
        A2 = apply(W2, lamf)
        loss = (
            torch.linalg.norm((A1 - W2).flatten()) ** 2
            + torch.linalg.norm((A2 - W1).flatten()) ** 2
        )
        loss = float(loss.item())
        if best is None or loss < best:
            best = loss
            best_lam = lamf
    return best_lam


def merge_wa_norm_pres(tensors: torch.Tensor, **kwargs) -> torch.Tensor:
    """Plain mean. Ws_delta: (T, Do, Di)"""
    N, Do, Di = tensors.shape
    return tensors.sum(dim=0) / math.sqrt(N)


def merge_preserve_var(Ws: torch.Tensor, **kwargs):
    """
    Ws: List of tensors or (N, Do, Di) tensor.
    Returns: (Do, Di) merged tensor avoiding variance collapse.
    """
    if isinstance(Ws, list):
        Ws = torch.stack(Ws)

    # 1. Consensus Direction (Linear Average)
    W_avg = torch.mean(Ws, dim=0)

    # 2. Target Energy (RMS of input Frobenius norms)
    input_norms = torch.norm(Ws, p="fro", dim=(-2, -1))
    target_norm = torch.sqrt(torch.mean(input_norms**2))

    # 3. Rescale
    current_norm = torch.norm(W_avg, p="fro")
    return W_avg * (target_norm / (current_norm + 1e-8))


# def merge_project_mean(W: torch.Tensor, rtol: float = 1e-6, **kwargs) -> torch.Tensor:
#     """
#     W: (T, Do, Di)
#     returns W_merge: (Do, Di) = P * mean_t W_t, where P projects onto Col([W1..WT]).
#     """
#     T, Do, Di = W.shape

#     # mean
#     Wbar = W.mean(dim=0)  # (Do, Di)

#     # basis U for S = Col([W1..WT])
#     A = W.permute(1, 0, 2).reshape(Do, T * Di)  # (Do, T*Di) = [W1 .. WT]
#     U, S, _ = torch.linalg.svd(A, full_matrices=False)  # U: (Do, k)
#     thr = rtol * S.max()
#     R = int((S > thr).sum().item())
#     U = U[:, :R]  # (Do, R)

#     # project mean onto S
#     Wm = U @ (U.T @ Wbar)  # (Do, Di) = UU^T Wbar

#     return Wm


def merge_project_mean(
    tensors: torch.Tensor,
    rtol: float = 1e-6,
    mode: str = "col",  # "col", "row", or "both"
    low_rank: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    W: (T, Do, Di)

    mode="col":  UU^T @ mean(W)
    mode="row":  mean(W) @ VV^T
    mode="both": UU^T @ mean(W) @ VV^T
    """

    T, Do, Di = tensors.shape
    if low_rank:
        U, S, Vt = torch.linalg.svd(tensors, full_matrices=False)
        # U:  (T, Do, K)
        # S:  (T, K)
        # Vt: (T, K, Di)

        K = min(Do, Di)
        R = min(K, Do // T, Di // T)

        U_r = U[:, :, :R]  # (T, Do, R)
        S_r = S[:, :R]  # (T, R)
        Vt_r = Vt[:, :R, :]  # (T, R, Di)

        W = U_r @ torch.diag_embed(S_r) @ Vt_r  # (T, Do, Di)
    else:
        W = tensors

    Wbar = W.mean(dim=0)  # (Do, Di)
    if mode in ("col", "both"):
        # (T, Do, Di) -> (Do, T, Di) -> (Do, T*Di)
        Acol = W.permute(1, 0, 2).reshape(Do, T * Di)  # (Do, T*Di)
        U, S, _ = torch.linalg.svd(Acol, full_matrices=False)
        thr = rtol * S.max()
        R = int((S > thr).sum().item())
        U = U[:, :R]  # (Do, R)
        # issue warning if U is full rank
        if R == Do:
            print("Warning: U is full rank")

    if mode in ("row", "both"):
        # (T, Do, Di) -> (Di, T, Do) -> (Di, T*Do)
        Arow = W.permute(2, 0, 1).reshape(Di, T * Do)  # (Di, T*Do)
        V, S, _ = torch.linalg.svd(Arow, full_matrices=False)
        thr = rtol * S.max()
        R = int((S > thr).sum().item())
        V = V[:, :R]  # (Di, R)
        if R == Di:
            print("Warning: V is full rank")

    if mode == "col":
        Wm = U @ (U.T @ Wbar)
    elif mode == "row":
        Wm = Wbar @ (V @ V.T)
    elif mode == "both":
        Wm = U @ (U.T @ Wbar) @ (V @ V.T)
    else:
        raise ValueError("mode must be 'col', 'row', or 'both'")

    return Wm


# import torch


# def merge_project_mean(
#     W: torch.Tensor,
#     rtol: float = 1e-6,
#     mode: str = "col",  # "col" or "row"
#     **kwargs,
# ) -> torch.Tensor:
#     """
#     W: (T, Do, Di)

#     mode="col": Wm = P_col @ mean(W)
#     mode="row": Wm = mean(W) @ P_row
#     """
#     T, Do, Di = W.shape
#     Wbar = W.mean(dim=0)  # (Do, Di)

#     if mode == "col":
#         # A = [W1 .. WT]  -> (Do, T*Di)
#         A = W.permute(1, 0, 2).reshape(Do, T * Di)
#         U, S, _ = torch.linalg.svd(A, full_matrices=False)
#         thr = rtol * S.max()
#         R = int((S > thr).sum().item())
#         U = U[:, :R]  # (Do, R)
#         Wm = U @ (U.T @ Wbar)  # (Do, Di)

#     elif mode == "row":
#         # A = [W1^T .. WT^T] -> (Di, T*Do)
#         A = W.permute(2, 0, 1).reshape(Di, T * Do)
#         V, S, _ = torch.linalg.svd(A, full_matrices=False)
#         thr = rtol * S.max()
#         R = int((S > thr).sum().item())
#         V = V[:, :R]  # (Di, R)
#         Wm = Wbar @ (V @ V.T)  # (Do, Di)

#     else:
#         raise ValueError("mode must be 'col' or 'row'")

#     return Wm


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


# -----------------------------
# Basic merge methods (on deltas)
# -----------------------------


def merge_wa(Ws_delta: torch.Tensor) -> torch.Tensor:
    """Plain mean. Ws_delta: (T, Do, Di)"""
    return Ws_delta.mean(dim=0)


def merge_trimmed_mean(
    Ws_delta: torch.Tensor, trim_frac: float = 0.1, **kwargs
) -> torch.Tensor:
    """Entrywise trimmed mean across tasks."""
    T = Ws_delta.shape[0]
    k = int(math.floor(trim_frac * T))
    if k <= 0:
        return Ws_delta.mean(dim=0)
    vals, _ = Ws_delta.sort(dim=0)
    return vals[k : T - k].mean(dim=0)


def merge_huber_mean(
    Ws_delta: torch.Tensor,
    delta: float = 1.5,
    iters: int = 10,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """Entrywise Huber M-estimator via iterative reweighting (approx)."""
    mu = Ws_delta.mean(dim=0)
    for _ in range(iters):
        r = Ws_delta - mu
        absr = r.abs()
        w = torch.ones_like(r)
        mask = absr > delta
        w[mask] = (delta / absr[mask]).clamp_min(eps)
        mu_new = (w * Ws_delta).sum(dim=0) / w.sum(dim=0).clamp_min(eps)
        if torch.linalg.norm((mu_new - mu).flatten()) <= 1e-6 * torch.linalg.norm(
            mu.flatten()
        ).clamp_min(1e-12):
            mu = mu_new
            break
        mu = mu_new
    return mu


def merge_geometric_median(
    Ws_delta: torch.Tensor, max_iter: int = 50, **kwargs
) -> torch.Tensor:
    """Geometric median across tasks under Frobenius norm (robust to outlier tasks)."""
    return _geometric_median_fro(Ws_delta, max_iter=max_iter)


def merge_median_of_means(
    Ws_delta: torch.Tensor, n_groups: int = 8, **kwargs
) -> torch.Tensor:
    """Median-of-means across tasks (robust + scalable)."""
    return _median_of_means(Ws_delta, n_groups=n_groups)


# -----------------------------
# SVD shrinkage baselines (on merged delta)
# -----------------------------


def merge_svd_shrink(
    Ws_delta: torch.Tensor,
    shrink: Literal["soft", "hard", "ridge"] = "soft",
    lam: Optional[float] = None,
    lam_grid: int = 16,
    cv: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Average first, then shrink singular values of the mean delta.
    """
    Wm = Ws_delta.mean(dim=0)
    U, S, Vh = _svd(Wm)

    if lam is None:
        if not cv:
            lam = float(S.median().item()) * 0.5
        else:
            smax = float(S[0].item())
            lo = max(1e-8, 0.01 * smax)
            hi = 0.5 * smax
            candidate = torch.logspace(
                math.log10(lo), math.log10(hi), steps=lam_grid, device=Ws_delta.device
            )
            method = {"soft": "svd_soft", "hard": "svd_hard", "ridge": "svd_ridge"}[
                shrink
            ]
            lam = _cv_lambda_two_split(Ws_delta, candidate, method)

    lam = float(lam)
    if shrink == "soft":
        S2 = torch.clamp(S - lam, min=0.0)
    elif shrink == "hard":
        S2 = S * (S > lam)
    elif shrink == "ridge":
        S2 = (S * S * S) / (S * S + lam)
    else:
        raise ValueError(shrink)

    return _reconstruct(U, S2, Vh)


def merge_localization_aware(
    Ws_delta: torch.Tensor,
    base: Literal["soft", "ridge"] = "soft",
    lam: Optional[float] = None,
    ipr_mult: float = 8.0,
    extra_shrink: float = 2.0,
    energy_for_rank: float = 0.99,
    **kwargs,
) -> torch.Tensor:
    """
    Same as merge_svd_shrink, but shrink more on components whose singular vectors are localized.
    """
    Wm = Ws_delta.mean(dim=0)
    U, S, Vh = _svd(Wm)
    Do, Di = Wm.shape

    ipr_u = _ipr(U.T, dim=1)
    ipr_v = _ipr(Vh, dim=1)
    thr_u = ipr_mult / float(Do)
    thr_v = ipr_mult / float(Di)
    flagged = (ipr_u > thr_u) | (ipr_v > thr_v)

    if lam is None:
        r_mid = _pick_rank_energy(S, energy=energy_for_rank)
        lam = float(S[min(r_mid - 1, S.numel() - 1)].item()) * 0.5
    lam = float(lam)

    lam_k = torch.where(
        flagged,
        torch.tensor(lam * extra_shrink, device=Ws_delta.device, dtype=S.dtype),
        torch.tensor(lam, device=Ws_delta.device, dtype=S.dtype),
    )

    if base == "soft":
        S2 = torch.clamp(S - lam_k, min=0.0)
    elif base == "ridge":
        S2 = (S * S * S) / (S * S + lam_k)
    else:
        raise ValueError(base)

    return _reconstruct(U, S2, Vh)


# -----------------------------
# Robust PCA (optional; on merged delta)
# -----------------------------


def robust_pca_ialm(
    M: torch.Tensor,
    lam: Optional[float] = None,
    mu: Optional[float] = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robust PCA via inexact augmented Lagrange multiplier (IALM):
        min ||L||_* + lam ||S||_1  s.t.  M = L + S
    """
    device = M.device
    Do, Di = M.shape
    if lam is None:
        lam = 1.0 / math.sqrt(max(Do, Di))

    norm_M = torch.linalg.norm(M, ord="fro")
    if mu is None:
        mu = (Do * Di) / (4.0 * (M.abs().sum().item() + 1e-12))
        mu = float(mu)

    L = torch.zeros_like(M)
    S = torch.zeros_like(M)
    Y = M / max(norm_M.item(), 1e-12)

    for it in range(max_iter):
        U, sig, Vh = _svd(M - S + (1.0 / mu) * Y)
        sig_thresh = torch.clamp(sig - (1.0 / mu), min=0.0)
        L = _reconstruct(U, sig_thresh, Vh)

        S = _soft_thresh(M - L + (1.0 / mu) * Y, lam / mu)

        Z = M - L - S
        Y = Y + mu * Z

        err = torch.linalg.norm(Z, ord="fro") / max(
            norm_M, torch.tensor(1e-12, device=device)
        )
        if verbose and (it % 10 == 0 or it == max_iter - 1):
            print(
                f"[RPCA] iter={it:4d} err={float(err.item()):.3e} rank(L)={(sig_thresh>0).sum().item()}"
            )

        if float(err.item()) < tol:
            break

    return L, S


def merge_robust_pca(
    Ws_delta: torch.Tensor,
    aggregator: Literal["mean", "trimmed", "huber", "geom_median", "mom"] = "mean",
    trim_frac: float = 0.1,
    huber_delta: float = 1.5,
    mom_groups: int = 8,
    rpca_lam: Optional[float] = None,
    rpca_max_iter: int = 100,
    rpca_tol: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    Robust pipeline:
      1) robust aggregate across tasks to get M
      2) RPCA(M) = L + S
      3) return L (a "cleaned" low-rank-ish delta)
    """
    if aggregator == "mean":
        M = Ws_delta.mean(dim=0)
    elif aggregator == "trimmed":
        M = merge_trimmed_mean(Ws_delta, trim_frac=trim_frac)
    elif aggregator == "huber":
        M = merge_huber_mean(Ws_delta, delta=huber_delta)
    elif aggregator == "geom_median":
        M = merge_geometric_median(Ws_delta)
    elif aggregator == "mom":
        M = merge_median_of_means(Ws_delta, n_groups=mom_groups)
    else:
        raise ValueError(aggregator)

    L, _S = robust_pca_ialm(
        M, lam=rpca_lam, max_iter=rpca_max_iter, tol=rpca_tol, verbose=False
    )
    return L


# =============================================================================
# Power-law / anisotropic shrinkage (NEW)
# =============================================================================


def _fit_power_law_to_spectrum(
    S: torch.Tensor,
    k_min: int,
    k_max: int,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Fit log S_k ≈ log c - alpha * log k on k in [k_min, k_max] (1-indexed k).
    Returns (alpha, c).

    This is a *heuristic* fit used as a smooth target spectrum.
    """
    r = S.numel()
    k_min = max(1, min(k_min, r))
    k_max = max(k_min + 1, min(k_max, r))

    k = torch.arange(1, r + 1, device=S.device, dtype=S.dtype)
    kk = k[k_min - 1 : k_max]  # (m,)
    yy = torch.log(torch.clamp(S[k_min - 1 : k_max], min=eps))
    xx = torch.log(kk)

    # simple least squares for yy = b + m*xx, where m = -alpha
    x_mean = xx.mean()
    y_mean = yy.mean()
    denom = torch.sum((xx - x_mean) ** 2) + eps
    m = torch.sum((xx - x_mean) * (yy - y_mean)) / denom
    b = y_mean - m * x_mean

    alpha = float((-m).item())
    c = float(torch.exp(b).item())
    return alpha, c


def merge_powerlaw_spectral(
    Ws_delta: torch.Tensor,
    gamma: float = 0.5,
    fit_range: Tuple[float, float] = (0.1, 0.7),
    tail_only: bool = True,
    head_frac_keep: float = 0.05,
    use_log_space: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Power-law spectral smoothing WITHOUT W_pre.

    Intuition:
      - Form mean delta Wm.
      - Compute its SVD: Wm = U diag(S) V^T.
      - Fit a power-law curve to the *mid-spectrum* of S (avoids head spikes + noisy tail).
      - Shrink S toward that smooth curve:
            S_new = (1-gamma) * S + gamma * S_target
        optionally only for tail (k beyond head_frac_keep).

    This targets the "few spiky singular values" pathology while keeping directions fixed.
    """
    Wm = Ws_delta.mean(dim=0)
    U, S, Vh = _svd(Wm)
    r = S.numel()

    # choose a mid-spectrum window in ranks [k_min, k_max]
    lo_frac, hi_frac = fit_range
    k_min = max(1, int(math.floor(lo_frac * r)))
    k_max = max(k_min + 2, int(math.floor(hi_frac * r)))

    alpha, c = _fit_power_law_to_spectrum(S, k_min=k_min, k_max=k_max)
    k = torch.arange(1, r + 1, device=S.device, dtype=S.dtype)
    S_target = (c * (k ** (-alpha))).to(dtype=S.dtype)

    if use_log_space:
        # Shrink in log-space can be more stable across decades:
        # log S_new = (1-gamma) log S + gamma log S_target
        eps = torch.finfo(S.dtype).tiny
        logS = torch.log(torch.clamp(S, min=eps))
        logT = torch.log(torch.clamp(S_target, min=eps))
        S_new = torch.exp((1.0 - gamma) * logS + gamma * logT)
    else:
        S_new = (1.0 - gamma) * S + gamma * S_target

    if tail_only:
        k0 = max(1, int(math.floor(head_frac_keep * r)))
        # keep the very top singular values unchanged (avoid over-smoothing strong shared directions)
        S_new[:k0] = S[:k0]

    return _reconstruct(U, S_new, Vh)


def merge_pretrained_anchored_anisotropic(
    Ws_delta: torch.Tensor,  # (T, Do, Di) = W_ft - W_pre
    W_pre: torch.Tensor,  # (Do, Di)
    r: Optional[int] = None,  # pretrained basis rank to use (None => full)
    tau: Optional[float] = None,  # shrinkage scale in pretrained basis
    tau_mult: float = 1.0,
    mode: Literal["var_shrink", "powerlaw_spectrum"] = "var_shrink",
    # powerlaw_spectrum settings (used when mode=="powerlaw_spectrum")
    gamma: float = 0.5,
    fit_range: Tuple[float, float] = (0.1, 0.7),
    tail_only: bool = True,
    head_frac_keep: float = 0.05,
    **kwargs,
) -> torch.Tensor:
    """
    Pretrained-anchored shrinkage WITH W_pre.

    You asked for an "anisotropic" idea: use pretrained singular vectors as the shared basis.

    Step 1) Compute SVD of W_pre:
            W_pre = U0 diag(S0) V0^T
           Use U0,V0 as *fixed* directions.

    Step 2) Represent each delta in that basis (like a coordinate system):
            A_t = U0^T  Δ_t  V0      (r x r)

    Step 3) Merge + shrink inside that coordinate system, then reconstruct:
            A_bar = mean_t A_t
            Δ_hat = U0  A_tilde  V0^T

    Two shrinkage options inside the pretrained basis:

      (A) mode="var_shrink"  (anisotropic / task-stability shrinkage)
          - Compute per-coordinate variance across tasks: Var[A_t(i,j)]
          - Shrink A_bar(i,j) more when it varies a lot across tasks:
                A_tilde = A_bar * (tau / (Var + tau))
            So coordinates that are unstable across tasks are suppressed.

      (B) mode="powerlaw_spectrum" (power-law smoothing in pretrained basis)
          - Take SVD of A_bar = P diag(s) Q^T (this is r x r)
          - Fit a power-law to s and shrink s toward it (same idea as merge_powerlaw_spectral)
          - This preserves pretrained directions at the outer level, while smoothing within the
            pretrained-coordinate merged update.
    """
    assert Ws_delta.ndim == 3
    T, Do, Di = Ws_delta.shape
    assert W_pre.shape == (Do, Di)

    U0, S0, Vh0 = _svd(W_pre)
    V0 = Vh0.transpose(-2, -1)

    r0 = min(Do, Di)
    if r is None:
        r = r0
    r = int(max(1, min(r, r0)))

    U = U0[:, :r]  # (Do, r)
    V = V0[:, :r]  # (Di, r)

    # Project deltas into pretrained basis: A_t = U^T Δ_t V  -> (T, r, r)
    Ut = U.transpose(0, 1)  # (r, Do)
    A = Ut.unsqueeze(0) @ Ws_delta  # (T, r, Di)
    A = A @ V.unsqueeze(0)  # (T, r, r)

    A_bar = A.mean(dim=0)  # (r, r)

    if mode == "var_shrink":
        # Anisotropic shrinkage based on task-to-task variability in pretrained coordinates.
        A_var = A.var(dim=0, unbiased=False)  # (r, r)

        # tau controls how aggressively we shrink unstable coordinates.
        # If tau is None, use a robust-ish scale from the median variance.
        if tau is None:
            tau = float(A_var.median().item()) * tau_mult + 1e-12
        tau = float(tau)

        # Weight in [0,1]: smaller when var is large => stronger shrink
        w = tau / (A_var + tau)
        A_tilde = A_bar * w

    elif mode == "powerlaw_spectrum":
        # Power-law smoothing applied to the singular values of A_bar itself.
        P, s, Qh = _svd(A_bar)
        rr = s.numel()

        lo_frac, hi_frac = fit_range
        k_min = max(1, int(math.floor(lo_frac * rr)))
        k_max = max(k_min + 2, int(math.floor(hi_frac * rr)))
        alpha, c = _fit_power_law_to_spectrum(s, k_min=k_min, k_max=k_max)
        k = torch.arange(1, rr + 1, device=s.device, dtype=s.dtype)
        s_target = (c * (k ** (-alpha))).to(dtype=s.dtype)

        # log-space smoothing is generally more stable for heavy tails
        eps = torch.finfo(s.dtype).tiny
        logs = torch.log(torch.clamp(s, min=eps))
        logt = torch.log(torch.clamp(s_target, min=eps))
        s_new = torch.exp((1.0 - gamma) * logs + gamma * logt)

        if tail_only:
            k0 = max(1, int(math.floor(head_frac_keep * rr)))
            s_new[:k0] = s[:k0]

        A_tilde = _reconstruct(P, s_new, Qh)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Reconstruct merged delta in original coordinates
    dW_aligned = U @ A_tilde @ V.transpose(-2, -1)  # (Do, Di)

    # Optional: you could also handle the "residual" delta outside span(U,V) here.
    # In practice, people often either:
    #   (i) ignore it (assume most useful update lives in pretrained span), or
    #   (ii) aggressively shrink it with merge_svd_shrink / merge_localization_aware.
    return dW_aligned


# -----------------------------
# Convenience dispatcher
# -----------------------------


def merge_weights(
    Ws_delta: torch.Tensor,
    method: str = "wa",
    W_pre: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Ws_delta: (T, Do, Di) == W_ft - W_pre
    Returns: merged DELTA (Do, Di)

    method options:
      - "wa"
      - "trimmed"
      - "huber"
      - "geom_median"
      - "mom"
      - "svd_soft", "svd_hard", "svd_ridge"
      - "loc_soft", "loc_ridge"
      - "rpca"
      - "powerlaw"                       (no W_pre needed)
      - "pre_var_shrink"                 (needs W_pre)
      - "pre_powerlaw"                   (needs W_pre)
    """
    if method == "wa":
        return merge_wa(Ws_delta)
    if method == "trimmed":
        return merge_trimmed_mean(Ws_delta, **kwargs)
    if method == "huber":
        return merge_huber_mean(Ws_delta, **kwargs)
    if method == "geom_median":
        return merge_geometric_median(Ws_delta, **kwargs)
    if method == "mom":
        return merge_median_of_means(Ws_delta, **kwargs)

    if method in ("svd_soft", "svd_hard", "svd_ridge"):
        shrink = method.split("_", 1)[1]
        return merge_svd_shrink(Ws_delta, shrink=shrink, **kwargs)

    if method in ("loc_soft", "loc_ridge"):
        base = method.split("_", 1)[1]
        return merge_localization_aware(Ws_delta, base=base, **kwargs)

    if method == "rpca":
        return merge_robust_pca(Ws_delta, **kwargs)

    if method == "powerlaw":
        return merge_powerlaw_spectral(Ws_delta, **kwargs)

    if method in ("pre_var_shrink", "pre_powerlaw"):
        if W_pre is None:
            raise ValueError(f"method={method} requires W_pre")
        mode = "var_shrink" if method == "pre_var_shrink" else "powerlaw_spectrum"
        return merge_pretrained_anchored_anisotropic(
            Ws_delta, W_pre=W_pre, mode=mode, **kwargs
        )

    raise ValueError(f"Unknown method: {method}")


# *******


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
        "glowhite": merge_glowhite,
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
        "maor_agnostic": lambda x, **kw: merge_maor(x, magnitude_agnostic=True, **kw),
        "maor_aware": lambda x, **kw: merge_maor(x, magnitude_agnostic=False, **kw),
        "karcher": merge_karcher,
        "project_mean_col": lambda x, **kw: merge_project_mean(x, mode="col", **kw),
        "project_mean_row": lambda x, **kw: merge_project_mean(x, mode="row", **kw),
        "project_mean_both": lambda x, **kw: merge_project_mean(x, mode="both", **kw),
        "project_mean_col_lr": lambda x, **kw: merge_project_mean(
            x, mode="col", low_rank=True, **kw
        ),
        "project_mean_row_lr": lambda x, **kw: merge_project_mean(
            x, mode="row", low_rank=True, **kw
        ),
        "project_mean_both_lr": lambda x, **kw: merge_project_mean(
            x, mode="both", low_rank=True, **kw
        ),
        "preserve_var": merge_preserve_var,
        # New methods:
        "trimmed_mean": merge_trimmed_mean,
        "huber_mean": merge_huber_mean,
        "geometric_median": merge_geometric_median,
        "median_of_means": merge_median_of_means,
        "svd_shrink": merge_svd_shrink,
        "localization_aware": merge_localization_aware,
        "robust_pca": merge_robust_pca,
        "powerlaw_spectral": merge_powerlaw_spectral,
        "pretrained_anchored_anisotropic": merge_pretrained_anchored_anisotropic,
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
