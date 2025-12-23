#### Regularized RegMean
import itertools
from typing import List, Dict
from copy import deepcopy
import torch

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
    mis = batched_kron(pus, pvs)  # (B, Do*Di, Do*Di)
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


# ===================================================
#                  Merge Methods (per layer)
# ===================================================


def merge_avg(a_tilde):
    return a_tilde.mean(dim=0)


def merge_tsv(tensors):
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
    s_hat = s.reshape(-1)
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = compute_procrustes(vt_hat.T).T  # (Rp, Do)
    tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)
    return tau_l


def merge_lsopt(a_tilde, verbose=False):
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
    u, s, vt = torch.linalg.svd(a_tilde, full_matrices=False)
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
    sol = torch.linalg.lstsq(mis_stack, a_tilde_flat)
    print("Done.")
    a_hat_vec = sol.solution
    a_hat = a_hat_vec.reshape(Do, Di)
    return a_hat


def compute_loss(w: torch.Tensor, mats: List[torch.Tensor], alpha: float = 0.1):
    """Computes the loss for the Regularized RegMean solution."""
    loss_recons = torch.tensor(0.0, device=w.device)
    loss_reg = torch.tensor(0.0, device=w.device)
    for mat in mats:
        loss_recons += torch.linalg.norm(mat - w, ord="fro")
    if alpha > 0.0:
        for i, j in itertools.combinations(range(len(mats)), 2):
            if i == j:
                pass
            z_t = w.T @ mats[i]
            z_tp = w.T @ mats[j]
            loss_reg_ij = (
                alpha
                * (z_t.T @ z_tp)
                / (
                    torch.linalg.norm(z_t, ord="fro")
                    * torch.linalg.norm(z_tp, ord="fro")
                    + 1e-9
                )
            )
            loss_reg += torch.linalg.norm(loss_reg_ij, ord="fro")
    return {
        "loss_recons": loss_recons,
        "loss_reg": loss_reg,
    }


def find_rrmean_solution(
    mats: List[torch.Tensor],
    alpha: float = 1.0,
    max_iters: int = 10_000,
    lr: float = 1e-3,
    atol: float = 1e-6,
    patience: int = 20,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):

    # # DEBUG:
    # # just return the mean of mats
    # return torch.mean(torch.stack(mats).to(device), dim=0)
    # # END DEBUG
    M, N = mats[0].shape
    if torch.stack(mats).min() == torch.stack(mats).max() == 0:
        return torch.zeros(M, N, device=device)

    # TODO: init from mean of mats
    # w = torch.nn.Parameter(torch.randn(M, N, device=device))
    w = torch.nn.Parameter(torch.mean(torch.stack(mats).to(device), dim=0))
    _mats = [mat.to(device) for mat in mats]

    opt = torch.optim.AdamW([w], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=patience
    )

    loss_prev = float("inf")
    for i in range(max_iters):
        opt.zero_grad(set_to_none=True)
        loss_dict = compute_loss(w, _mats, alpha)
        loss = sum(loss_dict.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()

        if not loss.isfinite():
            raise ValueError(f"Loss is not finite at iteration {i}")

        lv = loss.item()
        sched.step(lv)

        if abs(lv - loss_prev) < atol:
            print(
                f"Iteration {i}, loss (recons): {loss_dict['loss_recons']:.2f}, loss (reg): {loss_dict['loss_reg']:.2f}, lr: {opt.param_groups[0]['lr']:.2e} *** Converged ***"
            )
            break
        loss_prev = lv

        if i % 1000 == 0:
            print(
                f"Iteration {i}, loss (recons): {loss_dict['loss_recons']:.2f}, loss (reg): {loss_dict['loss_reg']:.2f}, lr: {opt.param_groups[0]['lr']:.2e}"
            )

    return w.detach()


def compute_rrmean_task_vector(task_vectors, config, *args, **kwargs):
    """Computes the Regularized RegMean task vector.

    Args:
        task_vectors (List[Dict]): A list of task vector objects (state dicts)
        config (Object): Contains the following attributes: [DATASETS, device]
    """
    output_vector = {}
    device = config.device
    for layer_name in task_vectors[0].vector.keys():
        tensors = [tv.vector[layer_name] for tv in task_vectors]

        # If it's 2D we do RRMean
        layer_tensor_shape = task_vectors[0].vector[layer_name].shape
        if len(layer_tensor_shape) == 2 and "text_projection" not in layer_name:
            w = find_rrmean_solution(tensors, alpha=0.2, device=device)
            output_vector[layer_name] = w
        else:  # if not 2D we compute the mean
            output_vector[layer_name] = torch.mean(
                torch.stack(tensors).to(device), dim=0
            )

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
    output_vector = compute_rrmean_task_vector(task_vectors)
