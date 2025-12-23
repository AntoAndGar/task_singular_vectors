#### Regularized RegMean
import itertools
from typing import List, Dict
from copy import deepcopy
import torch
from tqdm import tqdm

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
    u = u[:, :, :Rp]
    vt = vt[:, :Rp, :]
    pus = u @ u.transpose(1, 2)  # (B, Do, Do)
    v = vt.transpose(1, 2)
    pvs = v @ vt  # (B, Di, Di)
    preds = torch.einsum("boj,jk,bki->boi", pus, a_hat, pvs)
    loss = torch.linalg.norm(preds - a_tilde, ord="fro", dim=(1, 2)).square().mean()
    return loss


# ===================================================
#                  Merge Methods (per layer)
# ===================================================


def merge_avg(a_tilde, *args, **kwargs):
    return a_tilde.mean(dim=0)


def merge_tsv(tensors, *args, **kwargs):
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


def compute_opmerge_task_vector(task_vectors, config, *args, **kwargs):
    """Computes the OpMerge task vector.

    Args:
        task_vectors (List[Dict]): A list of task vector objects (state dicts)
        config (Object): Contains the following attributes: [DATASETS, device]
    """
    output_vector = {}
    device = config.device

    merge_func = {
        "avg": merge_avg,
        "tsv": merge_tsv,
        "lsopt": merge_lsopt_lstsq,
        "lsopt_gd": merge_lsopt_gd,
        "tsvopt": merge_tsvopt,
    }  # selected by config.opm

    for layer_name in task_vectors[0].vector.keys():
        tensors = torch.stack([tv.vector[layer_name] for tv in task_vectors]).to(device)

        # If it's 2D we apply the merge function
        layer_tensor_shape = task_vectors[0].vector[layer_name].shape
        if (
            len(layer_tensor_shape) == 2
            and "text_projection" not in layer_name
            and max(layer_tensor_shape) < 10_000
        ):
            w = merge_func[config.opm](tensors)
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
