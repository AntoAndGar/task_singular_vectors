#### Regularized RegMean
import itertools
from typing import List, Dict
from copy import deepcopy
import torch


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
