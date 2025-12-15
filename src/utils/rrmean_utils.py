#### Regularized RegMean
import itertools
from typing import List, Dict
from copy import deepcopy
import torch


def compute_loss(w: torch.Tensor, mats: List[torch.Tensor], alpha: float = 0.1):
    """Computes the loss for the Regularized RegMean solution."""
    loss = torch.tensor(0.0)
    for mat in mats:
        loss += torch.linalg.norm(mat - w, ord="fro")
    for i, j in itertools.combinations(range(len(mats)), 2):
        loss += alpha * (w.T @ mats[i] @ w.T @ mats[j] @ w)
    return loss


def find_rrmean_solution(
    mats: List[torch.Tensor],
    alpha: float,
    max_iters: int = 10_000,
    lr: float = 1e-3,
    atol: float = 1e-6,
    patience: int = 20,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    M, N = mats[0].shape
    w = torch.nn.Parameter(torch.randn(M, N, device=device))
    _mats = [mat.to(device) for mat in mats]

    opt = torch.optim.AdamW([w], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=patience
    )

    loss_prev = float("inf")
    for i in range(max_iters):
        opt.zero_grad(set_to_none=True)
        loss = compute_loss(w, _mats, alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()

        lv = loss.item()
        sched.step(lv)

        if abs(lv - loss_prev) < atol:
            print(
                f">>> Converged @ iteration {i}, loss: {lv:.2f}, lr: {opt.param_groups[0]['lr']:.2e}"
            )
            break
        loss_prev = lv

        if i % 1000 == 0:
            print(f"Iteration {i}, loss: {lv:.2f}, lr: {opt.param_groups[0]['lr']:.2e}")

    return w.detach().cpu()


def compute_rrmean_task_vector(task_vectors, *args, **kwargs):
    """Computes the Regularized RegMean task vector.

    Args:
        task_vectors (List[Dict]): A list of task vector objects (state dicts)
        config (Object): Contains the following attributes: [DATASETS, device]
    """
    output_vector = {}
    for layer_name, layer_tensor in task_vectors[0].items():
        # If it's a linear layer we do the regular mean
        if len(layer_tensor.shape) == 2 and "text_projection" not in layer_name:
            mats = [tv[layer_name] for tv in task_vectors]
            w = find_rrmean_solution(mats, alpha=0.1)
            output_vector[layer_name] = w
    return output_vector


if __name__ == "__main__":
    task_vectors = [
        {
            "linear1": torch.randn(10, 10),
            "linear2": torch.randn(10, 10),
        }
    ]
    output_vector = compute_rrmean_task_vector(task_vectors)
