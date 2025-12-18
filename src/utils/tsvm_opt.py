import torch
from tqdm import tqdm


def compute_tsvm_opt(task_vectors, config, *args, **kwargs):
    """Computes the ISO-C task vector. See https://arxiv.org/pdf/2502.04959 for more details.

    Args:
        task_vectors (List[Dict]): A list of task vector objects (state dicts)
        config (Object): Contains the following attributes: [DATASETS, device]
    """
    tau = {}
    device = config.device
    pbar = tqdm(
        task_vectors[0].vector.keys(),
        desc="Computing SVD",
        total=len(task_vectors[0].vector.keys()),
        leave=False,
    )
    for layer_name in pbar:
        tensors = torch.stack([tv.vector[layer_name] for tv in task_vectors]).to(device)

        # If it's 2D we do SVD
        layer_tensor_shape = task_vectors[0].vector[layer_name].shape
        if len(layer_tensor_shape) == 2 and "text_projection" not in layer_name:

            tau_l_skewed = tensors.mean(dim=0)
            u, s, vt = torch.linalg.svd(tau_l_skewed, full_matrices=False)
            s_iso = torch.ones_like(s) * s.mean()
            tau_l = torch.einsum("ij,j,jk->ik", u, s_iso, vt)
            tau[layer_name] = tau_l
        else:  # if not 2D we compute the mean
            tau[layer_name] = torch.mean(tensors, dim=0)

    return tau
