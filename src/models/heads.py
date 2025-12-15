import argparse
import os
from typing import Callable, List

import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets.registry import get_dataset
from src.datasets.templates import get_templates
from src.models.modeling import ClassificationHead, ImageEncoder
from src.utils.variables_and_paths import TQDM_BAR_FORMAT


def build_classification_head(
    model: nn.Module,
    dataset_name: str,
    template: List[Callable[[str], str]],
    data_location: str,
    device: torch.device,
) -> ClassificationHead:
    """
    Builds a classification head for a given model and dataset.

    Args:
        model (nn.Module): The model to use for text encoding.
        dataset_name (str): The name of the dataset to use for zero-shot classification.
        template (List[Callable[[str], str]]): A list of functions that generate text templates for each class.
        data_location (str): The location of the dataset.
        device (torch.device): The device to use for computation.

    Returns:
        A ClassificationHead object with normalized weights for zero-shot classification.
    """
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    dataset = get_dataset(dataset_name, None, location=data_location)
    model.eval()
    model.to(device)

    print("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames, bar_format=TQDM_BAR_FORMAT):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            embeddings = model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    print(f"zeroshot shape, P{zeroshot_weights.shape}")
    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    return classification_head


def get_classification_head(args: argparse.Namespace, dataset: str) -> nn.Module:
    """
    Retrieves or builds a classification head for a given model and dataset.

    If the classification head file does not exist, it builds one from scratch in the location specified by `args.save_dir`.

    Args:
        args (argparse.Namespace): The command-line arguments.
        dataset (str): The name of the dataset.

    Returns:
        nn.Module: The classification head module.

    Raises:
        FileNotFoundError: If the classification head file does not exist.

    """
    if not dataset.endswith("Val"):
        dataset += "Val"

    filename = os.path.join(args.save_dir, f"head_{dataset}.pt")
    if os.path.exists(filename):
        print(
            f"Loading classification head for {args.model} on {dataset} from {filename}"
        )
        return ClassificationHead.load(filename)
    print(
        f"Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch."
    )
    model = ImageEncoder(args.model, keep_lang=True).model
    template = get_templates(dataset)

    classification_head = build_classification_head(
        model, dataset, template, args.data_location, args.device
    )
    os.makedirs(args.save_dir, exist_ok=True)
    classification_head.save(filename)
    return classification_head
