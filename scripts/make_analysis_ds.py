# Load model for Cars
import torch
import certifi
import os
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader_v2, maybe_dictionarize
from src.models.modeling import ImageClassifier, ImageEncoder
from src.models.heads import build_classification_head
from src.datasets.templates import get_templates

os.environ["SSL_CERT_FILE"] = certifi.where()

dataset_name = "CIFAR10"
batch_size = 32
template = get_templates(dataset_name)
data_location = "datasets"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "ViT-B-16"

# Build model
image_encoder = ImageEncoder(model_name, keep_lang=True)

classification_head = build_classification_head(
    image_encoder.model, dataset_name, template, data_location, device
)
model = ImageClassifier(image_encoder, classification_head)

dataset_name = "CIFAR10"
batch_size = 32
dataset = get_dataset(
    dataset_name,
    model.val_preprocess,
    location=data_location,
    batch_size=batch_size,
)
dataloader = get_dataloader_v2(
    dataset, is_train=False, image_encoder=None, batch_size=batch_size, device=device
)


stats = {}  # name -> [sum, sumsq, count, shape]


def hook(name):
    def h(_m, _i, o):
        """Hook to collect statistics of all activations.

        Args:
            name (str): Name of the activation.
            _m (torch.nn.Module): The module.
            _i (int): The index of the batch.
            o (torch.Tensor): The output of the module.
        """
        if torch.is_tensor(o):
            x = o.detach().float()
            if name not in stats:
                stats[name] = [0.0, 0.0, 0, tuple(x.shape)]
            s, ss, n, sh = stats[name]
            stats[name][0] = s + x.sum().item()
            stats[name][1] = ss + (x * x).sum().item()
            stats[name][2] = n + x.numel()

    return h


with torch.no_grad():
    for i, batch in enumerate(dataloader):
        x = maybe_dictionarize(batch)["images"]
        model(x)
        print("Batch", i)
        if i > 10:
            break
