import os
from pprint import pprint

import hydra

import wandb
import torch
from src.models.modeling import ImageEncoder
from omegaconf import DictConfig, OmegaConf
import certifi
import os


from src.eval.aggregation import create_task_vector
from src.eval.eval_utils import perform_eval_with_merged_vector
from src.utils.variables_and_paths import ALL_DATASETS, get_zeroshot_path

os.environ["SSL_CERT_FILE"] = certifi.where()


def compute_subspace_alignment(a, b):
    # compute proj mat for a
    u, s, vt = torch.linalg.svd(a, full_matrices=False)
    k = 10
    pi = u[:, :k] @ u[:, :k].T
    return torch.linalg.norm(pi @ b, ord="fro") / torch.linalg.norm(b, ord="fro")


def avg_subspace_alignment(
    tv_dict,
    pt_dict,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    sar_list = []
    for key in tv_dict.keys():
        tv_tens = tv_dict[key]
        pt_tens = pt_dict[key]
        # if 2d and not text_projection, compute the subspace alignment
        if len(tv_tens.shape) == 2 and "text_projection" not in key:
            merged_tens = pt_tens + tv_tens
            sar_l = compute_subspace_alignment(merged_tens, tv_tens)
            sar_list.append(sar_l)
    print(f"Avg subspace alignment: {torch.mean(torch.tensor(sar_list))}")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:

    if cfg.DATASETS == "":
        cfg.DATASETS = ALL_DATASETS[: cfg.num_tasks]
    else:
        cfg.num_tasks = len(cfg.DATASETS)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in cfg.DATASETS]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    OmegaConf.set_struct(cfg, True)

    # set up experiment for WandB
    print(cfg.method.full_name)
    print()
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, True)

    # create final task vector
    task_vector_dict, eval_masks, svd_dict = create_task_vector(cfg)
    pt_ckpt_path = get_zeroshot_path(cfg.model_location, "MNIST", cfg.model)
    pt_ckpt = torch.load(pt_ckpt_path, map_location="cpu")

    # load pretrained checkpoint
    pt_model = ImageEncoder(cfg.model)
    pt_model.load_state_dict(pt_ckpt)
    pt_dict = ImageEncoder(cfg.model).state_dict()
    tv_dict = task_vector_dict.vector
    avg_subspace_alignment(tv_dict, pt_dict)
    # task_vector_dict.apply_to(pretrained_checkpoint, scaling_coef=1.0, args=cfg)
    # model=ViT-B-16 method="rrmean" DATASETS=[Cars,DTD,EuroSAT,GTSRB,MNIST,SVHN] n_eval_points=3


if __name__ == "__main__":
    my_app()


# """
# Results (alpha = 1.0)
#  'val_best': {'CarsVal:normalized_top1': 0.7528409090909091,
#               'CarsVal:top1': 0.6511056511056511,
#               'DTDVal:normalized_top1': 0.5734341252699784,
#               'DTDVal:top1': 0.47074468085106386,
#               'EuroSATVal:normalized_top1': 0.5582089552238806,
#               'EuroSATVal:top1': 0.554074074074074,
#               'GTSRBVal:normalized_top1': 0.5033809166040571,
#               'GTSRBVal:top1': 0.503003003003003,
#               'MNISTVal:normalized_top1': 0.6765829145728643,
#               'MNISTVal:top1': 0.6732,
#               'SVHNVal:normalized_top1': 0.4958660603555188,
#               'SVHNVal:top1': 0.4798,
#               'avg_normalized_top1': np.float64(0.593385646852868),
#               'avg_top1': np.float64(0.5553212348389654)}}
# # Results (alpha = 0.0)
#  'val_best': {'CarsVal:normalized_top1': 0.7670454545454546,
#               'CarsVal:top1': 0.6633906633906634,
#               'DTDVal:normalized_top1': 0.5993520518358532,
#               'DTDVal:top1': 0.4920212765957447,
#               'EuroSATVal:normalized_top1': 0.732089552238806,
#               'EuroSATVal:top1': 0.7266666666666667,
#               'GTSRBVal:normalized_top1': 0.7227648384673178,
#               'GTSRBVal:top1': 0.7222222222222222,
#               'MNISTVal:normalized_top1': 0.9585929648241206,
#               'MNISTVal:top1': 0.9538,
#               'SVHNVal:normalized_top1': 0.7294336502687061,
#               'SVHNVal:top1': 0.7058,
#               'avg_normalized_top1': np.float64(0.7515464186967096),
#               'avg_top1': np.float64(0.7106501381458828)}}
# # # Results (alpha = 0.1)
#  'val_best': {'CarsVal:normalized_top1': 0.7798295454545455,
#               'CarsVal:top1': 0.6744471744471745,
#               'DTDVal:normalized_top1': 0.5993520518358532,
#               'DTDVal:top1': 0.4920212765957447,
#               'EuroSATVal:normalized_top1': 0.7115671641791045,
#               'EuroSATVal:top1': 0.7062962962962963,
#               'GTSRBVal:normalized_top1': 0.592411720510894,
#               'GTSRBVal:top1': 0.5919669669669669,
#               'MNISTVal:normalized_top1': 0.8363819095477387,
#               'MNISTVal:top1': 0.8322,
#               'SVHNVal:normalized_top1': 0.5946672178586193,
#               'SVHNVal:top1': 0.5754,
#               'avg_normalized_top1': np.float64(0.6857016015644591),
#               'avg_top1': np.float64(0.6453886190510305)}}
# # # # Results (alpha = 0.2)
#  'val_best': {'CarsVal:normalized_top1': 0.7727272727272727,
#               'CarsVal:top1': 0.6683046683046683,
#               'DTDVal:normalized_top1': 0.5863930885529157,
#               'DTDVal:top1': 0.48138297872340424,
#               'EuroSATVal:normalized_top1': 0.6548507462686567,
#               'EuroSATVal:top1': 0.65,
#               'GTSRBVal:normalized_top1': 0.5522163786626596,
#               'GTSRBVal:top1': 0.5518018018018018,
#               'MNISTVal:normalized_top1': 0.7630150753768844,
#               'MNISTVal:top1': 0.7592,
#               'SVHNVal:normalized_top1': 0.551674245556015,
#               'SVHNVal:top1': 0.5338,
#               'avg_normalized_top1': np.float64(0.646812801190734),
#               'avg_top1': np.float64(0.6074149081383124)}}
# My TSV w/o decorrelation:
#  'val_best': {'CarsVal:normalized_top1': 0.6789772727272727,
#               'CarsVal:top1': 0.5872235872235873,
#               'DTDVal:normalized_top1': 0.6382289416846653,
#               'DTDVal:top1': 0.523936170212766,
#               'EuroSATVal:normalized_top1': 0.6048507462686566,
#               'EuroSATVal:top1': 0.6003703703703703,
#               'GTSRBVal:normalized_top1': 0.8429752066115701,
#               'GTSRBVal:top1': 0.8423423423423423,
#               'MNISTVal:normalized_top1': 0.9889447236180905,
#               'MNISTVal:top1': 0.984,
#               'SVHNVal:normalized_top1': 0.895824720959074,
#               'SVHNVal:top1': 0.8668,
#               'avg_normalized_top1': np.float64(0.7749669353115548),
#               'avg_top1': np.float64(0.7341120783581777)}}
# Average
#  'val_best': {'CarsVal:normalized_top1': 0.7855113636363635,
#               'CarsVal:top1': 0.6793611793611793,
#               'DTDVal:normalized_top1': 0.6155507559395249,
#               'DTDVal:top1': 0.5053191489361702,
#               'EuroSATVal:normalized_top1': 0.7317164179104478,
#               'EuroSATVal:top1': 0.7262962962962963,
#               'GTSRBVal:normalized_top1': 0.6739293764087152,
#               'GTSRBVal:top1': 0.6734234234234234,
#               'MNISTVal:normalized_top1': 0.9503517587939698,
#               'MNISTVal:top1': 0.9456,
#               'SVHNVal:normalized_top1': 0.7331541959487392,
#               'SVHNVal:top1': 0.7094,
#               'avg_normalized_top1': np.float64(0.7483689781062934),
#               'avg_top1': np.float64(0.7065666746695115)}}
# # My TSV w/ decorrelation:
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
# """
