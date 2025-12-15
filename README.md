# Task Singular Vectors: Reducing Task Interference in Model Merging

<p align="center">
  <a href="https://arxiv.org/pdf/2505.11427">
    <img src="https://img.shields.io/badge/2025-2025?style=flat&label=CVPR&labelColor=blue&color=grey&link=https%3A%2F%2Farxiv.org%2Fabs%2F2412.00081" alt="CVPR 2025"/>
  </a>
</p>

This is the source code to reproduce the experiments for ["Task Singular Vectors: Reducing Task Interference in Model Merging"](https://arxiv.org/abs/2412.00081) by Antonio Andrea Gargiulo, Donato Crisostomi, Maria Sofia Bucarelli, Simone Scardapane, Fabrizio Silvestri, and Emanuele Rodolà.

Our paper studies task vectors at the layer level, focusing on task layer matrices and their singular value decomposition. We refer to the resulting singular vectors as **Task Singular Vectors** (**TSV**). Recognizing that layer task matrices are often low-rank, we propose:
1) **TSV-Compress** (**TSV-C**), a compression scheme reducing TV to 10\% of their original size while retaining 99\% of accuracy. 
2) **TSV-Merge** (**TSV-M**), a novel approach that combines compression with interference reduction to improve model merging performance.

https://github.com/user-attachments/assets/768342e1-ae1e-4f66-bfa0-979e59810202

## Dependencies

To run the code, please install all its dependencies:
```sh
conda env create
conda activate tsv
```

## Checkpoints
We provide the checkpoints in [this link](https://drive.google.com/drive/folders/1UEM1Thcz1c7dc1nji1i5uTN53Kf6G3-e?usp=sharing). The checkpoints and masks are the previous versions of the ones in [this repository](https://github.com/nik-dim/tall_masks), downloaded from there at the beginning of our research. 

## Datasets
Most datasets being used should be downloaded automatically with torchvision or huggingface. For the datasets requiring manual preparation (like Cars, DTD, EuroSAT, SUN397), please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1). Depending on the torchvision version, some issues might arise when downloading specific datasets like [here](https://github.com/basveeling/pcam/issues/4) or [here](https://github.com/pytorch/vision/issues/5662). In this case, using a different torchvision version might solve the issue. 

## Finetuning
The script `finetune.py` can be used to reproduce the training protocol.
```sh 
# Finetune on 2 GPUs
python finetune.py --model=ViT-B-32 --world-size=2 
```

## Task Singular Vectors Evaluation

### Model merging evaluation

Evaluation is performed with Hydra, please modify `model_location` and `data_location` in `config/config.yaml` before evaluation. 

##### Evaluate with baseline model merging methods:
```bash
# Evaluate with Task Arithmetic
python main.py model=ViT-B-32 method="sum" 

# Evaluate with weight averaging
python main.py model=ViT-B-32 method="average"
```

##### Evaluate model merging with TSV + baseline model merging methods:
```bash
# Evaluate with TSV-Merge Orthogonalization
python main.py model=ViT-B-32 method="TSVM"

# Evaluate with TSV-Merge Eigendecomposition
python main.py model=ViT-B-32 method="TSVM_2"

# Evaluate with Tall mask + Task Arithmetic (load tall masks from storage)
python main.py model=ViT-B-32 method="tall_mask" method.load_mask=True

# Evaluate with Tall mask + Task Arithmetic (construct tall masks from scratch)
python main.py model=ViT-B-32 method="tall_mask"
```

##### Evaluate for compression methods:
``` bash
# Evaluate with TSV-Compress
python main.py model=ViT-B-32 method="TSVC"

# Evaluate with Consensus Task Arithmetic (after constructing TALL masks)
python main.py model=ViT-B-32 method="consensus" method.prun_thre_k=2
```

Note that you can set different numbers of tasks by setting `num_tasks`. Then, the first `num_tasks` will be selected from the list defined in `src/utils/variables_and_paths.py`. Alternatively, you can directly specify the tasks as a list of strings (e.g. `DATASETS=["MNIST","Cars"]`). The results of the papers can be retrieved by setting `num_tasks` to 8, 14 and 20 for the corresponding experiments.

### Single-task evaluation
You can evaluate the performance of the fine-tuned weights on every single task by running
```sh 
# Evaluate pre-trained models.
python eval_single_task.py --model=ViT-B-32 --finetuning-mode=none

# Evaluate non-linearly fine-tuned models.
python eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard
```

The results are saved in the `results/` folder. 

## Reference
If you find this code useful, please cite the following paper:
```bibtex
@INPROCEEDINGS{11092448,
  author={Gargiulo, Antonio Andrea and Crisostomi, Donato and Bucarelli, Maria Sofia and Scardapane, Simone and Silvestri, Fabrizio and Rodolà, Emanuele},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Task Singular Vectors: Reducing Task Interference in Model Merging}, 
  year={2025},
  volume={},
  number={},
  pages={18695-18705},
  keywords={Training;Analytical models;Accuracy;Merging;Buildings;Interference;Vectors;Matrix decomposition;Through-silicon vias;Tuning;model merging;parameter-efficient fine-tuning (peft);task vectors;singular value decomposition (svd);model compression;multi-task learning;deep learning;neural networks;computer vision},
  doi={10.1109/CVPR52734.2025.01742}
}
```
Code adapted from:
- [Task Arithmetic](https://github.com/mlfoundations/task_vectors)
- [Consensus Merging](https://github.com/nik-dim/tall_masks)

