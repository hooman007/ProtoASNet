# ProtoASNet
Official repository for the paper:

> **ProtoASNet: Dynamic Prototypes for Inherently Interpretable and Uncertainty-Aware Aortic Stenosis Classification in Echocardiography**              
> Hooman Vaseli*, Ang Nan Gu*, S. Neda Ahmadi Amiri*, Michael Y. Tsang*, Andrea Fung, Nima Kondori, Armin Saadat, Purang Abolmaesumi, Teresa S. M. Tsang </br>
> (*Equal Contribution) </br> 
> **Published in MICCAI 2023** </br> 
> [Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_36) </br> 
> [arXiv Link](https://arxiv.org/abs/2307.14433) 

--------------------------------------------------------------------------------------------------------
## Contents
- [Introduction](#Introduction)
- [Environment Setup](#Environment Setup)
- [Train and Test](#Train and Test)
- [Local Explanation](#Local Explanation)
- [Description of Files and Folders](#Description of Files and Folders)
- [Acknowledgement](#Acknowledgement)
- [Citation](#Citation)


## Introduction 

This work has the aim to detect severity of Aortic Stenosis (AS) in B-Mode echo of 
Parasternal Long and Short axes (PLAX and PSAX) views. 
Due to privacy issues, we cannot share the private dataset on which we experimented on.
We also experimentd on the [TMED-2 public dataset](https://tmed.cs.tufts.edu/tmed_v2.html), however that would be only for the image-based models.  


--------------------------------------------------------------------------------------------------------
## Environment Setup

1. Clone the repo

```bash
git clone https://github.com/hooman007/ProtoASNet.git
cd ProtoASNet
```
2. place your data in the `data` folder.

3. If using Docker, it can be setup by running `docker_setup.sh` on your server. Change the parameters according to your needs:
   1. the name of the container `--name=your_container_name`  \
   2. Find the suitable pytorch image tag from https://hub.docker.com/r/pytorch/pytorch/tags based on your server.
   For example, we used: `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime`

4. Python library dependencies can be installed using:

```bash
pip install --upgrade pip
pip install torch torchvision  # if pytorch docker is not used
pip install pandas wandb tqdm seaborn torch-summary opencv-python jupyter jupyterlab imageio array2gif moviepy scikit-image scikit-learn torchmetrics termplotlib
pip install -e .
# sanity check 
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

--------------------------------------------------------------------------------------------------------
## Train and Test

To train the model `cd` to the project folder, then use the command `python main.py` with the arguments described below:

- `--config_path="src/configs/<config-name>.yml"`: yaml file containing hyper-parameters for model, experiment, loss objectives, dataset, and augmentations. all are stored in `src/configs`
- `--run_name="<your run name>"`: the name used by wandb to show the training results.
- `--save_dir="logs/<path-to-save>"` the folder to save all the trained model checkpoints, evaluations, and visualization of learned prototypes
- `--eval_only=True` a flag that evaluates the trained model
- `--eval_data_type="valid"` or  `--eval_data_type="test"` evaluates the model using valid or test dataset respectively. only applied when `--eval_only` flag is ON. 
- `--push_only=True` a flag to project (and then save the visualization of) the trained prototypes to the nearest relevant extracted features of training dataset. (this is done during training as well, but we can do it on any model checkpoint as standalone function using this flag)
- **Note:** You can modify any of the parameters included in the `config.yml` file on the fly by adding it as a parameter to python call in bash. For hierarchical parameters, the format is `--parent.child.child=value`
Examples for model checkpoint path:

  - `python main.py --config_path="src/configs/Ours_ProtoASNet_Video.yml" --run_name="ProtoASNet_test_run" --save_dir="logs/ProtoASNet/VideoBased_testrun_00" --model.checkpoint_path="logs/ProtoASNet/VideoBased_testrun_00/last.pth"`
  This bash command runs the last checkpoint saved in `VideoBased_testrun_00` folder.

**Note: You can find the training/testing commands with finalized hyper-parameters and yaml config files for the models reported in the MICCAI 2023 paper (both our models and baselines) in the `MICCAI2023_ProtoASNet_Deploy.sh` script.** 

```bash
bash MICCAI2023_ProtoASNet_Deploy.sh
```

### outputs 

the important content saved in save_dir folder are:

- `model_best.pth`: checkpoint of the best model based on a metric of interest (e.g. mean AUC or F1 score)
- `last.pth`: checkpoint of the model saved on the last epoch
- `<epoch_num>push_f1-<meanf1>.pth`: saved checkpoint after every prototype projection.

- `img/epoch-<epoch_num>_pushed`: folder containing:
  
  - visualization of projected prototypes

  - `prototypes_info.pickle`: stored dictionary containing:
    
    - `prototypes_filenames`: filenames of the source images
    - `prototypes_src_imgs`: source images in numpy
    - `prototypes_gts`: label of the source images
    - `prototypes_preds`: prediction of the source images (how model sees the source images)
    - `prototypes_occurrence_maps`: occurence map correpsonding to each prototype (where the model looks at for each prototype)
    - `prototypes_similarity_to_src_ROIs`: similarity score of the prototype vector before projection to the ROI it is projected to,

------------------------------------------------------------------------------
## Local Explanation
You can run the local exlanation to explain a given image locally by showing how similar it is to the learnt prototypes
and how the model made its decision to classify the image as such.

To explain all the data in validation or test set, run the command bellow:

```bash
python explain.py --explain_locally=True --eval_data_type='val' --config_path="src/configs/<your config>.yml" --run_name="LocalExplain_<your name>"  --wandb_mode="disabled" --save_dir="logs/<your run name>" --model.checkpoint_path="logs/<your run name>/model_best.pth"
```
 
outputs are stored in folder `/path/to/saved/checkpoint/epoch_<##>/val` with this format:

- `local/filename/test_clip_AS-<AsLabel>.MP4`: showing the input echo video 
- `local/filename/AS-<AsLable>_<sim_score>_<prototype#>.png`


--------------------------------------------------------------------------------------------------------
## Description of files and folders

### logs
Once you run the system, it will contain the saved models, logs, and evaluation results (visualization of explanations, etc)

### pretrained_models
When training is done for the first time, pretrained backbone models are saved here.

### src
- `agents/`: folder containing agent classes for each of the architectures. contains the main framework for the training process
- `configs/`: folder containing the yaml files containing hyper-parameters for model, experiment, loss objectives, dataset, and augmentations.
- `data/`: folder for dataset and dataloader classes
- `loss/`: folder for loss functions
- `models/`: folders for model architectures
- `utils/`: folder for some utility scripts and local explanation 

--------------------------------------------------------------------------------------------------------
## Acknowledgement
Some code is borrowed from [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet), 
and we developed XprotoNet architecture based on their [paper](https://arxiv.org/abs/2103.10663), 
--------------------------------------------------------------------------------------------------------

## Citation
If you find this work useful in your research, please cite:
```
@InProceedings{10.1007/978-3-031-43987-2_36,
author="Vaseli, Hooman and Gu, Ang Nan and Ahmadi Amiri, S. Neda and Tsang, Michael Y. and Fung, Andrea and Kondori, Nima and Saadat, Armin and Abolmaesumi, Purang and Tsang, Teresa S. M.",
editor="Greenspan, Hayit and Madabhushi, Anant and Mousavi, Parvin and Salcudean, Septimiu
and Duncan, James and Syeda-Mahmood, Tanveer and Taylor, Russell",
title="ProtoASNet: Dynamic Prototypes for Inherently Interpretable and Uncertainty-Aware Aortic Stenosis Classification in Echocardiography",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="368--378",
isbn="978-3-031-43987-2"
}
```