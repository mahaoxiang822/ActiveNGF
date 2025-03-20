# ActiveNGF

**Active Perception for Grasp Detection via Neural Graspness Field**<br> [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/4364fef031fdf7bfd9d1c9c56b287084-Paper-Conference.pdf)

_Haoxiang Ma, Modi Shi, Boyang Gao, Di Huang_<br>
In NeurIPS'2024

## Note
This repository is still under development and I will update it gradually in few weeks.

## Introduction
This repository is official PyTorch implementation for our NeurIPS2024 paper.
The code is based on [ESLAM](https://github.com/idiap/ESLAM)

## Installation

Get the code.
```bash
git clone https://github.com/mahaoxiang822/ActiveNGF.git
cd ActiveNGF
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Download the pretrained graspness prediction model from [Google Drive](https://drive.google.com/file/d/1OswUcXVJv_LAgyyNt_KjfOPhIE4LEyk7/view?usp=sharing) and put it under
```
ckpts/
---|graspnet.pth
```
## Run
Using the scene config file in `config/` to run the pipeline. As an example, to run ActiveNGF on for the `scene_0100`, run: 

```bash
python -W ignore run.py configs/GraspNet/scene_0100.yaml
```

## ToDo
```angular2html
update evaluation code
```


## Contact
You can contact the author through email: mahaoxiang822@buaa.edu.cn

## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{ma2024activengf,
  author       = {Haoxiang Ma and
                  Modi Shi and
                  Boyang Gao and
                  Di Huang},
  title        = {Active Perception for Grasp Detection via Neural Graspness Field},
  booktitle    = {Annual Conference on Neural Information Processing Systems},
  year         = {2024},
}
```