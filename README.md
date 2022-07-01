
# **Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation (Pose2Pose codes)**
  
<p align="center">  
<img src="assets/qualitative_results.png">  
</p> 

<p align="middle">
<img src="assets/3DPW_1.gif" width="720" height="240"><img src="assets/3DPW_2.gif" width="720" height="240">
</p>

High-resolution video link: [here](https://youtu.be/Ym_CH8yxBso)


## Introduction  
This repo is official **[PyTorch](https://pytorch.org)** implementation of **[**Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation** (CVPRW 2022)](https://arxiv.org/abs/2011.11534)**. **This repo contains body-only, hand-only, and face-only codes of the Pose2Pose. The whole-body codes of the Hand4Whole are available at [here](https://github.com/mks0601/Hand4Whole_RELEASE).**
  
  
## Quick demo  
* Slightly change `torchgeometry` kernel code following [here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527).
* Download the pre-trained Pose2Pose from any of [here (body)](https://drive.google.com/file/d/1uUbit0jvSKc9o2DW6GZ9VmFfskNxPfyE/view?usp=sharing), [here (hand)](https://drive.google.com/file/d/15wYR8psO2U3ZhFYQEH1-DWc81XkWvK2Y/view?usp=sharing), and [here (face)](https://drive.google.com/file/d/1xUBye3YQHPZe5mDQfx1NWefsdWWKr_Ov/view?usp=sharing).
* Prepare `input.png` and pre-trained snapshot at any of `demo/body`, `demo/hand`, and `demo/face` folders.
* Download [human_model_files](https://drive.google.com/drive/folders/1jV5n1B_1dXkwpGz66SkH7GIXFDidkCPo?usp=sharing) and it at `common/utils/human_model_files`.
* Go to any of `demo/body`, `demo/hand`, and `demo/face` folders and edit `bbox`.
* Run `python demo.py --gpu 0`.
* If you run this code in ssh environment without display device, do follow:
```
1、Install oemesa follow https://pyrender.readthedocs.io/en/latest/install/
2、Reinstall the specific pyopengl fork: https://github.com/mmatl/pyopengl
3、Set opengl's backend to egl or osmesa via os.environ["PYOPENGL_PLATFORM"] = "egl"
```

## Directory  
### Root  
The `${ROOT}` is described as below.  
```  
${ROOT}  
|-- data  
|-- demo
|-- common  
|-- main  
|-- output  
```  
* `data` contains data loading codes and soft links to images and annotations directories.  
* `demo` contains demo codes.
* `common` contains kernel codes for Pose2Pose.  
* `main` contains high-level codes for training or testing the network.  
* `output` contains log, trained models, visualized outputs, and test result.  
  
### Data  
You need to follow directory structure of the `data` as below.  
```  
${ROOT}  
|-- data  
|   |-- AGORA
|   |   |-- data
|   |   |   |-- AGORA_train.json
|   |   |   |-- AGORA_validation.json
|   |   |   |-- AGORA_test_bbox.json
|   |   |   |-- 1280x720
|   |   |   |-- 3840x2160
|   |-- FFHQ
|   |   |-- FFHQ_FLAME_NeuralAnnot.json
|   |   |-- FFHQ.json
|   |-- FreiHAND
|   |   |-- data
|   |   |   |-- training
|   |   |   |-- evaluation
|   |   |   |-- freihand_train_coco.json
|   |   |   |-- freihand_train_data.json
|   |   |   |-- freihand_eval_coco.json
|   |   |   |-- freihand_eval_data.json
|   |-- Human36M  
|   |   |-- images  
|   |   |-- annotations   
|   |-- InterHand26M
|   |   |-- images
|   |   |-- annotations
|   |-- MPII
|   |   |-- data
|   |   |   |-- images
|   |   |   |-- annotations
|   |-- MPI_INF_3DHP
|   |   |-- data
|   |   |   |-- images_1k
|   |   |   |-- MPI-INF-3DHP_1k.json
|   |   |   |-- MPI-INF-3DHP_camera_1k.json
|   |   |   |-- MPI-INF-3DHP_joint_3d.json
|   |   |   |-- MPI-INF-3DHP_SMPL_NeuralAnnot.json
|   |-- MSCOCO  
|   |   |-- images  
|   |   |   |-- train2017  
|   |   |   |-- val2017  
|   |   |-- annotations 
|   |-- PW3D
|   |   |-- data
|   |   |   |-- 3DPW_train.json
|   |   |   |-- 3DPW_validation.json
|   |   |   |-- 3DPW_test.json
|   |   |-- imageFiles

```

* 3D body datasets: AGORA, Human3.6M, MPII, MPI-INF-3DHP, MSCOCO, and 3DPW
* 3D hand datasets: AGORA, FreiHAND, and MSCOCO
* 3D face datasets: AGORA, FFHQ, and MSCOCO
* Download AGORA parsed data [[data](https://drive.google.com/drive/folders/18CWsL28e8v50rqEbYMoU4yHHWoGJdpg_?usp=sharing)][[parsing codes](tool/AGORA)]
* Download FFHQ parsed data and FLAME parameters [[data](https://drive.google.com/file/d/1GbS5LaKgBlNuOfSXH82Lytni3yY6XI4x/view?usp=sharing)][[FLAME parameters from NeuralAnnot](https://drive.google.com/file/d/1u2Y2B5tVuZOnWy5oiNOKMkI22QUxiOKL/view?usp=sharing)]
* Download FreiHAND parsed data [[data](https://drive.google.com/drive/folders/13qR8EhHFgvJ_AjgHz-JZKQQzYJX7oLpE?usp=sharing)] [[bbox](https://drive.google.com/file/d/1LqKP3gFCDNC2epV-vsdwOZeR6_5sUZU2/view?usp=sharing)]
* Download Human3.6M parsed data and SMPL parameters [[data](https://drive.google.com/drive/folders/1r0B9I3XxIIW_jsXjYinDpL6NFcxTZart?usp=sharing)][[SMPL parameters from NeuralAnnot](https://drive.google.com/drive/folders/1ySxiuTCSdUEqbgTcx7bx02uMglPOkKjc?usp=sharing)]
* Download InterHand2.6M dataset from [here](https://mks0601.github.io/InterHand2.6M/).
* Download MPII parsed data and SMPL parameters [[data](https://drive.google.com/drive/folders/1rrL_RxhwQgwhq5BU1iIRPwl285B_KTpU?usp=sharing)][[SMPL parameters from NeuralAnnot](https://drive.google.com/file/d/1Zat9Wf41IIt9P1TVW8dPh7dckF7N-Ed-/view?usp=sharing)]
* Download MPI-INF-3DHP parsed data and SMPL parameters [[data](https://drive.google.com/drive/folders/1wQbHEXPv-WH1sNOLwdfMVB7OWsiJkq2M?usp=sharing)][[SMPL parameters from NeuralAnnot](https://drive.google.com/file/d/1A6yxW1cs2MVH3awQ-Yrgb7DNSRjGBI2p/view?usp=sharing)]
* Download MSCOCO SMPL/MANO/FLAME parameters [[SMPL parameters from NeuralAnnot](https://drive.google.com/file/d/1pFFCKuAyGY6uh112Uafw-hkbJtKCrhL_/view?usp=sharing)] [[MANO parameters from NeuralAnnot](https://drive.google.com/file/d/1OuWlMor5f0TZLVSsojz5Mh6Ut93WkcJc/view?usp=sharing)] [[FLAME parameters from NeuralAnnot](https://drive.google.com/file/d/1HlaGmwIEM6nqXXlkaNN_Cygi39oakddy/view?usp=sharing)]
* Download 3DPW parsed data [[data](https://drive.google.com/drive/folders/1HByTBsdg_A_o-d89qd55glTl44ya3dOs?usp=sharing)]
* All annotation files follow [MSCOCO format](http://cocodataset.org/#format-data). If you want to add your own dataset, you have to convert it to [MSCOCO format](http://cocodataset.org/#format-data).  
  
  
### Output  
You need to follow the directory structure of the `output` folder as below.  
```  
${ROOT}  
|-- output  
|   |-- log  
|   |-- model_dump  
|   |-- result  
|   |-- vis  
```  
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.  
* `log` folder contains training log file.  
* `model_dump` folder contains saved checkpoints for each epoch.  
* `result` folder contains final estimation files generated in the testing stage.  
* `vis` folder contains visualized results.  


## Running Pose2Pose
* In the `main/config.py`, you can change datasets to use.  

### Train 
In the `main` folder, run  
```bash  
python train.py --gpu 0-3 --parts body
```  
to train body-only Pose2Pose on the GPU 0,1,2,3. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`. You can chnage `body` to `hand` or `face` for the hand-only and face-only Pose2Pose, respectively. To train body-only Pose2Pose from the pre-trained 2D human pose estimation network, download [this](https://drive.google.com/file/d/1E_gEoY7pS5BTNcpxeM1IftkkboPKebs-/view?usp=sharing) and place it at `output/model_dump`. Then, run
```bash  
python train.py --gpu 0-3 --parts body --continue
```  

  
### Test  
Place trained model at the `output/model_dump/`. 
  
In the `main` folder, run  
```bash  
python test.py --gpu 0-3 --parts body --test_epoch 6
```  
to test body-only Pose2Pose on the GPU 0,1,2,3 with60th epoch trained model. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`.  You can chnage `body` to `hand` or `face` for the hand-only and face-only Pose2Pose, respectively.
  
## Models
* Download body-only Pose2Pose trained on H36M+MSCOCO+MPII from [here](https://drive.google.com/file/d/1uUbit0jvSKc9o2DW6GZ9VmFfskNxPfyE/view?usp=sharing).
* Download body-only Pose2Pose fine-tuned on AGORA from [here](https://drive.google.com/file/d/1G9w_jBjh7XhKQ9i7PBOuvXixW6R071jq/view?usp=sharing).
* Download hand-only Pose2Pose trained on FreiHAND+InterHand2.6M+MSCOCO from [here](https://drive.google.com/file/d/15wYR8psO2U3ZhFYQEH1-DWc81XkWvK2Y/view?usp=sharing).
* Download face-only Pose2Pose trained on FFHQ+MSCOCO from [here](https://drive.google.com/file/d/1xUBye3YQHPZe5mDQfx1NWefsdWWKr_Ov/view?usp=sharing).

## Results

### 3D body-only and hand-only results
<p align="middle">
<img src="assets/AGORA_SMPL.PNG" width="450" height="300">
</p>


<p align="middle">
<img src="assets/3DPW.PNG" width="360" height="264">
<img src="assets/FreiHAND.PNG" width="360" height="264">
</p>

## Troubleshoots
* `RuntimeError: Subtraction, the '-' operator, with a bool tensor is not supported. If you are trying to invert a mask, use the '~' or 'logical_not()' operator instead.`: Go to [here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527)

## Reference  
```  
@InProceedings{Moon_2022_CVPRW_Hand4Whole,  
author = {Moon, Gyeongsik and Choi, Hongsuk and Lee, Kyoung Mu},  
title = {Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation},  
booktitle = {Computer Vision and Pattern Recognition Workshop (CVPRW)},  
year = {2022}  
}  
```
