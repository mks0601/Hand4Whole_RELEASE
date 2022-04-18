
# **Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation (Hand4Whole codes)**
  
<p align="center">  
<img src="assets/qualitative_results.png">  
</p> 

<p align="middle">
<img src="assets/3DPW_1.gif" width="720" height="240"><img src="assets/3DPW_2.gif" width="720" height="240">

High-resolution video link: [here](https://youtu.be/Ym_CH8yxBso)


## Introduction  
This repo is official **[PyTorch](https://pytorch.org)** implementation of **[**Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation** (CVPRW 2022)](https://arxiv.org/abs/2011.11534)**. **This repo contains whole-body codes**
  
  
## Quick demo  
* Slightly change `torchgeometry` kernel code following [here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527).
* Download the pre-trained Hand4Whole from [here](https://drive.google.com/file/d/1fHF_llZSxbjJNL_Gsz_NbiWAfb1fSsNZ/view?usp=sharing).
* Prepare `input.png` and pre-trained snapshot at `demo` folder.
* Download [human_model_files](https://drive.google.com/drive/folders/1jOzMo9Rl0iSgbzGiYBKlxuEKwmCih1qc?usp=sharing) and it at `common/utils/human_model_files`.
* Go to any of `demo` folders and edit `bbox`.
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
* `common` contains kernel codes for Hand4Whole.  
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
|   |-- Human36M  
|   |   |-- images  
|   |   |-- annotations   
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

* Download AGORA parsed data [[data](https://drive.google.com/drive/folders/1ZaoYEON2WX9O_8gyPVsnBO2hph8v6lPS?usp=sharing)]
* Download Human3.6M parsed data and SMPL-X parameters [[data](https://drive.google.com/drive/folders/1kgVH-GugrLoc9XyvP6nRoaFpw3TmM5xK?usp=sharing)][[SMPL-X parameters from NeuralAnnot](https://drive.google.com/drive/folders/1opns6ta471PPzvVhhm9Anv5HMd5hCdoj?usp=sharing)]
* Download MPI-INF-3DHP parsed data and SMPL-X parameters [[data](https://drive.google.com/drive/folders/1oHzb4oJHPZllLgN_yjyatp1LdqdP0R61?usp=sharing)][[SMPL-X parameters from NeuralAnnot](https://drive.google.com/file/d/1lBJyu95xN4EhDyDA1GLkLqlh0SfAKU9a/view?usp=sharing)]
* Download MSCOCO SMPL-X parameters [[SMPL-X parameters from NeuralAnnot](https://drive.google.com/file/d/1Jrx7IWdjg-1HYwv0ztLNv0oy3Y_MOkVy/view?usp=sharing)]
* Download 3DPW parsed data [[data](https://drive.google.com/drive/folders/1fWrx0jnWzcudU6FN6QCZWefaOFSAadgR?usp=sharing)]
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


## Running Hand4Whole
* In the `main/config.py`, you can change datasets to use.  

### Train 
The training consists of three stages.

#### 1st: pre-train Hand4Whole 
In the `main` folder, run  
```bash  
python train.py --gpu 0-3 --lr 1e-4 --continue
```  
to train Hand4Whole on the GPU 0,1,2,3. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`. To train Hand4Whole from the pre-trained 2D human pose estimation network, download [this](https://drive.google.com/file/d/1zHAVs1v0Ix03ug5Ym425YE3gKr8GpeAn/view?usp=sharing) and place it at `output/model_dump`.

#### 2nd: pre-train hand-only Pose2Pose
Download pre-trained hand-only Pose2Pose from [here](https://drive.google.com/file/d/18vLbJSr0FaTpzqPYdCNHDmhXbE5yeeOJ/view?usp=sharing).
Place the hand-only Pose2Pose to `tool/snapshot_12_hand.pth.tar`.
Also, place the pre-trained Hand4Whole of the first stage to `tool/snapshot_6_all.pth.tar`.
Then, go to `tool` folder and run `python merge_hand_to_all.py`.
Place the generated `snapshot_0.pth.tar` to `output/model_dump`.

#### 3rd: combine pre-trained Hand4Whole and hand-only Pose2Pose and fine-tune it
In the `main` folder, run  
```bash  
python train.py --gpu 0-3 --lr 1e-5 --continue
```  
to train Hand4Whole on the GPU 0,1,2,3. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`. 

  
### Test  
Place trained model at the `output/model_dump/`. 
  
In the `main` folder, run  
```bash  
python test.py --gpu 0-3 --test_epoch 6
```  
to test Hand4Whole on the GPU 0,1,2,3 with60th epoch trained model. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`.  
  
## Models
* Download Hand4Whole trained on H36M+MPI-INF-3DHP+MSCOCO from [here](https://drive.google.com/file/d/1fHF_llZSxbjJNL_Gsz_NbiWAfb1fSsNZ/view?usp=sharing).
* Download Hand4Whole fine-tuned on AGORA (with gender classification) from [here](https://drive.google.com/file/d/1iEc0uPhhKTH-QVuf-qH-IpYWuNeb2jfP/view?usp=sharing).

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
