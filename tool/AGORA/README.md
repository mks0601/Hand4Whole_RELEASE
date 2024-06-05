# AGORA dataset parsing process

* All files can be downloaded from [here](https://agora.is.tue.mpg.de/download.php).
* If you want only one of SMPL and SMPLX data, you can ignore the other one.
* This repo requires only SMPLX data, while [Pose2Pose branch](https://github.com/mks0601/Hand4Whole_RELEASE/tree/Pose2Pose) requires only SMPL data.

## Make annotation files
* This code will dump SMPL/SMPLX parameters in camera-centered coordinate system and camera parameters at `smpl_params_cam/smplx_params_cam` and `cam_params` folders, respectively, in $DATASET_PATH. Also, it will generate `AGORA_train_SMPL.json`, `AGORA_validation_SMPL.json`, `AGORA_train_SMPLX.json`, and `AGORA_validation_SMPLX.json` in $DATASET_PATH.
* For the SMPL data, 1) download and unzip  `smpl_gt.zip`, `train_SMPL.zip`, and `validation_SMPL.zip` and 2) run  `python agora2coco_smpl.py --dataset_path $DATASET_PATH`.
* For the SMPLX data, 1) download and unzip  `smplx_gt.zip`, `train_SMPLX.zip` and `validation_SMPLX.zip` and 2) run  `python agora2coco_smplx.py --dataset_path $DATASET_PATH`.
* $DATASET_PATH denotes AGORA dataset path. 

## Preparing 1280x720 image files
* This code will prepare 1280x720 image files.
* Download and unzip 1280x720 image files.
* Then, make `1280x720` folder in AGORA dataset path.
* For the $i$th zip file of training set, make `train_$i$` folder and move all image files to that folder. For example, make `train_0` folder at AGORA dataset path and move all image files from `train_images_1280x720_0.zip` to that folder.
* For the images of validation and test sets, make `validation` and `test` folders and move all images files to corresponding folders.

## Preparing 3840x2160 image files
* This code will prepare 3840x2160 image files.
* Do the same process of 1280x720 image files
* As the image resolution is too high, you need to crop and resize humans to prevent the dataloader from being stuck.
* To this end, run `python affine_transom.py --dataset_path $DATASET_PATH --out_height 512 --out_width 384`. $DATASET_PATH denotes AGORA dataset path. 

## Download `AGORA_test_bbox.json`
* Download human detection results on test set from [here](https://drive.google.com/file/d/1dGIMsX00xUIwlFTa1gtU9bTxbfTpMt9T/view?usp=share_link).
* The human detection results are from YOLO v5.

## Final directory
```
${DATASET_PATH}
|-- AGORA_train_SMPL.json
|-- AGORA_validation_SMPL.json
|-- AGORA_train_SMPLX.json
|-- AGORA_validation_SMPLX.json
|-- AGORA_test_bbox.json
|-- smpl_params_cam
|   |-- train_0
|   |-- train_1
|   |-- train_2
|   |-- train_3
|   |-- train_4
|   |-- train_5
|   |-- train_6
|   |-- train_7
|   |-- train_8
|   |-- train_9
|   |-- validation
|-- smplx_params_cam
|   |-- train_0
|   |-- train_1
|   |-- train_2
|   |-- train_3
|   |-- train_4
|   |-- train_5
|   |-- train_6
|   |-- train_7
|   |-- train_8
|   |-- train_9
|   |-- validation
|-- cam_params
|   |-- train_0
|   |-- train_1
|   |-- train_2
|   |-- train_3
|   |-- train_4
|   |-- train_5
|   |-- train_6
|   |-- train_7
|   |-- train_8
|   |-- train_9
|   |-- validation
|-- 1280x720
|   |-- train_0
|   |-- train_1
|   |-- train_2
|   |-- train_3
|   |-- train_4
|   |-- train_5
|   |-- train_6
|   |-- train_7
|   |-- train_8
|   |-- train_9
|   |-- validation
|   |-- test
|-- 3840x2160
|   |-- train_0
|   |-- train_0_crop
|   |-- train_1
|   |-- train_1_crop
|   |-- train_2
|   |-- train_2_crop
|   |-- train_3
|   |-- train_3_crop
|   |-- train_4
|   |-- train_4_crop
|   |-- train_5
|   |-- train_5_crop
|   |-- train_6
|   |-- train_6_crop
|   |-- train_7
|   |-- train_7_crop
|   |-- train_8
|   |-- train_8_crop
|   |-- train_9
|   |-- train_9_crop
|   |-- validation
|   |-- validation_crop
|   |-- test
|   |-- test_crop
```

