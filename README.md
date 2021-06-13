# pig-face-posture-detection
## Description
This is the implementation code of our paper named *Parallel Channel and Position Attention-guided Feature Pyramid for Pig Face Posture Detection (Under Reviewer)*, **the code will be uploaded once this paper is officially accepted**.
## Install Dependencies
```
conda create -n env_pig 
conda activate env_pig
pip install torch==1.4.0
pip install cython
pip install torchvision==0.5.0
pip install albumentations
pip install imagecorruptions
pip install pycocotools
pip install terminaltables
pip install mmcv-full
sudo pip install -v -e . 
```
The version of **mmdetection** we use is [2.7.0](https://codeload.github.com/open-mmlab/mmdetection/zip/v2.7.0), and the version of **mmcv-full** we use is [1.2.1](https://download.openmmlab.com/mmcv/dist/cu100/torch1.4.0/mmcv_full-1.2.1-cp38-cp38-manylinux1_x86_64.whl),The version of **python** we use is 3.8.5.

If you run **pip install mmcv-full** meet wrong notification, you can see [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions. In our case, we use following command to successfully install mmcv.
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu100/torch1.4.0/index.html
```
## Prepare in advance
* You should first process the data into coco format and put it in the **data** path.
* For subsequent training, you should modify the file under **config**. In our case, we have modified the following parts:
  * For **configs\_base_\models\mask_rcnn_r50_fpn.py** file
    * change **num_classes=1**
  * For **configs\_base_\datasets\coco_instance.py** file
    * change **data_root = 'data/pig/'** (line 2)
    * search **img_scale**, change it to (512, 256)
    * change **workers_per_gpu=0**
    * change **samples_per_gpu=4**
    * change all train/val/test related information, search **ann_file** and **img_prefix** to your datasets path.
  * For **\mmdet\datasets\coco.py** file
    * change **CLASSES = ('pig')** (line 32)
  * For **\mmdet\core\evaluation\class_names.py** file
    * change **coco_classes** (line 69)
* Besides, for **mask_rcnn_r50_fpn_1x.py**, we also made the following changes to this config file:
```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=1)
total_epochs = 10
work_dir = './logs_pig/mask_rcnn_r50/normal'
```
## Train
Take Mask R-CNN-R50 as example, you should cd the project root path, latter execute the following command
```
sh scripts/train_mask_rcnn_r50_fpn_1x.sh
```
You can see the logs by following command
```
tail -f logs_console/train_mask_rcnn_r50_fpn_1x.out
```
## Test
Take Mask R-CNN-R50 as example, you should cd the project root path, latter execute the following command
```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/pig/mask_rcnn_r50_fpn_1x.py logs_pig/mask_rcnn_r50/normal/latest.pth --show-dir show_test/mask_rcnn_r50/normal --eval bbox segm
```
Then at **show_test/mask_rcnn_r50/normal** you will find the predict result with bbox and segmentation.
## Postscript
* If you want to modify the related display effects of the detection box or segmentation color, such as the color of the detection box, the thickness of the detection box, etc., you can modify the **show_result** method in **/mmdet/models/detectors/base.py**. For details, please refer to this [document](https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/models/detectors/base.html?highlight=imshow_det_bboxes#). Pay attention to re-execute **pip install -v -e .** command after modification.
* When the loss value is nan. The solution to this problem can be [referred to](https://github.com/open-mmlab/mmdetection/issues/3013). Specifically, add the following line of code **optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))** to the config file.
* If you encounter the following problem **AttributeError: module 'pycocotools' has no attribute '__version__'**. You should execute the following commands in order: 
  * pip uninstall mmpycocotools
  * pip uninstall pycocotools
  * pip install -v -e .
* If you encounter the following problem **cannot import name 'Config' from 'mmcv'**. You should re-install mmcv.
