# Hyper-YOLO v1.1

In this repository, we provide the implementation of Hyper-YOLO v1.1, which intergrates the advantages of [YOLOv9](https://github.com/WongKinYiu/yolov9) and [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO), achieving the state-of-the-art performance on MS COCO dataset. 


<div align="center">
    <a href="./">
        <img src="docs/performance.png" width="79%"/>
    </a>
</div>


## Performance on MS COCO

We replace the neck of [YOLOv9](https://github.com/WongKinYiu/yolov9) with the proposed HyperC2Net of [Hyper-YOLO](https://github.com/iMoonLab/Hyper-YOLO), termed Hyper-YOLOv1.1. Clearly, in each scale, the Hyper-YOLOv1.1 outperforms the YOLOv9, which demonstrates the effectiveness of our HyperC2Net in capturing high-order feature correlations. The comparison of four scale models are provided in the following table

| Model            | Test Size | $AP^{val}$ | $AP^{val}_{50}$ | Params | FLOPs |
| ---              | ---       | ---  | ---  | ---    | ---     | 
| YOLOv9-T         | 640       | 38.3 | 53.1 | 2.0M   | 7.7G    |
| YOLOv9-S         | 640       | 46.8 | 63.4 | 7.1M   | 26.4G   |
| YOLOv9-M         | 640       | 51.4 | 68.1 | 20.0M  | 76.3G   |
| YOLOv9-C         | 640       | 53.0 | 70.2 | 25.3M  | 102.1G  |
| Hyper-YOLOv1.1-T | 640       | 40.3 | 55.6 | 2.5M   | 10.8G   |
| Hyper-YOLOv1.1-S | 640       | 48.0 | 64.5 | 7.6M   | 29.9G   |
| Hyper-YOLOv1.1-M | 640       | 51.8 | 69.2 | 21.2M  | 87.4G   |
| Hyper-YOLOv1.1-C | 640       | 53.2 | 70.4 | 29.8M  | 115.5G  |



## Installation

Clone repo and create conda environment (recommended).
Then install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8.

```bash
git clone https://github.com/iMoonLab/Hyper-YOLOv1.1.git  # clone
cd Hyper-YOLOv1.1
conda create -n Hyper-YOLOv1.1 python=3.8
conda activate Hyper-YOLOv1.1
pip install -r requirements.txt  # install
```
You can also use the environment.yaml file and the conda command to install the required environment.
```bash
conda env create -f environment.yaml
```

## Datasets
Data Preparation: Download the MS COCO dataset images (training, validation, and test sets) and corresponding labels, or prepare your custom dataset as shown below. Additionally, modify the dataset path in data/coco.yaml to reflect the location of your data.
```bash
coco
--images
  --train2017
  --val2017
--labels
  --train2017
  --val2017
```

## Training
Training configurations can be modified within the argument parser of “train.py” or “train_dual.py”.
You can adjust the training hyperparameters in the `data/hyps/hyp.scratch-XXX.yaml` file. Here, `XXX` can be set to `low`, `med`, or `high`, which correspond to low, medium, and high levels of data augmentation, respectively.
```bash
python train.py  --hyp hyp.scratch-low.yaml
                    hyp.scratch-med.yaml
                    hyp.scratch-high.yaml
```
The key factors are model, data, img, epoches, batch, device and training hyperparameters.
For instance, you can employ “yolov9-s-hyper.yaml” to train the “HyperYOLOv1.1-S” object detection model, and subsequently use “convert.py” along with “gelan-s-hyper.yaml” to remove the Auxiliary Reversible Branch.
# Single GPU training
```bash
# train yolov9-s-hyper models
python train_dual.py --workers 8 --device 0 --batch 16 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-s-hyper.yaml --weights '' --name yolov9-s-hyper --hyp hyp.scratch-low.yaml --epochs 500 

# train gelan-s-hyper models
# python train.py --workers 8 --device 0 --batch 32 --data data/coco.yaml --img 640 --cfg models/detect/gelan-s-hyper.yaml --weights '' --name gelan-s-hyper --hyp hyp.scratch-low.yaml --epochs 500
```
# Multiple GPU training
```bash
# train yolov9-s-hyper models
python -m torch.distributed.run --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-s-hyper.yaml --weights '' --name yolov9-s-hyper --hyp hyp.scratch-low.yaml --epochs 500 

# train gelan-s-hyper models
# python -m torch.distributed.run --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/gelan-s-hyper.yaml --weights '' --name gelan-s-hyper --hyp hyp.scratch-low.yaml --epochs 500
```

## Evaluation
The key factors are model(weight), data, img, batch, conf, iou, half.
```bash
# evaluate converted yolov9-s-hyper models
python val.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights './yolov9-s-hyper-converted.pt' --save-json --name yolov9_s_hyper_c_640_val

# evaluate yolov9-s-hyper models
# python val_dual.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights './yolov9-s-hyper.pt' --save-json --name yolov9_s_hyper_640_val

# evaluate gelan-s-hyper models
# python val.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights './gelan-s-hyper.pt' --save-json --name gelan_s_hyper_640_val
```
### Detection
The key factors are model(weight), source, img, conf, iou.
```bash
# inference converted yolov9-s-hyper models
python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-s-hyper-converted.pt' --name yolov9_s_hyper_c_640_detect

# inference yolov9 model
# python detect_dual.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-s-hyper.pt' --name yolov9_s_hyper_640_detect

# inference gelan models
# python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './gelan-s-hyper.pt' --name gelan_s_hyper_640_detect
```

![Detection](docs/vis_det.jpg)

### Segmentation

![Detection](docs/vis_seg.jpg)

## Export
Please refer to YOLOv8 or YOLOv9.


# Citation
If you find our work useful in your research, please consider citing:

```bibtex
xxx
```

# About Hypergraph Computation
Hypergraph computation is a powerful tool to capture high-order correlations among visual features. Compared with graphs, each hyperedge in a hypergraph can connect more than two vertices, which is more flexible to model complex correlations. Now, learning with high-order correlations still remains a under-explored area in computer vision. We hope our work can inspire more research in this direction. If you are interested in hypergraph computation, please refer to our series of works on hypergraph computation in the follows:

- [Hypergraph Learning: Methods and Practices](https://ieeexplore.ieee.org/abstract/document/9264674)
- [Hypergraph Nerual Networks](https://arxiv.org/abs/1809.09401)
- [HGNN+: General Hypergraph Nerual Networks](https://ieeexplore.ieee.org/document/9795251/)
- [Hypergraph Isomorphism Computation](https://arxiv.org/pdf/2307.14394)

# Contact
Hyper-YOLO is maintained by [iMoon-Lab](http://moon-lab.tech/), Tsinghua University. If you have any questions, please feel free to contact us via email: [Yifan Feng](mailto:evanfeng97@gmail.com) and [Jiangang Huang](mailto:mywhy666@stu.xjtu.edu.cn).
