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


## Datasets


## Training


## Evaluation

### Detection

![Detection](docs/vis_det.jpg)

### Segmentation

![Detection](docs/vis_seg.jpg)

## Export

## Re-parameterization

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