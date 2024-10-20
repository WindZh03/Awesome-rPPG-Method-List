# Awesome-rPPG-Method-List

This repository is a collection of awesome rPPG methods. I will update it regularly. If you have any suggestions or questions, please feel free to contact me. 

## Table of Contents
- [Awesome-rPPG-Method-List](#Awesome-rPPG-Method-List)
  - [Table of Contents](#table-of-contents)
  - [Awesome Papers](#Awesome-Papers)
    - [Survey](#survey)
    - [Traditional Methods](#traditional-methods)
    - [Supervised Learning](#supervised-learning)
       - [2D CNN](#2d-cnn)
       - [3D CNN](#3d-cnn)
  - [Acknowledgments](#acknowledgments)

the classification of the paper references the paper [Remote photoplethysmography for heart rate measurement: A review](https://www.sciencedirect.com/science/article/abs/pii/S1746809423010418)


## Awesome Papers

### Survey

|  Title  |   Publication  |  Code   |
|:--------|:--------:|:--------:|
|[**Remote photoplethysmography for heart rate measurement: A review**](https://www.sciencedirect.com/science/article/abs/pii/S1746809423010418)| Biomedical Signal Processing and Control 2023 | - |
|[**Camera Measurement of Physiological Vital Signs**](https://arxiv.org/pdf/2111.11547)| ACM Computing Surveys 2021 | - |
|[**Video-Based Heart Rate Measurement: Recent Advances and Future Prospects**](https://ieeexplore.ieee.org/document/8552414) | TIM 2019 | - |


### Traditional Methods

|  Title  |   Publication  |  Code   |
|:--------|:--------:|:--------:|
|[**Face2PPG: An Unsupervised Pipeline for Blood Volume Pulse Extraction From Faces**](https://ieeexplore.ieee.org/document/10227326) (**OMIT**)| IEEE JBHI 2023| - |
|[**Local Group Invariance for Heart Rate Estimation from Face Videos in the Wild**](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w27/Pilz_Local_Group_Invariance_CVPR_2018_paper.pdf) (**LGI**)| CVPR Workshop 2018 | - |
|[**Algorithmic Principles of Remote PPG**](https://ieeexplore.ieee.org/document/7565547) (**POS**)| IEEE TBME 2016 | - |
|[**A Novel Algorithm for Remote Photoplethysmography: Spatial Subspace Rotation**](https://ieeexplore.ieee.org/document/7355301) (**2SR**)| IEEE TBME 2015 | - |
|[**Improved motion robustness of remote-PPG by using the blood volume pulse signature**](https://iopscience.iop.org/article/10.1088/0967-3334/35/9/1913) (**PBV**)| Physiological Measurement 2014 | - |
|[**Robust Pulse Rate From Chrominance-Based rPPG**](https://ieeexplore.ieee.org/document/6523142) (**CHROM**)| IEEE TBME 2013 | - |
|[**Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity**](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7ad15b6fecdb9b2ad49be5bf26efafe22c9a8945) (**PCA**)| FedCSIS 2011 | - |
|[**Remote plethysmographic imaging using ambient light**](https://pdfs.semanticscholar.org/7cb4/46d61a72f76e774b696515c55c92c7aa32b6.pdf?_gl=1*1q7hzyz*_ga*NTEzMzk5OTY3LjE2ODYxMDg1MjE.*_ga_H7P4ZT52H5*MTY4NjEwODUyMC4xLjAuMTY4NjEwODUyMS41OS4wLjA)| Optics Express 2008 | |


### Supervised Learning

#### 2D CNN

|  Title  |   Publication  |  Code   |
|:--------|:--------:|:--------:|
|[**Visual heart rate estimation with convolutional neural network**](https://www.sciencedirect.com/science/article/pii/S2096579620301121) (**NAS-HR**)| Virtual Reality & Intelligent Hardware 2021| - |
|[**Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement**](https://arxiv.org/pdf/2006.03790) (**MTTS-CAN**)| NeurPIS 2020| [github](https://github.com/xliucs/MTTS-CAN) |
|[**RhythmNet: End-to-End Heart Rate Estimation From Face via Spatial-Temporal Representation**](http://refhub.elsevier.com/S1746-8094(23)01041-8/sb47) (**RhythmNet**)| IEEE TIP 2020| [github](https://github.com/AnweshCR7/RhythmNet) |
|[**Video-Based Remote Physiological Measurement via Cross-Verified Feature Disentangling**](http://refhub.elsevier.com/S1746-8094(23)01041-8/sb48) (**CVD**)| ECCV 2020 oral| [github](https://github.com/nxsEdson/CVD-Physiological-Measurement) |
|[**Robust Remote Heart Rate Estimation from Face Utilizing Spatial-temporal Attention**](https://ieeexplore.ieee.org/document/8756554) |FG 2019| - |
|[**SynRhythm: Learning a Deep Heart Rate Estimator from General to Specificn**](https://ieeexplore.ieee.org/document/8546321)(**SynRhythm**) | ICPR 2018| - |
|[**EVM-CNN: Real-Time Contactless Heart Rate Estimation From Facial Video**](https://ieeexplore.ieee.org/abstract/document/8552438) (**EVM-CNN**)| IEEE TMM 2018| - |
|[**DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Weixuan_Chen_DeepPhys_Video-Based_Physiological_ECCV_2018_paper.pdf) (**DeepPhys**)| ECCV 2018| - |
|[**Visual heart rate estimation with convolutional neural network**](https://cmp.felk.cvut.cz/~spetlrad/ecg-fitness/visual-heart-rate.pdf) (**HR-CNN**)| BMVC 2018| [github](https://github.com/radimspetlik/hr-cnn) |


#### 3D CNN

|  Title  |   Publication  |  Code   |
|:--------|:--------:|:--------:|
|[**Robust Heart Rate Estimation With Spatial–Temporal Attention Network From Facial Videos**](https://ieeexplore.ieee.org/document/9364289) | IEEE TCDS 2022| - |
|[**ETA-rPPGNet: Effective Time-Domain Attention Network for Remote Heart Rate Measurement**](https://ieeexplore.ieee.org/abstract/document/9353569) (**ETA-rPPGNet**)| IEEE TIM 2021| - |
|[**Siamese-rPPG network: remote photoplethysmography signal estimation from face videos**](https://dl.acm.org/doi/abs/10.1145/3341105.3373905) (**Siamese-rPPG**)| ACM SAC 2020| - |
|[**A General Remote Photoplethysmography Estimator with Spatiotemporal Convolutional Network**](https://ieeexplore.ieee.org/document/9320234) | IEEE FG 2020| - |
|[**AutoHR: A Strong End-to-End Baseline for Remote Heart Rate Measurement With Neural Searching**](https://ieeexplore.ieee.org/document/9133501) (**AutoHR**)| IEEE SPL 2020| - |
|[**HeartTrack: Convolutional neural network for remote video-based heart rate monitoring**](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w19/Perepelkina_HeartTrack_Convolutional_Neural_Network_for_Remote_Video-Based_Heart_Rate_Monitoring_CVPRW_2020_paper.pdf) (**HeartTrack**)| CVPR 2020| - |
|[**Remote Heart Rate Measurement from Highly Compressed Facial Videos: an End-to-end Deep Learning Solution with Video Enhancement**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Remote_Heart_Rate_Measurement_From_Highly_Compressed_Facial_Videos_An_ICCV_2019_paper.pdf) (**rPPGNet**)| ICCV 2019| [github](https://github.com/ZitongYu/STVEN_rPPGNet) |
|[**Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks**](https://arxiv.org/abs/1905.02419) (**PhysNet**)| BMCV 2019| [github](https://github.com/ZitongYu/PhysNet) |
|[**3D Convolutional Neural Networks for Remote Pulse Rate Measurement and Mapping from Facial Video**](https://www.mdpi.com/2076-3417/9/20/4364)(**rppg-3dcnn**) | Applied Science 2019| [github](https://github.com/frederic-bousefsaf/ippg-3dcnn) |


## Acknowledgments

- https://github.com/zx-pan/Awesome-rPPG
