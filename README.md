# Temporally Sorting Images from Real-World Events
Repository with code and (pointers to) model weights for the paper: 


>**Temporally Sorting Images from Real-World Events** 
>
>R. Padilha, F.A. Andaló, B. Lavi, L.A.M. Pereira, A. Rocha 
>
>Pattern Recognition Letters (volume 147, pages 212-219, 2021)
>
>DOI: [https://doi.org/10.1016/j.patrec.2021.04.027](https://doi.org/10.1016/j.patrec.2021.04.027)

Please, if you use or build upon this code, cite the publication above. 

In case you have any doubts, shoot us an email! We will be glad to help and/or answer any questions about the method or evaluation. 

---------

## Abstract
> As smartphones become ubiquitous in modern life, every major event — from musical concerts to terrorist attempts — is massively captured by multiple devices and instantly uploaded to the Internet. Once shared through social media, the chronological order between available media pieces cannot be reliably recovered, hindering the understanding and reconstruction of that event. In this work, we propose data-driven methods for temporally sorting images originated from heterogeneous sources and captured from distinct angles, viewpoints, and moments. We model the chronological sorting task as an ensemble of binary classifiers whose answers are combined hierarchically to estimate an image’s temporal position within the duration of the event. We evaluate our method on images from the Notre-Dame Catedral fire and the Grenfell Tower fire events and discuss research challenges for analyzing data from real-world forensic events. Finally, we employ visualization techniques to understand what our models have learned, offering additional insights to the problem.


![alt text](https://github.com/rafaspadilha/temporal-sorting-event/blob/main/featured.png)



For more information and recent author publications, please refer to:
- [Rafael Padilha](https://rafaspadilha.github.io)
- [Fernanda Andaló](http://fernanda.andalo.net.br)


---------

## Dependencies

The codes were implemented and tested with the following libraries/packages:

| Package / Library        | Version           | 
| ------------- |-------------| 
| Python | 2.7.15 | 
| Numpy | 1.16.5 | 
| Scikit Learn | 0.20.4 | 
| Keras | 2.3.1 | 
| Tensorflow | 1.15.0 | 
| [PrettyTable](https://pypi.org/project/prettytable/) | 2.1.0 | 



---------

## Dataset and Model Weights

The **dataset** can be found at [FigShare](https://doi.org/10.6084/m9.figshare.11787333.v2). After downloading it, place both folders ('images' and 'splits') inside this `dataset` folder. 

The **model weights** can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1bjwPE7j0KHRkJlKKUBY4Jx4gFP42-lds?usp=sharing ). After downloading them, place them in the respective `models` folders inside `non_hierarchical` and `hierarchical`. 

Use the **Project Structure** bellow to help you. 

---------

## Project Structure

```
.
├── LICENSE
├── README.md
├── dataset
│   ├── README.md
│   ├── images
│   └── splits
├── hierarchical
│   ├── dataLoaderBinary.py
│   ├── dataLoaderMultiClass.py
│   ├── models
│   │   ├── README.md
│   │   ├── cutoffClass_1
│   │   │   ├── setA
│   │   │   │   └── w.epoch-0088_val_loss-0.232605.hdf5
│   │   │   └── setB
│   │   │       └── w.epoch-0185_val_loss-0.240775.hdf5
│   │   └── cutoffClass_3
│   │       ├── setA
│   │       │   └── w.epoch-0035_val_loss-0.086825.hdf5
│   │       └── setB
│   │           └── w.epoch-0349_val_loss-0.061852.hdf5
│   ├── testing_allClassifiers_Hierarchical.py
│   ├── testing_singleBvAClassifier.py
│   ├── training.py
│   └── utils.py
└── non_hierarchical
    ├── dataLoaderBinary.py
    ├── dataLoaderMultiClass.py
    ├── models
    │   ├── README.md
    │   ├── cutoffClass_1
    │   │   ├── setA
    │   │   │   └── w.epoch-0693_val_loss-0.138357.hdf5
    │   │   └── setB
    │   │       └── w.epoch-0264_val_loss-0.163483.hdf5
    │   ├── cutoffClass_2
    │   │   ├── setA
    │   │   │   └── w.epoch-0473_val_loss-0.124818.hdf5
    │   │   └── setB
    │   │       └── w.epoch-0462_val_loss-0.138809.hdf5
    │   └── cutoffClass_3
    │       ├── setA
    │       │   └── w.epoch-0084_val_loss-0.175408.hdf5
    │       └── setB
    │           └── w.epoch-0141_val_loss-0.175573.hdf5
    ├── testing_allClassifiers_NonHierarchical.py
    ├── testing_singleBvAClassifier.py
    ├── training.py
    └── utils.py
```

The folder structure for Non-Hierarchical and Hierarchical pipelines are similar. We provide a quick description of each file within them. Running instructions can be found in each file:

### Data Loaders
Files responsible for loading and preprocessing images, building batches that are fed for training and testing scripts.
- *dataLoaderBinary.py* : load batches of images for the Before vs After scenario (binary classification) used mostly during training;
- *dataLoaderMultiClass.py* : load batches of images in the Multi-class scenario (without coverting labels to Before or After classes), used during evaluation.

### Training
- *training.py* : training script of the Before vs After models

### Testing / Evaluation
- *testing_singleBvAClassifier.py* : testing code for evaluating a single Before vs After model
- *testing_allClassifiers_*.py* : testing code for evaluating the Non-Hierarchical / Hierarchical pipeline. This is the main evaluation script. 

### Additional Files
- *utils.py* : auxiliary methods (e.g., metrics, formating, etc)


