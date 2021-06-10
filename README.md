# Temporally Sorting Images from Real-World Events
Repository with code and (pointers to the) model weights for the paper: 

**Temporally Sorting Images from Real-World Events** 
R. Padilha, F.A. Andaló, B. Lavi, L.A.M. Pereira, A. Rocha 
Pattern Recognition Letters (volume 147, pages 212-219, 2021)
DOI: [https://doi.org/10.1016/j.patrec.2021.04.027](https://doi.org/10.1016/j.patrec.2021.04.027)


## Abstract
> As smartphones become ubiquitous in modern life, every major event — from musical concerts to terrorist attempts — is massively captured by multiple devices and instantly uploaded to the Internet. Once shared through social media, the chronological order between available media pieces cannot be reliably recovered, hindering the understanding and reconstruction of that event. In this work, we propose data-driven methods for temporally sorting images originated from heterogeneous sources and captured from distinct angles, viewpoints, and moments. We model the chronological sorting task as an ensemble of binary classifiers whose answers are combined hierarchically to estimate an image’s temporal position within the duration of the event. We evaluate our method on images from the Notre-Dame Catedral fire and the Grenfell Tower fire events and discuss research challenges for analyzing data from real-world forensic events. Finally, we employ visualization techniques to understand what our models have learned, offering additional insights to the problem.


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

