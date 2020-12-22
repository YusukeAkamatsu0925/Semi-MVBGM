# Semi-MVBGM
The code of Semi-Supervised Multi-View Bayesian Generative Model (Semi-MVBGM)

"Semi-MVBGM.ipynb" produces the results of TABLE I and Fig.8 in the paper.

# cite
Please cite our paper if you want to use this code in your own work:
```
@article{akamatsu2020brain,
  title={Brain Decoding of Viewed Image Categories via Semi-Supervised Multi-View Bayesian Generative Model},
  author={Akamatsu, Yusuke and Harakawa, Ryosuke and Ogawa, Takahiro and Haseyama, Miki},
  journal={IEEE Transactions on Signal Processing},
  volume={68},
  pages={5769--5781},
  year={2020},
  publisher={IEEE}
}
```

Dataset is provided from Ref. [5] (https://github.com/KamitaniLab/GenericObjectDecoding).

[5] T. Horikawa and Y. Kamitani, “Generic decoding of seen and imagined objects using hierarchical visual features,” Nature Commun., vol. 8, no. 15037, pp. 1–15, 2017.

# code
- Semi-MVBGM.ipynb : code of Semi-MVBGM to produce TABLE I and Fig.8 of the manuscript (Jupyter Notebook)
- Semi-MVBGM.py : code of Semi-MVBGM to produce TABLE I and Fig.8 of the manuscript (Python3.7)
  *Note that the above codes do not produce exactly the same results as TABLE I and Fig.8 since prior distributions are randomly initialized by a multivariate normal distribution.
# data
- candidate_name.txt : category names of candidates (10,000 candidates, top 50 categories are 50 test categories)
- test_images : images of the original 50 test categories
- fMRI_activiy.mat : fMRI activity of five subjects        
    - subxx_train : training fMRI activity of subjectxx
    - subxx_test_ave : test fMRI activity of subjectxx
- visual&category.mat : visual and category features
    - VGG19_train : training visual features (subject01 ~ subject03) 
    - VGG19_train_subxx : training visual features (subjectxx (04 and 05))
    - VGG19_candidate : visual features of candidate categories
    - word2vec_train : training category features (subject01 ~ subject03)
    - word2vec_train_subxx : training category features (subjectxx(04 and 05))
    - word2vec_candidate : category features of candidate categories
    - candidate_names : ImageNet ID of candidate categories
- additional_visual&category.mat
    - VGG19_ILSVRC : additional visual features for Semi-MVBGM
    - word2vec_ILSVRC : additional category features for Semi-MVBGM
    - VGG19_ILSVRC_without : additional visual features for Semi-MVBGM-w/o
    - word2vec_ILSVRC_without : additional category features for Semi-MVBGM-w/o
