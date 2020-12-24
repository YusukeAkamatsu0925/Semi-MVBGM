# Semi-MVBGM
The code of Semi-Supervised Multi-View Bayesian Generative Model (Semi-MVBGM)

# cite
Please cite our paper if you want to use this code in youwork:
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
# code
- Semi-MVBGM.ipynb : code of Semi-MVBGM to produce TABLE I and Fig.8 of the manuscript (Jupyter Notebook)
- Semi-MVBGM.py : code of Semi-MVBGM to produce TABLE I and Fig.8 of the manuscript (Python3.7)
  
  *Note that the above codes do not produce exactly the same results as TABLE I and Fig.8 since prior distributions are randomly initialized by a multivariate normal distribution.
# data
fMRI dataset is provided from Ref. [5] (https://github.com/KamitaniLab/GenericObjectDecoding).

[5] T. Horikawa and Y. Kamitani, “Generic decoding of seen and imagined objects using hierarchical visual features,” Nature Commun., vol. 8, no. 15037, pp. 1–15, 2017.

- candidate_name.txt : category names of candidates (10,000 candidates, top 50 categories are 50 test categories)
- test_images : images of the original 50 test categories
- fMRI_data : fMRI activity of five subjects
  - subjectxx.mat
    - subxx_train : fMRI activity of subjectxx for training data
    - subxx_test_ave : fMRI activity of subjectxx for test data
- visual&category.mat : visual and category features
    - VGG19_train : visual features for training data (subject01 ~ subject03) 
    - VGG19_train_subxx : visual features for training data (subjectxx (04 and 05))
    - VGG19_candidate : visual features of candidate categories
    - word2vec_train : category features for training data (subject01 ~ subject03)
    - word2vec_train_subxx : category features for training data (subjectxx(04 and 05))
    - word2vec_candidate : category features of candidate categories
    - candidate_names : ImageNet ID of candidate categories
- additional_visual&category.mat
    - VGG19_ILSVRC : additional visual features for Semi-MVBGM
    - word2vec_ILSVRC : additional category features for Semi-MVBGM
    - VGG19_ILSVRC_without : additional visual features for Semi-MVBGM-w/o
    - word2vec_ILSVRC_without : additional category features for Semi-MVBGM-w/o
