# Semi-MVBGM
Semi-Supervised Multi-View Bayesian Generative Model (Semi-MVBGM)

The code for the following paper:
> [Brain decoding of viewed image categories via semi-supervised multi-view Bayesian generative model](https://ieeexplore.ieee.org/abstract/document/9214493)
IEEE Transactions on Signal Processing, vol. 68, pp. 5769–5781, 2020.

# code
<!-- We are doing maintenance of codes. If you want to use codes as soon as possible, please send mail to yusukeakamatsu0925@gmail.com.-->
- Semi-MVBGM.ipynb : code of Semi-MVBGM to produce TABLE I and Fig.8 of the manuscript (Jupyter Notebook)
- Semi-MVBGM.py : code of Semi-MVBGM to produce TABLE I and Fig.8 of the manuscript (Python3.7)
- MVBGM.ipynb : code of MVBGM to produce TABLE I and Fig.8 of the manuscript (Jupyter Notebook)
- MVBGM.py : code of MVBGM to produce TABLE I and Fig.8 of the manuscript (Python3.7)
  
  Note that the above codes do not produce exactly the same results as TABLE I and Fig.8 since prior distributions are randomly initialized by a multivariate normal distribution.
  
# requirements
- Jupyter Notebook
- numpy
- scipy
- matplotlib
- Pillow


# data
fMRI dataset is provided from Ref. [5] (https://github.com/KamitaniLab/GenericObjectDecoding)*.

*Copyright : CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

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


# cite
Please cite the following papers if you want to use this code in your work.
```
@article{akamatsu2020brain,
  title={Brain Decoding of Viewed Image Categories via Semi-Supervised Multi-View Bayesian Generative Model},
  author={Akamatsu, Yusuke and Harakawa, Ryosuke and Ogawa, Takahiro and Haseyama, Miki},
  journal={IEEE Transactions on Signal Processing},
  volume={68},
  pages={5769--5781},
  year={2020}
}
```
```
@inproceedings{akamatsu2019estimating,
  title={Estimating Viewed Image Categories from fMRI Activity via Multi-view Bayesian Generative Model},
  author={Akamatsu, Yusuke and Harakawa, Ryosuke and Ogawa, Takahiro and Haseyama, Miki},
  booktitle={IEEE Global Conference on Consumer Electronics (GCCE)},
  pages={127--128},
  year={2019}
}
```
