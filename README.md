**Title - "Can Self-Supervised Representation Learning Methods Withstand Distribution Shifts and Corruptions?"**



This repository evaluates rbosutness of self-supervised learning methods for out-of-distribution samples, algorithmically generated corruptions (blur, noise) applied to the ImageNet test-set. analysis is carried out for qualitatively and quantitatively.

*Dataset (all corruptions, all difficulty levels)* 
 -ImageNet-C, [link](https://zenodo.org/record/2235448#.ZA4ct3bMI2w)

*Self-supervised Learning Methods*

1. [A simple framework for contrastive learning of visual representations (simCLR)](http://proceedings.mlr.press/v119/chen20j.html)
2. [Exploring Simple Siamese Representation Learning (SimSiam)](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html)
3. [Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning(BYOL)](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html) 
4. [Barlow Twins: Self-Supervised Learning via Redundancy Reduction (Barlow Twins)](http://proceedings.mlr.press/v139/zbontar21a.html)
5. [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAE)](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)
6. [Emerging Properties in Self-Supervised Vision Transformers (DINO)](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)

*Evaluation Metrics* 
 -Mean Corrpution Error
 -Class Activations


**A. Prerequisites** 
•	Download the complete ImageNet-C dataset with all 19-corruptions having all 5 severity levels.
•	Download and store the ImageNet validation store.
•	Preferably place both the datasets on "ssl_robustness/data" to use default settings for command line inputs.
•	SSL pretrained models’ weights are available at mmengine (link: https://github.com/open-mmlab/mmselfsup) for chosen SSL methods (simCLR, simSiam, barlow twins, dino (RN50), SwAE, and BYOL). 
•	Installations - pytorch 1.1x with torchvision, mmengine, tqdm, pandas

**B. Quantitative Evaluation for all SSL methods with ResNet50 backbone on ImageNet-C corruption for MCE**
   *Pretrained weights for each SSL model should be stored under models’ directory in their respective directory
   '''python -m eval_mean_corruption_error --test_dataset <path for imagenet_c dataset> --output <path to save results> --device <which GPU>'''

**C. Quantitative Evaluation for DINO SSL method with ViT transformer backbone on ImageNet-C corruptions for MCE**
    *Pretrained weights are fetched from online resource
   '''python -m eval_mean_corruption_error_vit --test_dataset <path for imagenet_c dataset> --output <path to save results> --device <which GPU>'''

**D. Qualitative Evaluation for all SSL method with ResNet50 backbone on ImageNet-C and comparison with standard ImageNet images**
   *Pretrained weights for each SSL model should be stored under models’ directory in their respective directory
   '''python -m generate_gradcams --test_dataset <path for imagenet_c dataset> --imagenet_path <imagenet dataset path> --output <path to save results> --device <which GPU>'''

**E. Compute Structural Similarity for snow, elastic transform, and saturate corruptions**   
    *Pretrained weights for each SSL model should be stored under models’ directory in their respective directory
   '''python -m compute_ssim --test_dataset <path for imagenet_c dataset> --imagenet_path <imagenet dataset path> --output <path to save results> --device <which GPU>'''








