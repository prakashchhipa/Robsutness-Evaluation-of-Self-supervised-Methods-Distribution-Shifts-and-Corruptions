# Title
Can Self-Supervised Representation Learning Methods Withstand Distribution Shifts and Corruptions?

# Article
[Arxiv Version](https://arxiv.org/pdf/2308.02525.pdf)

ICCV workshop proceedings on CVF

# Poster & Presentation Slides 

**Click [here](https://github.com/prakashchhipa/Robsutness-Evaluation-of-Self-supervised-Methods-Distribution-Shifts-and-Corruptions/blob/main/contents/ICCV23_OOD-Workshop_SSL_robustness_poster.pdf) for enlarged view**
<p align="center" >
  <img src="https://github.com/prakashchhipa/Robsutness-Evaluation-of-Self-supervised-Methods-Distribution-Shifts-and-Corruptions/blob/main/contents/iccvw_poster_logo.png" height= 30%  width= 50%>
</p>

**Slides for short video presentation describing the work**
**Click [here](https://github.com/prakashchhipa/Robsutness-Evaluation-of-Self-supervised-Methods-Distribution-Shifts-and-Corruptions/blob/main/contents/ICCV_OOD_Workshop_Presentation_SSL_robustness.pdf) slides deck**
<p align="center" >
  <img src="https://github.com/prakashchhipa/Robsutness-Evaluation-of-Self-supervised-Methods-Distribution-Shifts-and-Corruptions/blob/main/contents/slides_logo.PNG" height= 70%  width= 50%>
</p>


# Venue
Accepted at [CVF International Conference on Computer Vision Workshop](https://iccv2023.thecvf.com/list.of.accepted.workshops-90.php) (ICCVW 2023), Paris, France.

Workshop Name:  [Workshop and Challenges for Out Of Distribution Generalization in Computer Vision in conjuction with ICCV'23](http://www.ood-cv.org/index.html)

# Abstract
Self-supervised learning in computer vision aims to leverage the inherent structure and relationships within data to learn meaningful representations without explicit human annotation, enabling a holistic understanding of visual scenes. 
Robustness in vision machine learning ensures reliable and consistent performance, enhancing generalization, adaptability, and resistance to noise, variations, and adversarial attacks. 
Self-supervised paradigms, namely contrastive learning, knowledge distillation, mutual information maximization, and clustering, have been considered to have shown advances in invariant learning representations.
This work investigates the robustness of learned representations of self-supervised learning approaches focusing on distribution shifts and image corruptions in computer vision. Detailed experiments have been conducted to study the robustness of self-supervised learning methods on distribution shifts and image corruptions. The empirical analysis demonstrates a clear relationship between the performance of learned representations within self-supervised paradigms and the severity of distribution shifts and corruptions. Notably, higher levels of shifts and corruptions are found to significantly diminish the robustness of the learned representations. These findings highlight the critical impact of distribution shifts and image corruptions on the performance and resilience of self-supervised learning methods, emphasizing the need for effective strategies to mitigate their adverse effects. The study strongly advocates for future research in the field of self-supervised representation learning to prioritize the key aspects of safety and robustness in order to ensure practical applicability.

# Evaluated Self-supervised Representation Learning Methods
1. [A simple framework for contrastive learning of visual representations (simCLR)](http://proceedings.mlr.press/v119/chen20j.html)
2. [Exploring Simple Siamese Representation Learning (SimSiam)](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html)
3. [Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning(BYOL)](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html) 
4. [Barlow Twins: Self-Supervised Learning via Redundancy Reduction (Barlow Twins)](http://proceedings.mlr.press/v139/zbontar21a.html)
5. [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAE)](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)
6. [Emerging Properties in Self-Supervised Vision Transformers (DINO)](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)

*Evaluation Metrics* 
 -Mean Corrpution Error
 -Class Activations

# Dataset

Robustness Evaluation on ImageNet-C (all corruptions, all difficulty levels): [link](https://zenodo.org/record/2235448#.ZA4ct3bMI2w)

# Reproducibility Instructions

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








