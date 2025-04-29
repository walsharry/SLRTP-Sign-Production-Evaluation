# SLRTP-Sign-Production-Evaluation

This codebase supports the 3rd Workshop on Sign Language Recognition, Translation, and Production (SLRTP) at CVPR 2025. For more details, visit the [workshop website](https://slrtpworkshop.github.io/).

## Overview

We have seen an increase in Sign Language Production (SLP) approaches over the last few years. However, the lack of standardized evaluation metrics for SLP approaches hampers meaningful comparisons across different systems. The goal of this challenge is to develop a system that can translate spoken language to sign language. We release a standardized evaluation network, establishing a consistent baseline for the SLP field and enabling future researchers to compare their work against a broader range of methods.


## Installation

### Requirements

```bash
# Create and activate a new conda environment (recommended)
conda create --name slrtp python=3.8
conda activate slrtp

# Install PyTorch with CUDA support
# Please change PyTorch CUDA version to match your system!
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r ./requirements.txt
```

## Back Translation Model

The evaluation framework uses a back translation model to convert generated sign poses back to text for comparison.

- **Original Source**: Sign Language Transformers ([GitHub Repository](https://github.com/neccam/slt))
- **Pretrained Models and Phix Data**: Download from [Google Drive](https://drive.google.com/file/d/1fjKHigsEWHwsMHnwwWdFYZ8dECXslTKi/view) (alternative to training your own)

Please cite the original paper appropriately when using this code.

## Running the Evaluation

### Basic Usage

```bash
python main.py <input_path> <gt_path> <model_dir> --tag <evaluation_name> --fps <frame_rate>
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_path` | Path to predicted skeleton keypoints (.pt or .pkl file) | Required |
| `gt_path` | Path to ground truth skeleton keypoints and text | Required |
| `model_dir` | Directory containing the pretrained back translation model | Required |
| `--tag` | Name for the output results file | 'results' |

⚠️ **Important**: The expected frame rate for optimal performance is 25fps. Different frame rates may result in lower performance metrics.

### Example Usage

```bash
python main.py ./demo_data/progressive_transformer_baseline_dev.pt ./demo_data/ground_truth_dev.pt ./backTranslation_PHIX_model --tag demo --fps 12
```

### Example Output

```json
{
    "bleu": {
        "bleu1": 15.946522413242281,
        "bleu2": 4.580890661810554,
        "bleu3": 0.0,
        "bleu4": 0.0
    },
    "chrf": 19.755041720753972,
    "rouge": 15.536963615448718,
    "wer": 101.83727034120736,
    "dtw_mje": 0.04589659720659256,
    "total_distance": 0.30942216515541077
}
```

## Evaluation Metrics Explained

The framework evaluates sign language production using both text-based and pose-based metrics:

### Text-Based Metrics
- **BLEU (Bilingual Evaluation Understudy)**: Measures n-gram precision between predicted and reference text
  - BLEU-1 through BLEU-4 values represent different n-gram sizes
- **CHRF (Character n-gram F-score)**: Character-level metric that balances precision and recall
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Focuses on recall of n-grams
- **WER (Word Error Rate)**: Measures the minimum edit distance between predicted and reference texts
- **Total Distance**: This metric measures the overall distance the signer’s hands have moved in 3D space. This is to judge how expressive the productions are. The prediction is normalized by the ground truth distance, therefore, a score of 1 is optimal.
-  **Average Duration**: This metric assesses the accuracy of signed duration, a key component of prosody. It averages the duration of the predicted output against the duration of the ground truth reference.

### Pose-Based Metrics
- **DTW MJE (Dynamic Time Warping Mean Joint Error)**: Measures pose accuracy while accounting for timing differences
- **Total Distance**: Total distance between predicted and ground truth poses


### SLP 2025 CVPR Challenge Scores
**Table 1: RWTH-PHOENIX-Weather-2014T Test Set**

| Teams                       | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CHRF  | ROUGE | WER ↓  | DTW-MJE ↓ | Total Distance | Average Duration |
|-----------------------------|--------|--------|--------|--------|-------|-------|--------|-----------|----------------|------------------|
| Ground Truth                | 34.40  | 22.04  | 16.09  | 12.78  | 34.59 | 35.20 | 85.77  | 0.0000    | 1.000          | 1.000      |
| Team 1 (USTC-MoE)           | 34.85  | 21.96  | 15.65  | 12.06  | 36.83 | 36.59 | 93.49  | 0.0448    | 1.631          | 1.438      |
| Team 2 (hfut-lmc)           | 16.96  | 6.56   | 3.38   | 2.05   | 25.88 | 19.77 | 147.85 | 0.0403    | 2.512          | 2.463      |
| Team 3 (Hacettepe)          | 30.44  | 17.75  | 12.42  | 9.59   | 29.70 | 30.64 | 88.88  | 0.0423    | 0.798          | 1.026      |
| Progressive Transformer     | 22.17  | 10.71  | 7.09   | 5.43   | 24.13 | 21.98 | 101.45 | 0.0418    | 0.257          | 0.999      |

*Caption: SLP Challenge Results on the ph14t test set.*

**Table 2: Hidden Test Set**

| Teams                        | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CHRF  | ROUGE | WER ↓  | DTW-MJE ↓ | Total Distance |
|------------------------------|--------|--------|--------|--------|-------|-------|--------|-----------|----------------|
| Ground Truth                 | 37.94  | 19.87  | 10.67  | 5.90   | 30.64 | 38.60 | 101.25 | 0.0000    | 1.000          |
| Team 1 (USTC-MoE)            | 31.40  | 17.09  | 9.43   | 5.86   | 31.73 | 33.75 | 109.38 | 0.0574    | 1.185          |
| Team 2 (hfut-lmc)            | 30.54  | 16.22  | 9.33   | 5.66   | 30.17 | 32.92 | 107.93 | 0.0492    | 0.971          |
| Team 3 (Hacettepe)           | 27.51  | 11.13  | 5.36   | 2.91   | 23.37 | 27.29 | 105.49 | 0.0531    | 0.761          |
| Progressive Transformer [cite]| 18.33  | 4.99   | 1.74   | 0.78   | 21.65 | 21.10 | 141.93 | 0.0467    | 0.322          |

*Caption: SLP Challenge Results on the Hidden Test Set.*

#### Publicly Available Codebases
Team 1: [USTC-MoE](https://github.com/ZechengLi19/CVPRW-SLP-2025)
Team 2: [hfut-lmc](https://github.com/NaVi-start/Sign-Base.git)
Team 3: [Hacettepe](https://github.com/sumeyyemeryem/CVPR25-SLRTPChallenge)

## Dataset

For the SLRTP 2025 CVPR challenge, we use the RWTH-PHOENIX-Weather-2014T dataset. Access the challenge dataset and back translation model at [Google Drive](https://drive.google.com/file/d/1fjKHigsEWHwsMHnwwWdFYZ8dECXslTKi/view).

**File Content:**
```bash
├── backTranslation_PHIX_model
│   ├── best.ckpt
│   ├── config.yaml
│   ├── gls.vocab
│   ├── txt.vocab
│   └── validations.txt
└── data
    ├── dev.pt
    ├── PT_baseline_dev.pt
    ├── PT_baseline_test.pt
    ├── test.pt
    └── train.pt
```
Note that the data folder contains the PHOENIX-2014 train, test, and dev split. Plus, the prediction from the progressive transformer

The PHOENIX-2014 dataset is a large-scale dataset used for research in sign language recognition and translation. It contains continuous sign language videos along with their corresponding gloss annotations and spoken language translations. The dataset is derived from weather forecast broadcasts in German Sign Language (DGS).

<div style="display: flex; justify-content: center;">
    <img src="../demo_plots/01April_2010_Thursday_tagesschau-4330.gif" alt="GIF 1" width="200">
    <img src="../demo_plots/01April_2010_Thursday_tagesschau-4331.gif" alt="GIF 2" width="200">
    <img src="../demo_plots/01April_2010_Thursday_tagesschau-4332.gif" alt="GIF 3" width="200">
</div>

The dataset is split into training, development, and test sets. The training set contains 7096 videos, the development set contains 519 videos, and the test set contains 642 videos. 

### Challenge Data Format

Inside the provided .pt files, you will find a dictionary with the following structure:

```json
{
    "01April_2010_Thursday_heute-6704":{
      "name": (string),
      "text": (string),
      "gloss": (string),
      "poses_3d" : (N x K x D),
      "speaker" : (string),
    },
     "30September_2012_Sunday_tagesschau-4038": {
      "name": (string),
      "text": (string),
      "gloss": (string),
      "poses_3d" : (N x K x D),
      "speaker" : (string),
    },
    ...
    "27October_2009_Tuesday_tagesschau-6148":   {
      "name": (string),
      "text": (string),
      "gloss": (string),
      "poses_3d" : (N x K x D),
      "speaker" : (string),
    },
}
```

Split 80:10:10 for training, validation, and test sets.

The frame rate is equal to 25 frames per second.

## Skeleton Format

For each video we extract Mediapipe holistic keypoints and use the approach from "[Improving 3D Pose Estimation For Sign Language](https://personalpages.surrey.ac.uk/r.bowden/publications/2023/IvashechkinSLTAT2023.pdf)" by Ivashechkin, Maksym and Mendez, Oscar and Bowden, Richard, to uplift the predictions to 3D. We process the skeleton and provide 178 keypoint representations. 

    - 21 keypoints for each hand
    - 128 keypoints for the face
    - 8 keypoints for the body

The face is a subset of the 468 keypoint representations from the Mediapipe face mesh. See 'make_128_face_from_478.py' to create the 128 face mesh from mediapipe's 478 (mp_face_mesh.FaceMesh)

Example visualizations can be found in the `./demo_plots/` directory.

The skeleton format follows the standard used in the SLRTP CVPR 2025 Challenge. For detailed information on the format specification, visit the [challenge Skeleton Information]([https://github.com/walsharry/SLRTP_CVPR_2025_Challenge_Code](https://github.com/walsharry/SLRTP_Skeleton_Keypoint_information)).

## Reference Performance

### Ground Truth Back-Translation Scores

#### Development Set
Command:
```bash
python main.py ./data/dev.pt ./data/dev.pt ./backTranslation_PHIX_model --tag ground_truth_dev --fps 25
```

Results:
```json
{
    "bleu": {
        "bleu1": 35.64289683751959,
        "bleu2": 23.060216602791453,
        "bleu3": 16.907030146272817,
        "bleu4": 13.378651856913775
    },
    "chrf": 34.82214188714514,
    "rouge": 36.85085849846843,
    "wer": 83.37019018133569,
    "dtw_mje": 0.0,
    "total_distance": 1.0
}
```

#### Test Set
Command:
```bash
python main.py ./data/test.pt ./data/test.pt ./backTranslation_PHIX_model --tag ground_truth_test --fps 25
```

Results:
```json
{
    "bleu": {
        "bleu1": 34.39639390982878,
        "bleu2": 22.038589437767946,
        "bleu3": 16.092729031718463,
        "bleu4": 12.777148564612425
    },
    "chrf": 34.58509769785496,
    "rouge": 35.19608041310144,
    "wer": 85.77470203767781,
    "dtw_mje": 0.0,
    "total_distance": 1.0
}
```


## Citation

When using this evaluation framework or participating in the challenge, please cite:

```
@inproceedings{walsh2025slrtp,
title={SLRTP2025 Sign Language Production Challenge: Methodology, Results, and Future Work},
author={Walsh, Harry and Fish, Ed and Sincan, Ozge Mercanoglu and Lakhal, Mohamed Ilyes and Bowden, Richard and Fox, Neil and Cormier, Kearsy and Woll, Bencie and Wu, Kepeng and Li, Zecheng and Zhao, Weichao and Wang, Haodong and Zhou, Wengang and Li, Houqiang and Tang, Shengeng and He, Jiayi and Wang, Xu and Zhang, Ruobei and Wang, Yaxiong and Cheng, Lechao and Tasyurek, Meryem and Kiziltepe, Tugce and Keles, Hacer Yalim}, 
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2025}
}
```
