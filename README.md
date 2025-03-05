# SLRTP-Sign-Production-Evaluation

This codebase is part of the 3rd Workshop on Sign Language Recognition, Translation, and Production at CVPR 2025. For more details on the workshop please visit the website: https://slrtpworkshop.github.io/

We have seen an increase in Sign Language Production (SLP) approaches over the last few years. However, the lack of standardized evaluation metrics for SLP approaches hampers meaningful comparisons across different systems. The goal of this challenge is to develop a system that can translate spoken language to sign language. We release a standardized evaluation network, establishing a consistent baseline for the SLP field and enabling future researchers to compare their work against a broader range of methods.

## Getting started

To get started, you will need to install the requirements. You can do this by running the following command:

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r ./requirements.txt
```



## Training a Back Translation model

The original model comes from Sign Language Transformers, https://github.com/neccam/slt. Please cite the original paper appropriately. 

Alternatively, please see xxx for a pretrained model. 


### Arguments

- `input_path` (str): Path to the pickel or pytorch file containing the predicted skeleton keypoints.
- `model_dir` (str): Path to the directory containing the pretrained models.
- `split` (str): PHIX dev or test split.
- `--tag` (str, optional): Name for the output file. Default is 'results'.
- `--fps` (int, optional): the input frame rate of the predictions. Default is 25.

**Warning!!!** - the expected frame rate of the predictions is 25fps. If the frame rate is different, the performance will be lower than expected!

### Example Formate

```bash
python main.py /path/to/predictions /path/to/models dev --tag evaluation_results
```

### Example Command

### Evaluation Metrics
The script evaluates the predictions using the following metrics:  
- **BLEU**: Bilingual Evaluation Understudy Score
- **CHRF**: Character n-gram F-score
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **WER**: Word Error Rate
- **DTW MJE**: Dynamic Time Warping Mean Joint Error
- **Total Distance**: Total distance between predicted and ground truth poses

### GT Back translation score
The following is the expected output for the GT back translation model:

Command: 
```bash
python main.py ./data/dev.pt ./backTranslation_PHIX ./data/dev.pt --tag ground_truth_dev --fps 25
```

Dev:
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

Command: 
```bash
python main.py ./data/test.pt ./backTranslation_PHIX ./data/test.pt --tag ground_truth_dev --fps 25
```

Test:
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

## Skeleton Formate

The skeleton representation is made of 178 keypoints for an example visualization see ./demo_plots/ 

For more information on the skeleton formate see https://github.com/walsharry/SLRTP_CVPR_2025_Challenge_Code
