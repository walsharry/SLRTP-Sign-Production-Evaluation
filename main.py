"""
This code is for the SLRTP SLP challenge 2025

This runs the evaluation for people's predictions

Code accepts, a path to the pickel or pytorch pt file containing the predictions of the skeleton keypoint, [N x K x D].

```json
{
    "01April_2010_Thursday_heute-6704":         [N x K x D],
     "30September_2012_Sunday_tagesschau-4038": [N x K x D],
    "10March_2011_Thursday_tagesschau-5086":    [N x K x D],
    ...
    "27October_2009_Tuesday_tagesschau-6148":   [N x K x D],
}
```
Where k is 178 and D is 3.

"""

import json
import torch
import argparse

from pathlib import Path

from helpers import load_json, load_pickle
from back_translation.back_translate import back_translate, make_back_translation_model
from metrics import (
    wer,
    bleu,
    chrf,
    rouge,
    pose_dtw_mje,
    pose_distance,
)


def load_predictions(input_path: Path):
    if input_path.suffix == '.json':
        print(f"Path {input_path} is a json file.")
        pred = load_json(input_path)
    elif input_path.suffix == '.pt':
        print(f"Path {input_path} is a pytorch file.")
        pred = torch.load(input_path, weights_only=True)
    elif input_path.suffix == '.pkl' or input_path.suffix == '.pickle':
        print(f"Path {input_path} is a pickle file.")
        pred = load_pickle(input_path)
    else:
        raise ValueError(f"File type {input_path.suffix} not supported.")

    # if the ground truth is used as a test, then the keys are not needed.
    # change the formate
    k = list(pred.keys())[0]
    if isinstance(pred[k], dict):
        pred = {k: v['poses_3d'] for k, v in pred.items()}

    return pred


def load_gt(gt_path: Path = ''):
    data = torch.load(str(gt_path), weights_only=True)
    keys = list(data.keys())
    text = [data[k]['text'] for k in keys]
    pose = [data[k]['poses_3d'] for k in keys]
    return keys, text, pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLP Challenge 2024')
    parser.add_argument('input_path', type=str, help='Path to the json/video file containing the predictions of the skeleton keypoint, [N x 17 x 3].')
    parser.add_argument('gt_path', type=str, help='path to the ground trutgh data.')
    parser.add_argument('model_dir', type=str, help='Path to the pretrained back translation model.')
    parser.add_argument('--tag', type=str, help='name the output file.', default='results')
    parser.add_argument('--fps', type=int, help='input frame rate of the pose file', default=25)

    args = parser.parse_args()

    input_path = Path(args.input_path)
    gt_path = Path(args.gt_path)
    model_dir = Path(args.model_dir)
    fps = int(args.fps)

    assert input_path.exists(), f"Path {input_path} does not exist."
    assert gt_path.exists(), f"Path {gt_path} does not exist."

    # load the predictions
    pose_predictions = load_predictions(input_path)

    # load the gt text
    ids, gt_text, gt_pose = load_gt(gt_path)

    # Make sure the frame rate is 12fps!!!!!
    if fps == 25:
        pose_predictions = {k: v[::2, ...] for k, v in pose_predictions.items()}
    gt_pose = [v[::2, ...] for v in gt_pose]

    # check the gt and the predictions are the same length
    assert len(pose_predictions) == len(gt_pose), "Number of predictions and ground truth does not match."

    # make sure they are in the same order
    pose_predictions = [pose_predictions[id] for id in ids]

    # Run back translation and evaluation
    bt_model = make_back_translation_model(model_dir=model_dir)
    text_pred = back_translate(model=bt_model, poses=pose_predictions)

    # run metrics
    # back translation metrics
    bleu_score = bleu(hypotheses=text_pred, references=gt_text)
    chrf_score = chrf(hypotheses=text_pred, references=gt_text)
    rouge_score = rouge(hypotheses=text_pred, references=gt_text)
    wer_score = wer(hypotheses=text_pred, references=gt_text)

    # pose metrics
    dtwmje_score = pose_dtw_mje(hyps=pose_predictions, gt_pose=gt_pose)
    total_distance = pose_distance(hyps=pose_predictions, gt_pose=gt_pose)

    results = {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "rouge": rouge_score,
        "wer": wer_score,
        "dtw_mje": dtwmje_score,
        "total_distance": total_distance,
    }

    print(f"BLEU: {bleu_score}")
    print(f"CHRF: {chrf_score}")
    print(f"ROUGE: {rouge_score}")
    print(f"WER: {wer_score}")
    print(f"DTW MJE: {dtwmje_score}")
    print(f"Total Distance: {total_distance}")

    # save to json file
    save_path = Path('./results')
    save_path.mkdir(exist_ok=True, parents=True)
    with open(save_path / f'{args.tag}.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
