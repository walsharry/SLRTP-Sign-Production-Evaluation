import copy
import torch
import jiwer
import numpy as np

from tqdm import tqdm
from typing import List
from inspect import getfullargspec
from sacrebleu.metrics import CHRF
from fastdtw import fastdtw as dtw

from skeleton_def import RELEASE_JOINTS
from external_metrics.sacrebleu import raw_corpus_bleu
from external_metrics.rouge import calc_score as rouge_calc_score


def wer(hypotheses: list, references: list):
    hypotheses = copy.deepcopy(hypotheses)
    references = copy.deepcopy(references)
    transforms = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            # jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    wer = jiwer.wer(
        references,
        hypotheses,
        truth_transform=transforms,
        hypothesis_transform=transforms,
    )
    return wer * 100


def rank_rouge(top_n: int, hypotheses: list, references: list):
    rouge_score = []
    n_seq = len(hypotheses)

    for i, (h, r) in enumerate(zip(hypotheses, references)):
        rouge_score.append(rouge_calc_score(hypotheses=[h], references=[r]))
    rouge_score = torch.tensor(rouge_score)
    _, top_idx = torch.topk(rouge_score, top_n)
    return top_idx


def rouge(hypotheses: list, references: list):
    rouge_score = 0
    n_seq = len(hypotheses)

    for h, r in zip(hypotheses, references):
        rouge_score += rouge_calc_score(hypotheses=[h], references=[r]) / n_seq

    return rouge_score * 100


def chrf(hypotheses: List[str], references: List[str], **sacrebleu_cfg) -> float:
    """
    Character F-score from sacrebleu
    cf. https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: character f-score (0 <= chf <= 1)
             see Breaking Change in sacrebleu v2.0
    """
    kwargs = {}
    if sacrebleu_cfg:
        valid_keys = getfullargspec(CHRF).args
        for k, v in sacrebleu_cfg.items():
            if k in valid_keys:
                kwargs[k] = v

    metric = CHRF(**kwargs)
    score = metric.corpus_score(hypotheses=hypotheses, references=[references]).score

    return score


def mse(y_true, y_pred):
    if isinstance(y_true, list) and isinstance(y_pred, list):
        # numpy join
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def token_accuracy(hypotheses: list, references: list):
    """
    Calculate the token accuracy between hypotheses and references.

    Args:
    - hypotheses (list): List of hypothesis sequences.
    - references (list): List of reference sequences.

    Returns:
    - token_accuracy (float): Token accuracy.
    """
    token_accuracy = 0
    n_seq = len(hypotheses)

    for h, r in zip(hypotheses, references):
        token_accuracy += (h == r).sum() / len(r) / n_seq

    return token_accuracy


def bleu(hypotheses: List[str], references: List[str], **sacrebleu_cfg) -> dict:
    """
    Raw corpus BLEU from sacrebleu (without tokenization)
    cf. https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/bleu.py

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return: bleu score
    """
    bleu_scores = raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores


def mpjpe(gt_poses, hypo_poses):
    """
    Calculate the Mean Per Joint Position Error (MPJPE) between ground truth and hypothesis poses.

    Args:
    - gt_poses (torch.Tensor): Ground truth poses with shape (num_samples, num_joints, 3).
    - hypo_poses (torch.Tensor): Hypothesis poses with shape (num_samples, num_joints, 3).

    Returns:
    - mean_mpjpe (float): Mean Per Joint Position Error.
    """

    def cal_mpjpe(a, b):
        # Check if the input tensors have the same shape
        assert (
            a.shape == b.shape
        ), "Ground truth and hypothesis poses must have the same shape."

        # Calculate the Euclidean distance between corresponding joints
        joint_distances = torch.norm(a - b, dim=2)

        # Calculate the mean over all joints and samples
        mean_mpjpe = torch.mean(joint_distances)
        return mean_mpjpe

    if isinstance(gt_poses, list) and isinstance(hypo_poses, list):
        gt_poses = torch.cat(gt_poses)
        hypo_poses = torch.cat(hypo_poses)

    return cal_mpjpe(gt_poses, hypo_poses).item()


def pose_dtw_mje(hyps: list = None, gt_pose: torch.Tensor = None):
    """Coverts the codebook indexs to poses and calculates the MPJPE error"""

    def euclidean_distance(x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        return torch.sqrt(torch.sum((x - y) ** 2))

    def dtw_align_data(a: list, b: list, dist_fn=euclidean_distance):
        align_a = []
        align_b = []
        for _a, _b in tqdm(zip(a, b), total=len(a), desc="Aligning A to B:"):
            #  skip blank sequences
            if _a is None or _b is None:
                continue
            if len(_a) == 0 or len(_b) == 0:
                continue
            dist, path = dtw(_a.flatten(1, -1), _b.flatten(1, -1), dist=dist_fn)
            a_path, b_path = zip(*path)
            _a = _a[list(a_path)]
            _b = _b[list(b_path)]
            assert _a.shape == _b.shape
            align_a.append(_a)
            align_b.append(_b)
        return align_a, align_b

    h_gt, r_gt = dtw_align_data(hyps, gt_pose)
    mpjper_gt = mpjpe(h_gt, r_gt)

    return mpjper_gt


def pose_distance(hyps: list = None, gt_pose: torch.Tensor = None):
    """
    cal the total distance travled by the right hand to see if it is regressed to the mean.
    """

    def calculate_distance(tensor: torch.Tensor) -> float:
        """
        Calculate the total distance a point moves over N frames.
        """
        # Ensure the tensor has the correct shape
        assert tensor.ndimension() == 2 and tensor.size(1) == 3, "Tensor must be of shape (N, 3)"

        # Calculate the differences between consecutive frames
        differences = tensor[1:] - tensor[:-1]

        # Calculate the Euclidean distances
        distances = torch.norm(differences, dim=1)

        # Sum the distances to get the total distance
        total_distance = distances.sum().item()

        return total_distance

    hyp_dis_l = torch.mean(torch.tensor([calculate_distance(h[..., RELEASE_JOINTS['LWrist'], :]) for h in hyps]))
    hyp_dis_r = torch.mean(torch.tensor([calculate_distance(h[..., RELEASE_JOINTS['RWrist'], :]) for h in hyps]))

    gt_dis_l = torch.mean(torch.tensor([calculate_distance(h[..., RELEASE_JOINTS['LWrist'], :]) for h in gt_pose]))
    gt_dis_r = torch.mean(torch.tensor([calculate_distance(h[..., RELEASE_JOINTS['RWrist'], :]) for h in gt_pose]))

    l_score = hyp_dis_l / gt_dis_l
    r_score = hyp_dis_r / gt_dis_r
    score = (l_score + r_score) / 2

    return score.item()

