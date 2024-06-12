# ASAP++ 데이터셋을 위한 평가 지표
# 평가 점수 전처리, 후처리, 평가 결과 계산

import numpy as np
import json
import os
import logging
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

# ######################
# Score Processing
# ######################
def get_min_max_scores():
    return {
        1: {'overall': (2, 12), 'trait': (1, 6)},
        2: {'overall': (1, 6), 'trait': (1, 6)},
        3: {'overall': (0, 3), 'trait': (0, 3)},
        4: {'overall': (0, 3), 'trait': (0, 3)},
        5: {'overall': (0, 4), 'trait': (0, 4)},
        6: {'overall': (0, 4), 'trait': (0, 4)},
        7: {'overall': (0, 30), 'trait': (0, 6)},
        8: {'overall': (0, 60), 'trait': (2, 12)}}

def one_post_processed(value, max_score=60, min_score=0):
    value = round(value)
    value = max(value, min_score)
    value = min(value, max_score)
    return value

def post_processed(all_preds, max_score=3, min_score=0):
    post_list = []
    for preds in all_preds:
        pred_list = []
        # 리스트 안에 값을 하나씩 후처리 후 반환
        for p in preds:
            pred_list.append(one_post_processed(p, max_score, min_score))
        post_list.append(pred_list)
    return post_list

def rescale_to_intscore(all_probs, all_pids):
    """ 0~1사이의 예측값을 각 prompt별 점수 스케일에 맞춰서 반환 """
    min_max_scores = get_min_max_scores()
    total_scaled_value = []

    for i, probs in enumerate(all_probs):
        pid = all_pids[i]
        scaled_value = []
        # overall
        min_score, max_score = min_max_scores[pid]['overall'][0], min_max_scores[pid]['overall'][1]
        re_overall = probs[0] * (max_score-min_score) + min_score
        scaled_value.append(np.around(re_overall).astype(int))

        # trait
        min_score, max_score = min_max_scores[pid]['trait'][0], min_max_scores[pid]['trait'][1]
        for t in probs[1:]:
            re_score = t * (max_score-min_score) + min_score
            scaled_value.append(np.around(re_score).astype(int))

        total_scaled_value.append(scaled_value)

    return total_scaled_value

def rescale_for_scoring(all_scores, all_pids):
    """ gold score y에 대해서 min-max scaling 적용 """
    min_max_scores = get_min_max_scores()
    total_scaled_value = []

    # probs : [overall, trait0, trait1,...traitn]
    for i, score in enumerate(all_scores):
        pid = all_pids[i]
        scaled_value = []

        # overall
        min_val, max_val = min_max_scores[pid]['overall'][0], min_max_scores[pid]['overall'][1]
        re_overall = (score[0] - min_val) / (max_val - min_val)
        scaled_value.append(re_overall)

        # trait
        min_val, max_val = min_max_scores[pid]['trait'][0], min_max_scores[pid]['trait'][1]
        for t in score[1:]:
            re_score = (t - min_val) / (max_val - min_val)
            scaled_value.append(re_score)

        total_scaled_value.append(scaled_value)

    return total_scaled_value


# ######################
# evaluation metrics
# ######################
def compute_and_save_predictions(
    global_step,
    tokenizer,
    all_index,
    all_inputs,
    all_preds,
    all_golds,
    output_dir,
    score_list,
    all_eid=None,
    all_pid=None,
    fold=None,
):
    """ Write final predictions to the json file. """

    results = []
    result_detail = []

    for i in all_index:
        results.append({
            'index':all_index[i],
            'essay_id':all_eid[i] if all_eid is not None else "",     # TODO(jin): 무조건 eid와 pid 넣게 되어있는데 수정 필요
            'prompt_id':all_pid[i] if all_pid is not None else "",
            'input':tokenizer.decode(all_inputs[i]),  # input string, , skip_special_tokens=True
            'pred': all_preds[i].tolist(),
            'gold':all_golds[i].tolist(),
        })

        result_detail.append({
            'index': all_index[i],
            'essay_id': all_eid[i] if all_eid is not None else "",  # TODO(jin): 무조건 eid와 pid 넣게 되어있는데 수정 필요
            'prompt_id': all_pid[i] if all_pid is not None else "",
            'input': tokenizer.decode(all_inputs[i]),  # input string, , skip_special_tokens=True
            'pred': {k: p for k, p in zip(score_list, all_preds[i].tolist())},
            'gold': {k:g for k,g in zip(score_list, all_golds[i].tolist())},
        })

    if output_dir:
        logger.info(f"Writing predictions to: {output_dir}")

        if global_step is not None:
            detail = global_step
        else:
            if fold is not None:
                detail = f"d_{str(fold)}"
            else:
                detail = "eval"

        output_prediction_file = os.path.join(
            output_dir, f"essay_trait_score_predictions_{detail}.json"
        )

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(results, indent=4, ensure_ascii=False) + "\n")

        # trait별 디테일 파일
        output_prediction_detailfile = os.path.join(
            output_dir, f"trait_score_predictions_details_{detail}.json"
        )

        with open(output_prediction_detailfile, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(result_detail, indent=4, ensure_ascii=False) + "\n")

    return all_preds


def get_essay_score_table(predicts, golds):
    """ 각 클래스별 레이블별 성능을 리포트 """
    labels = list(int(l) for l in set(predicts).union(set(golds)))

    clf_tabel = classification_report(golds, predicts, labels=labels)

    return clf_tabel

def fleiss_kappa_score(score1, score2, score3):
    """
    주어진 3개의 점수 리스트들의 일치도를 계산한다.
    :param score1: 점수1 리스트
    :param score2: 점수2 리스트
    :param score3: 점수3 리스트
    :return: fleiss kappa score (float)
    """
    from statsmodels.stats import inter_rater as irr

    arr = [score1, score2, score3]
    agg = irr.aggregate_raters(arr)  # returns a tuple (data, categories)
    kappa = irr.fleiss_kappa(agg[0], method='fleiss')
    # print(f"Fleiss Kappa(3 raters): {kappa:.3f}")

    return kappa

