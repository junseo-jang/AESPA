# 학습 및 평가에 필요한 기본 모듈
# 데이터 로드 및 모델 저장, 로드 함수
import string
import logging
import sys
import os
import json
from tqdm import tqdm
import random
import numpy as np
import torch
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


#################################
# 기본 모듈 정리
#################################
def init_logger(filename=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    return logger

def set_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

def layer_freeze(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model

#################################
# 데이터 & 전처리
#################################

# 사용하지 않음
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    # 부호 처리 사용하지 않음
    # def remove_articles(text):
    #     regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    #     return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    # return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_punc(lower(s)))

#################################
# ASAP ++ essay 데이터 모듈 정리
#################################
ASAP_SCORE_LIST = ["score", "content", "organization", "word_choice", "sentence_fluency", "conventions",
                   "prompt_adherence", "language", "narrativity", "style", 'voice']     # 1+8개,. 11개

def load_asap_data(file_path, score_list, am_flag=False, prompt_flag=False):

    if prompt_flag:
        #import pdb;pdb.set_trace()
        with open('dataset/prompt_embd.pk', 'rb') as f:
            prompt_embd = pickle.load(f)

    # prompt 확인해서 train / validation / test 데이터셋 가져옴
    essay_set = []

    with open(file_path, "r", encoding="utf8") as json_file:
        data = json.load(json_file)['data']
        for essay in tqdm(data, desc='read_data'):
            essay_info = list(essay.keys())

            # score list: amlabel이 있으면 [4:]
            #scores = essay_info[4:] if am_flag else essay_info[2:-1]
            scores = essay_info[4:-1]

            # essay dict
            essay_dict = {}
            if prompt_flag:
                essay_dict['prompt_embd'] = prompt_embd[int(essay['prompt_id'])-1]
            for key in essay_info:

                # 에세이 key-value 저장
                # 점수가 아닌 경우
                if key not in scores:
                    if am_flag:
                        essay_dict["amlabel"] = essay["amlabel"]
                        essay_dict["amsent"] = essay["amsent"]
                    essay_dict[key] = essay[key]

                # 점수인 경우
                else:
                    score = []     # 무조건 길이 9
                    for trait_score in score_list:    # style, voice 사용안하는 중
                        try:
                            score.append(int(essay[trait_score]))
                        except:
                            score.append(100)      # 점수가 없는 경우 100 넣어줌
                    essay_dict["scores"] = score

            essay_set.append(essay_dict)

            # ############ 디버깅 ############
            # if len(essay_set) > 100:
            #     break
            # ###############################

    return essay_set


#################################
# 모델 & 세팅 저장 및 로드
#################################
def save_model(opt, step, model, prompt=None):
    # optimizer, scheduler 저장하지 않도록 수정
    # fold 파라미터 값이 있으면 teacher 모델 저장

    if prompt is not None:
        output_dir = os.path.join(opt.checkpoint_dir, opt.name, prompt, str(step))
    else:
        output_dir = os.path.join(opt.checkpoint_dir, opt.name, str(step))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(output_dir, 'model_checkpoint.pth.tar'))

    logger.info("Saving model checkpoint to %s", output_dir)


def load_model(opt, model, prompt=None, fold=None):

    # teacher 모델 불러오기
    if fold is not None:
        if prompt is not None:
            save_path = os.path.join(opt.checkpoint_dir, opt.name, prompt, f"teacher_{str(fold)}", opt.checkpoint_step)
        else:
            save_path = os.path.join(opt.checkpoint_dir, opt.name, f"teacher_{str(fold)}", opt.checkpoint_step)

    # 일반 모델 불러오기
    else:
        if prompt is not None:
            save_path = os.path.join(opt.checkpoint_dir, opt.name, prompt, opt.checkpoint_step)
        else:
            save_path = os.path.join(opt.checkpoint_dir, opt.name, opt.checkpoint_step)

    saved_model = os.path.join(save_path, 'model_checkpoint.pth.tar')
    logger.info("Loading %s" % save_path)

    checkpoint = torch.load(saved_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(opt.device)

    return model


def load_training_model(opt, model, save_path):
    # checkpoint/name/prompt/low_epoch0/model.pt
    saved_model = os.path.join(save_path, 'model_checkpoint.pth.tar')
    logger.info("Loading %s" % save_path)

    checkpoint = torch.load(saved_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(opt.device)

    return model


def save_dict_to_csv(target_dict, csv_columns, outputfile):
    import csv

    try:
        with open(outputfile, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in target_dict:
                writer.writerow(data)
        print(f"Done {outputfile}")
    except IOError:
        print("I/O error")


################################
# 참고용, 평가 출력 (타 논문 코드)
################################
# if print_info:
#     self.print_info()
#
# class Evaluator():
#
#     def __init__(self, dataset, prompt_id, out_dir, dev_x, test_x, dev_y, test_y, dev_y_org, test_y_org):
#         self.dataset = dataset
#         self.prompt_id = prompt_id
#         self.out_dir = out_dir
#         self.dev_x, self.test_x = dev_x, test_x
#         self.dev_y, self.test_y = dev_y, test_y
#         self.dev_y_org, self.test_y_org = dev_y_org, test_y_org
#         self.dev_mean = self.dev_y_org.mean()
#         self.test_mean = self.test_y_org.mean()
#         self.dev_std = self.dev_y_org.std()
#         self.test_std = self.test_y_org.std()
#         self.best_dev = [-1, -1, -1, -1]
#         self.best_test = [-1, -1, -1, -1]
#         self.best_dev_epoch = -1
#         self.best_test_missed = -1
#         self.best_test_missed_epoch = -1
#         self.batch_size = 180
#         self.low, self.high = self.dataset.get_score_range(self.prompt_id)
#         self.dump_ref_scores()
#
#     def dump_ref_scores(self):
#         np.savetxt(self.out_dir + '/preds/dev_ref.txt', self.dev_y_org, fmt='%i')
#         np.savetxt(self.out_dir + '/preds/test_ref.txt', self.test_y_org, fmt='%i')
#
#     def dump_predictions(self, dev_pred, test_pred, epoch):
#         np.savetxt(self.out_dir + '/preds/dev_pred_' + str(epoch) + '.txt', dev_pred, fmt='%.8f')
#         np.savetxt(self.out_dir + '/preds/test_pred_' + str(epoch) + '.txt', test_pred, fmt='%.8f')
#
#     def calc_correl(self, dev_pred, test_pred):
#         dev_prs, _ = pearsonr(dev_pred, self.dev_y_org)
#         test_prs, _ = pearsonr(test_pred, self.test_y_org)
#         dev_spr, _ = spearmanr(dev_pred, self.dev_y_org)
#         test_spr, _ = spearmanr(test_pred, self.test_y_org)
#         dev_tau, _ = kendalltau(dev_pred, self.dev_y_org)
#         test_tau, _ = kendalltau(test_pred, self.test_y_org)
#         return dev_prs, test_prs, dev_spr, test_spr, dev_tau, test_tau
#
#     def calc_qwk(self, dev_pred, test_pred):
#         # Kappa only supports integer values
#         dev_pred_int = np.rint(dev_pred).astype('int32')
#         test_pred_int = np.rint(test_pred).astype('int32')
#         dev_qwk = qwk(self.dev_y_org, dev_pred_int, self.low, self.high)
#         test_qwk = qwk(self.test_y_org, test_pred_int, self.low, self.high)
#         dev_lwk = lwk(self.dev_y_org, dev_pred_int, self.low, self.high)
#         test_lwk = lwk(self.test_y_org, test_pred_int, self.low, self.high)
#         return dev_qwk, test_qwk, dev_lwk, test_lwk
#
#     def evaluate(self, model, epoch, print_info=False):
#         self.dev_loss, self.dev_metric = model.evaluate(self.dev_x, self.dev_y, batch_size=self.batch_size, verbose=0)
#         self.test_loss, self.test_metric = model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size,
#                                                           verbose=0)
#
#         self.dev_pred = model.predict(self.dev_x, batch_size=self.batch_size).squeeze()
#         self.test_pred = model.predict(self.test_x, batch_size=self.batch_size).squeeze()
#
#         self.dev_pred = self.dataset.convert_to_dataset_friendly_scores(self.dev_pred, self.prompt_id)
#         self.test_pred = self.dataset.convert_to_dataset_friendly_scores(self.test_pred, self.prompt_id)
#
#         self.dump_predictions(self.dev_pred, self.test_pred, epoch)
#
#         self.dev_prs, self.test_prs, self.dev_spr, self.test_spr, self.dev_tau, self.test_tau = self.calc_correl(
#             self.dev_pred, self.test_pred)
#
#         self.dev_qwk, self.test_qwk, self.dev_lwk, self.test_lwk = self.calc_qwk(self.dev_pred, self.test_pred)
#
#         if self.dev_qwk > self.best_dev[0]:
#             self.best_dev = [self.dev_qwk, self.dev_lwk, self.dev_prs, self.dev_spr, self.dev_tau]
#             self.best_test = [self.test_qwk, self.test_lwk, self.test_prs, self.test_spr, self.test_tau]
#             self.best_dev_epoch = epoch
#             model.save_weights(self.out_dir + '/best_model_weights.h5', overwrite=True)
#
#         if self.test_qwk > self.best_test_missed:
#             self.best_test_missed = self.test_qwk
#             self.best_test_missed_epoch = epoch
#
#         if print_info:
#             self.print_info()
#
#     def print_info(self):
#         logger.info('[Dev]   loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (
#             self.dev_loss, self.dev_metric, self.dev_pred.mean(), self.dev_mean, self.dev_pred.std(), self.dev_std))
#         logger.info('[Test]  loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (
#             self.test_loss, self.test_metric, self.test_pred.mean(), self.test_mean, self.test_pred.std(),
#             self.test_std))
#         logger.info(
#             '[DEV]   QWK:  %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f, %.3f)' % (
#                 self.dev_qwk, self.dev_lwk, self.dev_prs, self.dev_spr, self.dev_tau, self.best_dev_epoch,
#                 self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[4]))
#         logger.info(
#             '[TEST]  QWK:  %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f, %.3f)' % (
#                 self.test_qwk, self.test_lwk, self.test_prs, self.test_spr, self.test_tau, self.best_dev_epoch,
#                 self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[4]))
#
#         logger.info(
#             '--------------------------------------------------------------------------------------------------------------------------')
#
#     def print_final_info(self):
#         logger.info(
#             '--------------------------------------------------------------------------------------------------------------------------')
#         logger.info('Missed @ Epoch %i:' % self.best_test_missed_epoch)
#         logger.info('  [TEST] QWK: %.3f' % self.best_test_missed)
#         logger.info('Best @ Epoch %i:' % self.best_dev_epoch)
#         logger.info('  [DEV]  QWK: %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f' % (
#         self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[4]))
#         logger.info('  [TEST] QWK: %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f' % (
#         self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[4]))
