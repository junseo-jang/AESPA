# cross-prompt trait-scoring 유형의 AES 모델을 학습하기 위한 코드
# am_flag: argument mining label을 임베딩으로 사용
# score_masking: 프롬프트별 점수 마스킹 사용
# step마다가 아니라 epoch마다 평가하도록 설정되어 있음

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from src.functions.metric import post_processed, rescale_for_scoring, rescale_to_intscore, compute_and_save_predictions
from src.functions.utils import save_model, ASAP_SCORE_LIST
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings(action='ignore')


def train(logger, opt, model, tokenizer, optimizer, scheduler, train_dataset, eval_dataset,
          collator, am_flag=False, prompt=None, prompt_flag=None):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=opt.train_batch_size,
                                  collate_fn=collator,
                                  )

    criterion = torch.nn.MSELoss()

    train_loss, total_loss = 0.0, 0.0
    global_step = 0
    metric_score, best_score = 0.0, 0.0

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Train batch size per GPU = %d", opt.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", opt.accumulation_steps)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        opt.train_batch_size * opt.accumulation_steps
    )
    logger.info("  Total train steps = %d", opt.total_steps)

    while global_step < opt.total_steps:
        for step, batch in enumerate(train_dataloader):

            model.train()
            batch = tuple(t.to(opt.device) for t in batch)

            if am_flag:
                if prompt_flag:
                    index, essay_id, prompt_id, input_ids, input_masks, amlabels, scores, score_masking, prompt_embds = batch
                else:
                    index, essay_id, prompt_id, input_ids, input_masks, amlabels, scores, score_masking, amsents = batch

            else:
                # (index, essay_id, prompt_id, input_ids, input_masks, scores, score_masking)
                index, essay_id, prompt_id, input_ids, input_masks, scores, score_masking = batch
            #import pdb;pdb.set_trace()
            outputs = model(
                input_ids=input_ids,
                attention_mask=input_masks,
                #am_label = amlabels if am_flag else None,    # 임베딩 추가
                essay_amlabel=amsents,
                #prompt_embds=prompt_embds
            )

            train_logits = outputs[0]     # [batch, num_labels]

            # score scaling
            rescaled_scores = rescale_for_scoring(scores.detach().cpu().numpy(), prompt_id.detach().cpu().numpy())
            rescaled_scores = torch.tensor(rescaled_scores).to(opt.device)

            # score masking
            train_logits = torch.multiply(train_logits, score_masking)
            rescaled_scores = torch.multiply(rescaled_scores, score_masking)

            train_loss = criterion(train_logits.to(torch.float32), torch.tensor(rescaled_scores).view(-1, opt.num_labels).to(torch.float32))

            # accumulation_steps
            if opt.accumulation_steps > 1:
                train_loss = train_loss / opt.accumulation_steps

            train_loss.requires_grad_(True)     # 추가
            train_loss.backward()
            total_loss += train_loss.item()
            

            if (step + 1) % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                torch.cuda.empty_cache()

                

            if (global_step + 1) % opt.loss_print_freq == 0:
                logger.info(f"  {global_step + 1} step | Current Loss: {train_loss:.5f}")  # 3f

            #################################
            # step 마다 validation & save
            #################################
            if (global_step + 1) % opt.eval_freq == 0:
                output_path = Path(opt.checkpoint_dir) / opt.name / prompt
                output_path.mkdir(parents=True, exist_ok=True)
                _, metric_score = evaluate(
                    logger, opt, model, tokenizer, eval_dataset, collator,
                    am_flag=am_flag, output_dir=output_path, global_step=global_step+1,
                    sm_flag=True, prompt=prompt, prompt_flag=prompt_flag
                )
                # logger.info(f"  {global_step + 1} step | Avg Kappa Score = {metric_score:.5f}|")

                # save best model : avg kappa
                if (metric_score > best_score):
                    best_score = metric_score
                    logger.info(f"   |Best Avg QWK score = {best_score} | Best Step= {global_step+1} |")
                    save_model(opt, "best", model=model, prompt=prompt)  # checkpoint/model/best
            global_step += 1
                # # 모델 저장
                # save_model(opt, step=global_step, model=model, prompt=prompt)

            # #################################
            # # epoch 마다 validation / 저장
            # #################################
            # output_path = Path(opt.checkpoint_dir) / opt.name / prompt
            # output_path.mkdir(parents=True, exist_ok=True)
            # _, metric_score = evaluate(
            #     logger, opt, model, tokenizer, eval_dataset, collator,
            #     am_flag=am_flag, output_dir=output_path, global_step=global_step,
            #     sm_flag=False, prompt=None,
            # )
            # logger.info(f"   |epoch{epoch} Avg Kappa Score = {metric_score:.5f}|")
            #
            # # 모델 저장
            # save_model(opt, step=epoch, model=model, prompt=prompt)

    return model


def evaluate(logger, opt, model, tokenizer, eval_dataset, collator, am_flag=False, output_dir=None,
             global_step=None, sm_flag=False, prompt=None, prompt_flag=False):

    sampler = SequentialSampler(eval_dataset)
    dataloader = DataLoader(eval_dataset,
                            sampler=sampler,
                            batch_size=opt.eval_batch_size,
                            num_workers=10,
                            collate_fn=collator
                            )

    logger.info("*****  Validation test !!!  *****")

    model.eval()

    all_inputs, all_preds, all_golds = [], [], []

    # TODO(jin): dict 사용으로 깔끔하게 만들 것
    # trait 별 점수 저장
    # score_list = ASAP_SCORE_LIST    # 전체 trait 목록
    # trait_pred_results, trait_gold_results = {}, {}
    # for trait in score_list:
    #     trait_pred_results[trait] = []
    #     trait_gold_results[trait] = []

    overall_p, cont_p, org_p, wc_p, sf_p, conv_p, pa_p, lan_p, nar_p, stl_p, voi_p = [], [], [], [], [], [], [], [], [], [], []
    overall_g, cont_g, org_g, wc_g, sf_g, conv_g, pa_g, lan_g, nar_g, stl_g, voi_g = [], [], [], [], [], [], [], [], [], [], []


    #overall_p, cont_p, org_p, wc_p, sf_p, conv_p, pa_p, lan_p, nar_p= [], [], [], [], [], [], [], [], []
    #overall_g, cont_g, org_g, wc_g, sf_g, conv_g, pa_g, lan_g, nar_g = [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(opt.device) for t in batch)

            if am_flag:
                if prompt_flag:
                    index, essay_id, prompt_id, input_ids, input_masks, amlabels, scores, score_masking, prompt_embds = batch
                else:
                    index, essay_id, prompt_id, input_ids, input_masks, amlabels, scores, score_masking, amsents = batch

            else:
                # (index, essay_id, prompt_id, input_ids, input_masks, scores, score_masking)
                index, essay_id, prompt_id, input_ids, input_masks, scores, score_masking = batch

            outputs = model(
                input_ids=input_ids,
                attention_mask=input_masks,
                #am_label=amlabels if am_flag else None,  # 임베딩 추가
                essay_amlabel=amsents,
                #prompt_embds=prompt_embds
            )

            logits = outputs[0]  # [batch, num_labels]

            # 특정 프롬프트에 대해서만 평가하는 경우
            if sm_flag:

                if (int(prompt) < 3):     # 1/2/8
                    # "score", "content", "organization", "word_choice", "sentence_fluency", "conventions"
                    mask = torch.ByteTensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0 ,0])
                elif (int(prompt) > 2) and (int(prompt) < 7):       # 3/4/5/6
                    # "score", "content", "prompt_adherence", "language", "narrativity"
                    mask = torch.ByteTensor([1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0])
                elif int(prompt) == 7:
                    # "score", "content", "organization", "conventions"
                    mask = torch.ByteTensor([1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0])
                else:
                    mask = torch.ByteTensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1])

                mask_num = mask.sum()
                score_list = [ASAP_SCORE_LIST[i] for i, m in enumerate(mask) if m == 1]
                mask = mask.expand(logits.size(0), mask.size(0))    # [batch, mask_size(9)]

                # scaling -> masking
                logits = rescale_to_intscore(logits.detach().cpu().numpy(), prompt_id.detach().cpu().numpy())
                preds = torch.masked_select(torch.tensor(logits), mask).view(-1, mask_num)
                scores = torch.masked_select(scores.detach().cpu(), mask).view(-1, mask_num)

                # 예측 점수
                all_inputs.extend(input_ids.detach().cpu().numpy())
                all_preds.extend(np.array(preds))
                all_golds.extend(scores.numpy())

            # 여러개의 프롬프트 에세이에 대해 평가하는 경우
            else:

                # gold 값이 있는 trait 성능 측정
                score_list = ASAP_SCORE_LIST

                # 정수 변환 -> 전처리
                # TODO(jin): 정수 변환한 예측값을 다시 후처리하도록 수정 필요
                preds = rescale_to_intscore(logits.detach().cpu().numpy(), prompt_id.detach().cpu().numpy())
                # preds = post_processed(preds, max_score=100, min_score=0)    # pid마다 다른 전처리 필요

                all_inputs.extend(input_ids.detach().cpu().numpy())
                all_preds.extend(np.array(preds))
                all_golds.extend(scores.detach().cpu().numpy())

                preds_array = np.array(preds)
                golds_array = scores.detach().cpu().numpy()
                for i, pid in enumerate(prompt_id):
                    if (pid < 3) or (pid == 8):    # 1, 2, 8
                        overall_p.append(preds_array[i][0])
                        cont_p.append(preds_array[i][1])
                        org_p.append(preds_array[i][2])
                        wc_p.append(preds_array[i][3])
                        sf_p.append(preds_array[i][4])
                        conv_p.append(preds_array[i][5])

                        overall_g.append(golds_array[i][0])
                        cont_g.append(golds_array[i][1])
                        org_g.append(golds_array[i][2])
                        wc_g.append(golds_array[i][3])
                        sf_g.append(golds_array[i][4])
                        conv_g.append(golds_array[i][5])

                    elif (pid > 3) and (pid < 7):    # 3, 4, 5, 6
                        overall_p.append(preds_array[i][0])
                        cont_p.append(preds_array[i][1])
                        pa_p.append(preds_array[i][6])
                        lan_p.append(preds_array[i][7])
                        nar_p.append(preds_array[i][8])

                        overall_g.append(golds_array[i][0])
                        cont_g.append(golds_array[i][1])
                        pa_g.append(golds_array[i][6])
                        lan_g.append(golds_array[i][7])
                        nar_g.append(golds_array[i][8])

                    else:    # 7
                        overall_p.append(preds_array[i][0])
                        cont_p.append(preds_array[i][1])
                        org_p.append(preds_array[i][2])
                        conv_p.append(preds_array[i][5])

                        overall_g.append(golds_array[i][0])
                        cont_g.append(golds_array[i][1])
                        org_g.append(golds_array[i][2])
                        conv_g.append(golds_array[i][5])

    # kappa score 출력 : dev trait 별 / test trait 별
    # do test
    if sm_flag:
        logger.info(f"***** prompt {prompt} : Cross-prompt test Result *****\n")
        avg_kappa = []
        for t, trait in enumerate(score_list):
            all_trait_preds = [p[t] for p in all_preds]
            all_trait_golds = [g[t] for g in all_golds]

            # class results
            trait_kappa = cohen_kappa_score(all_trait_golds, all_trait_preds, weights='quadratic')
            trait_pearson = np.corrcoef(all_trait_golds, all_trait_preds)[0, 1]

            avg_kappa.append(trait_kappa)

            logger.info(f"    [{trait}] Kappa Score: {trait_kappa:.3f}")
            logger.info(f"    [{trait}] Pearson: {trait_pearson:.3f}\n")

        logger.info(f"    Avg Kappa Score: {np.mean(avg_kappa):.3f}\n")

    else:
        pred_list = [overall_p, cont_p, org_p, wc_p, sf_p, conv_p, pa_p, lan_p, nar_p]
        gold_list = [overall_g, cont_g, org_g, wc_g, sf_g, conv_g, pa_g, lan_g, nar_g]
        avg_kappa = []
        for t, trait in enumerate(score_list):

            # class results
            trait_kappa = cohen_kappa_score(gold_list[t], pred_list[t], weights='quadratic')
            trait_pearson = np.corrcoef(gold_list[t], pred_list[t])[0, 1]

            avg_kappa.append(trait_kappa)

            logger.info(f"    [{trait}] Kappa Score: {trait_kappa:.3f}")
            logger.info(f"    [{trait}] Pearson: {trait_pearson:.3f}\n")

        logger.info(f"      Avg Kappa Score: {np.mean(avg_kappa):.3f}")

    # 결과 저장
    all_index = [eval_data["index"] for eval_data in eval_dataset]
    all_eid = [eval_data["essay_id"] for eval_data in eval_dataset]
    all_pid = [eval_data["prompt_id"] for eval_data in eval_dataset]

    # trait 별 점수 저장
    compute_and_save_predictions(global_step, tokenizer, all_index, all_inputs, all_preds, all_golds,
                                 output_dir=output_dir, score_list=score_list, all_eid=all_eid, all_pid=all_pid)

    return all_preds, np.mean(avg_kappa)

