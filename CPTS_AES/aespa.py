# prompt 별로 학습 -> 평가하는 AES 코드

import argparse
import os
import logging
from pathlib import Path
from src.functions.data_asap_new import *    # AsapDataset, AsapDatasetCollator
from src.functions.utils import *
from src.model.main_functions import train, evaluate  # loss 그냥 계산
from src.model.models_kop import gateway_sw
from transformers import AutoConfig, AutoTokenizer, AutoModel
# from src.model.modeling_roberta import RobertaModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import warnings

warnings.filterwarnings(action='ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger(__name__)


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--do_train', type=bool, default=True, help='Whether training or evaluation')
        self.parser.add_argument('--do_test', type=bool, default=False, help='Whether validation or test')
        self.parser.add_argument('--name', type=str, default='seen_gateway_sw_kop', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='CPTS_AES/checkpoint/', help='models are saved here')
        self.parser.add_argument('--checkpoint_step', type=str, default='best', help='checkpoint step')
        self.parser.add_argument('--device', type=str, default="cuda")
        self.parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

        # pretrained model parameters
        self.parser.add_argument('--pretrained_tokenizer', type=str, default="roberta-base",
                                 help="Load pre-trained tokenizer")
        self.parser.add_argument('--pretrained_model', type=str, default="roberta-base",
                                 help="Load pre-trained language model")

        # training parameters
        self.parser.add_argument("--train_batch_size", default=8, type=int,
                                 help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--eval_batch_size", default=8, type=int,
                                 help="Batch size per GPU/CPU for evaluation.")
        self.parser.add_argument('--loss_print_freq', type=int, default=100,  # 10    100
                                 help='print train loss <loss_print_freq> steps during training')
        self.parser.add_argument('--eval_freq', type=int, default=500)     # 500

    def add_optim_options(self):
        self.parser.add_argument('--total_steps', type=int, default=2500)  # 1
        self.parser.add_argument('--scheduler_steps', type=int, default=None,
                                 help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)  # 2
        self.parser.add_argument('--dropout_rate', type=int, default=0.3)
        self.parser.add_argument('--warmup_steps', type=int, default=1)
        self.parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adamw')
        self.parser.add_argument('--adam_epsilon', type=int, default=1e-8)
        self.parser.add_argument('--scheduler', type=str, default='linear')
        self.parser.add_argument('--weight_decay', type=float, default=0.0)

    def add_argument_options(self):
        # /home/jinji/workspace/dataset/asap/cross-prompt-trait-scoring/cross_prompt_attributes
        self.parser.add_argument('--data_dir', type=str,
                                 default='dataset/seen/')
        self.parser.add_argument('--train_data', type=str, default='new_train_plus.json')  # train / train_plus / cl_train
        self.parser.add_argument('--valid_data', type=str, default='new_dev_plus.json')
        self.parser.add_argument('--test_data', type=str, default='new_test_plus.json')  # dev / test / ..plus
        self.parser.add_argument('--am_flag', type=bool, default=True)
        self.parser.add_argument('--add_vocab', type=bool, default=False)

        # model parameter
        self.parser.add_argument('--lstm_hidden', type=int, default=256)
        self.parser.add_argument('--max_length', type=int, default=512)
        self.parser.add_argument('--lstm_num_layer', type=int, default=1)
        self.parser.add_argument('--bidirectional_flag', type=bool, default=True)
        self.parser.add_argument('--num_labels', type=int, default=11)  # overall + 8 trait

    def parse(self):
        opt = self.parser.parse_args()
        return opt


if __name__ == '__main__':

    # path setting
    current_path = Path(os.getcwd())  # /home/jinji/workspace/dataset/cross_prompt_attributes

    all_prompt = ["1", "2", "3", "4", "5", '6', '7', '8']
    
    # setting
    options = Options()
    options.add_argument_options()
    options.add_optim_options()
    opt = options.parse()

    # Path & logging
    output_path = Path(opt.checkpoint_dir) / opt.name  # / opt.tmp_prompt
    output_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger = init_logger(output_path / 'run.log')
    set_seed(opt)

    # 프롬프트 실행 확인
    if len(all_prompt) == 8:
        logger.info("Training all prompts")
    else:
        logger.info(all_prompt)

    for prompt in all_prompt:

        # info
        logger.info(f"experiment: {opt.name}")
        logger.info(f"tmp prompt: {prompt}")
        logger.info(f"SEED: {opt.seed}")
        logger.info(f"pre-trained: {opt.pretrained_model}")

        # Load model
        config = AutoConfig.from_pretrained(opt.pretrained_model)
        pre_trained_model = AutoModel.from_pretrained(opt.pretrained_model, config=config)  # "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_tokenizer)

        if opt.add_vocab:
            add_token = {}
            logger.info(f"Add tokens: {add_token}")
            tokenizer.add_special_tokens(add_token)
            pre_trained_model.resize_token_embeddings(len(tokenizer))

        # Dataset
        train_examaples = load_asap_data(file_path=os.path.join(opt.data_dir, prompt, opt.train_data),
                                         score_list=ASAP_SCORE_LIST,
                                         am_flag=opt.am_flag)
        train_dataset = AsapDataset(train_examaples, am_flag=opt.am_flag)
        valid_examples = load_asap_data(file_path=os.path.join(opt.data_dir, prompt, opt.valid_data),
                                        score_list=ASAP_SCORE_LIST,
                                        am_flag=opt.am_flag)
        valid_dataset = AsapDataset(valid_examples, am_flag=opt.am_flag)
        test_examples = load_asap_data(file_path=os.path.join(opt.data_dir, prompt, opt.test_data),
                                       score_list=ASAP_SCORE_LIST,
                                       am_flag=opt.am_flag)
        test_dataset = AsapDataset(test_examples, am_flag=opt.am_flag)

        model = gateway_sw(
            config=config,
            pre_trained_model=pre_trained_model,
            num_labels=opt.num_labels,
            n_hidden=opt.lstm_hidden
        )
        model.to(opt.device)
        collator = AsapDatasetCollator(tokenizer, opt.max_length, opt.am_flag)

        # Training Setting
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": opt.weight_decay}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)

        # 각 train별 scheduler
        # t_total = len(train_dataset) // opt.warmup_steps * opt.total_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_steps,
                                                    num_training_steps=opt.total_steps)  # opt.total_steps   t_total

        if opt.do_train:
            train(logger, opt, model, tokenizer, optimizer, scheduler,
                  train_dataset, valid_dataset, collator, opt.am_flag, prompt=prompt)

        # best model -> prompt test
        if opt.do_test:
            test_model = load_model(opt, model, prompt)
            evaluate(logger, opt, test_model, tokenizer, test_dataset, collator, opt.am_flag,
                     output_dir=os.path.join(output_path, prompt), global_step=None, sm_flag=True,
                     prompt=prompt)