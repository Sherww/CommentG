import datetime
import sys
import os
from typing import List

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)


import argparse

import copy
import tqdm
import numpy as np
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import transformers
from transformers import TranslationPipeline, Text2TextGenerationPipeline

from codeqg.question_gen.bart.dataset import BARTQuestionGenerationDataset
from codeqg.question_gen.bart.model import BartForMaskedLMModel
from codeqg.question_gen.utils import PadFunction
from codeqg.utils import question_types
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from codeqg.utils import chunked
from codeqg.decode.beam_strategy import BeamSearchSlow, BeamSearch
from codeqg.decode.greedy_strategy import GreedySearch

class MyLightningModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

def generate(model, predict_dataset, tokenizer, test_batch_size: int):
        
        print('predicting...')

        device = model.device
        model = model.to(device)
        model.eval()

        if not os.path.exists('./inference_test/'):
            os.mkdir('./inference_test/')

        search_strategy = GreedySearch(
            pad_id=tokenizer.pad_token_id,
            bos_id=tokenizer.bos_token_id,
            eos_id=tokenizer.eos_token_id,
            min_length=1,
            max_length=64,
            # top_k=3
        )

        pad_fn_object = PadFunction(pad_id=tokenizer.pad_token_id)

        def predit_fn(source_inputs: List[torch.Tensor], states: List[torch.Tensor]):
            batch_size = len(source_inputs)

            batch = pad_fn_object(list(zip(source_inputs, states)))
            output = model(source_tokens=batch[0], target_tokens=batch[1])
            return output

        # test and output
        test_datetime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        test_out = f'./inference_test/qg_bart_test_out-{test_datetime}.jsonl'
        out_data = []
        chunked_dataset = list(chunked(predict_dataset.dataset, test_batch_size))
        for data in tqdm.tqdm(chunked_dataset):
            code = []
            for d in data:
                for qtype in question_types:
                    code.append(qtype + ":" + d['code'])

            inputs = tokenizer(code, max_length=256, truncation=True, padding='longest', return_tensors='pt')

            source_inputs = inputs['input_ids'].to(device)
            batch_size = source_inputs.shape[0]
            init_states = torch.full((batch_size, 1), tokenizer.bos_token_id).to(device)
            translation_ids = search_strategy.search(source_inputs, init_states, predit_fn)

            for i, result in enumerate(translation_ids):
                d = data[i // len(question_types)]
                new_data = copy.copy(d)
                out = tokenizer.batch_decode(result, skip_special_tokens=True)

                new_data['prediction'] = out
                out_data.append(new_data)
                # print(new_data['code'], new_data['prediction'])
            
        with open(test_out, 'w', encoding='utf-8') as f:
            for data in out_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
def train(args: argparse.Namespace):
    pl.utilities.seed.seed_everything(seed=42, workers=True)

    tokenizer = transformers.BartTokenizerFast.from_pretrained("../facebook/bart-base")

    train_dataset = BARTQuestionGenerationDataset('../data/qg_data/train.jsonl.gz', tokenizer, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=16, collate_fn=PadFunction(pad_id=tokenizer.pad_token_id))

    valid_dataset = BARTQuestionGenerationDataset('../data/qg_data/valid.jsonl.gz', tokenizer, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch, num_workers=4, collate_fn=PadFunction(pad_id=tokenizer.pad_token_id))

    test_dataset = BARTQuestionGenerationDataset('../data/qg_data/test.jsonl.gz', tokenizer, tokenizer)
    # test_loader = DataLoader(test_dataset, batch_size=args.test_batch, num_workers=4, collate_fn=PadFunction(pad_id=tokenizer.pad_token_id))

    model = BartForMaskedLMModel(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.vocab_size)

    # default logger used by trainer
    # logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="qg_bart_logs")
    early_stop_callback = EarlyStopping(
        monitor='valid_loss_epoch',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        callbacks=[early_stop_callback],
        # logger=logger,
        # max_steps=100,
        max_epochs=args.max_epochs,
        default_root_dir="./qg_bart_logs",
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16, amp_backend="native",
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # trainer.test(model=model, dataloaders=test_loader)

    # if has gpu
    if torch.cuda.is_available():
        model = model.to("cuda")

    generate(model, test_dataset, tokenizer)
if __name__ == "__main__":

    tokenizer = transformers.BartTokenizerFast.from_pretrained("../facebook/bart-base")
    test_dataset = BARTQuestionGenerationDataset('../data/qg_data/test_del_dup.jsonl.gz', tokenizer, tokenizer)
    model=MyLightningModule.load_from_checkpoint('./qg_bart_logs/lightning_logs/version_0/checkpoints/epoch=8-step=301734.ckpt')

    # model=torch.load('.qg_bart_logs/lightning_logs/version_0/checkpoints/epoch=8-step=301734.ckpt')
    if torch.cuda.is_available():
        model = model.to("cuda")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--train_batch', '-trb', type=int, help='training batch size of model', required=False, default=32)
    parser.add_argument('--valid_batch', '-vab', type=int, help='validation batch size of model', required=False, default=32)
    parser.add_argument('--test_batch', '-teb', type=int, help='test batch size of model', required=False, default=16)
    parser.add_argument('--max_epochs', '-me', type=int, help='training max epoch of model', required=False, default=50)

    args = parser.parse_args()
    generate(model, test_dataset, tokenizer,args.test_batch)