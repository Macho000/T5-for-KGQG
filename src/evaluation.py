from AdjacencyAttentionWithoutSelfloopTransformers import (
  AdamW,
  T5ForConditionalGeneration,
  T5Tokenizer,
  get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader

from src.dataset import JsonDatasetWQ, JsonDatasetPQ
from datetime import strftime, localtime
from tqdm.auto import tqdm

from omegaconf import OmegaConf
import pandas as pd
from core.evaluation.eval import QGEvalCap


OmegaConf.register_new_resolver("now", lambda pattern: strftime(pattern, localtime()))

def saveOutputs(inputs: list, outputs: list, targets: list) -> None:
  data = pd.DataFrame(list(zip(inputs, outputs, targets)), columns =['inputs', 'outputs', 'targets'])
  data.to_csv("out/outputs.csv",index=False, header=True)

def run_eval(target_src, decoded_text) -> None:
  assert len(target_src) == len(decoded_text)
  eval_targets = {}
  eval_predictions = {}
  for idx in range(len(target_src)):
      eval_targets[idx] = [target_src[idx]]
      eval_predictions[idx] = [decoded_text[idx]]

  QGEval = QGEvalCap(eval_targets, eval_predictions)
  scores = QGEval.evaluate()
  return scores

class Evaluation():
  def __init__(self, hparams):
    self.hparams = hparams

  def run(self):
    # Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(self.hparams.experiment.model_dir, is_fast=True)
    trained_model = T5ForConditionalGeneration.from_pretrained(self.hparams.experiment.model_dir)

    # import test data
    if self.hparams.experiment.data=="mhqg-wq":
      test_dataset = JsonDatasetWQ(tokenizer, self.hparams.experiment.data_dir, "test.json", 
                                input_max_len=self.hparams.model.max_input_length, 
                                target_max_len=self.hparams.model.max_target_length)

    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

    trained_model.eval()

    inputs = []
    outputs = []
    targets = []

    for index, batch in enumerate(tqdm(test_loader)):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']
        if self.hparams.training.n_gpu:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        output = trained_model.generate(input_ids=input_ids, 
            attention_mask=input_mask, 
            max_length=self.hparams.model.max_target_length,
            temperature=1.0,          # 生成にランダム性を入れる温度パラメータ
            repetition_penalty=1.5,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
            )

        output_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False) 
                    for ids in output]
        target_text = [tokenizer.decode(ids, skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=False) 
                    for ids in batch["target_ids"]]
        input_text = [tokenizer.decode(ids, skip_special_tokens=False, 
                                  clean_up_tokenization_spaces=False) 
                    for ids in input_ids]

        inputs.extend(input_text)
        outputs.extend(output_text)
        targets.extend(target_text)

        saveOutputs(inputs, outputs, targets)
        run_eval(targets, outputs)