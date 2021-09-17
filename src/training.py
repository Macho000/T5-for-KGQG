import pytorch_lightning as pl
from AdjacencyAttentionWithoutSelfloopTransformers import (
  AdamW,
  T5ForConditionalGeneration,
  T5Tokenizer,
  get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader

from src.dataset import JsonDatasetWQ, JsonDatasetPQ


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # 事前学習済みモデルの読み込み
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.experiment.model_name_or_path)

        # トークナイザーの読み込み
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.experiment.tokenizer_name_or_path, is_fast=True)

        special_tokens_dict = {'sep_token': '<sep>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        special_tokens_dict = {'sep_token': '<answer>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        special_tokens_dict = {'sep_token': '<SEP>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        special_tokens_dict = {'sep_token': '<subject>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        special_tokens_dict = {'sep_token': '<relation>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        special_tokens_dict = {'sep_token': '<object>'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            three_dim_attention_mask=True
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked), 
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        logit = outputs[1]
        return (loss, logit)

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss, logit = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss, logit = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.hparams.training.learning_rate, 
                          eps=self.hparams.training.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.training.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_dataset(self, tokenizer, type_path, args, train_mode=None):
        """データセットを作成する"""
        if args.experiment.data=="mhqg-wq":
          return JsonDatasetWQ(
            tokenizer=tokenizer, 
            data_dir=args.experiment.data_dir, 
            type_path=type_path, 
            input_max_len=args.model.max_input_length,
            target_max_len=args.model.max_target_length,
            train_mode=train_mode)
        else:
          return JsonDatasetPQ(
            tokenizer=tokenizer, 
            data_dir=args.experiment.data_dir, 
            type_path=type_path, 
            input_max_len=args.model.max_input_length,
            target_max_len=args.model.max_target_length,
            train_mode=train_mode)
    
    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                             type_path="train.json", args=self.hparams, train_mode=True)
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                           type_path="dev.json", args=self.hparams, train_mode=False)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) // (self.hparams.training.train_batch_size * max(1, self.hparams.training.n_gpu)))
                // self.hparams.training.gradient_accumulation_steps
                * float(self.hparams.training.num_train_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset, 
                          batch_size=self.hparams.training.train_batch_size, 
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset, 
                          batch_size=self.hparams.training.eval_batch_size, 
                          num_workers=4)