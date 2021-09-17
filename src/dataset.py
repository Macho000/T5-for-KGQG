from torch.utils.data import Dataset
import numpy as np
import json
import torch
from collections import Counter, defaultdict, OrderedDict
import gc
import os

class JsonDatasetWQ(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512, train_mode=None):
        assert train_mode==True or train_mode==False
        self.file_path = os.path.join(data_dir, type_path)
        
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
 
        self.list_index = 0
 
        self.train_mode = train_mode
 
        if train_mode:
          self.SOURCE_ID_PATH = "/SOURCE_ID_memmap.npy"
          self.SOURCE_MASK_PATH = "/SOURCE_MASK_memmap.npy"
          self.SOURCE_CROSS_MASK_PATH = "/SOURCE_CROSS_MASK_memmap.npy"
          self.TARGET_ID_PATH = "/TARGET_ID_memmap.npy"
          self.TARGET_MASK_PATH = "/TARGET_MASK_memmap.npy"
          self.input_length = 18624
          # self.input_length = 18989
          SOURCE_ID_memmap = np.memmap(
            filename=self.SOURCE_ID_PATH, dtype=np.int64, mode="w+",shape=(self.input_length,512) 
          )
 
          SOURCE_MASK_memmap = np.memmap(
            filename=self.SOURCE_MASK_PATH, dtype=np.int64, mode="w+", shape=(self.input_length,512,512)
          )

          SOURCE_CROSS_MASK_memmap = np.memmap(
            filename=self.SOURCE_CROSS_MASK_PATH, dtype=np.int64, mode="w+", shape=(self.input_length,512)
          )
 
          TARGET_ID_memmap = np.memmap(
            filename=self.TARGET_ID_PATH, dtype=np.int64, mode="w+", shape=(self.input_length,100)
          )
 
          TARGET_MASK_memmap = np.memmap(
            filename=self.TARGET_MASK_PATH, dtype=np.int64, mode="w+",shape=(self.input_length,100)
          )
 
          del SOURCE_ID_memmap
          del SOURCE_MASK_memmap
          del SOURCE_CROSS_MASK_memmap
          del TARGET_ID_memmap
          del TARGET_MASK_memmap
        else:
          self.SOURCE_ID_PATH = "/VAL_SOURCE_ID_memmap.npy"
          self.SOURCE_MASK_PATH = "/VAL_SOURCE_MASK_memmap.npy"
          self.SOURCE_CROSS_MASK_PATH = "/VAL_SOURCE_CROSS_MASK_memmap.npy"
          self.TARGET_ID_PATH = "/VAL_TARGET_ID_memmap.npy"
          self.TARGET_MASK_PATH = "/VAL_TARGET_MASK_memmap.npy"
          self.input_length = 1985
          VAL_SOURCE_ID_memmap = np.memmap(
            filename=self.SOURCE_ID_PATH, dtype=np.int64, mode="w+",shape=(self.input_length,512) 
          )
 
          VAL_SOURCE_MASK_memmap = np.memmap(
            filename=self.SOURCE_MASK_PATH, dtype=np.int64, mode="w+", shape=(self.input_length,512,512)
          )

          VAL_SOURCE_CROSS_MASK_memmap = np.memmap(
            filename=self.SOURCE_CROSS_MASK_PATH, dtype=np.int64, mode="w+", shape=(self.input_length,512)
          )
 
          VAL_TARGET_ID_memmap = np.memmap(
            filename=self.TARGET_ID_PATH, dtype=np.int64, mode="w+", shape=(self.input_length,100)
          )
 
          VAL_TARGET_MASK_memmap = np.memmap(
            filename=self.TARGET_MASK_PATH, dtype=np.int64, mode="w+",shape=(self.input_length,100)
          )
 
          del VAL_SOURCE_ID_memmap
          del VAL_SOURCE_MASK_memmap
          del VAL_SOURCE_CROSS_MASK_memmap
          del VAL_TARGET_ID_memmap
          del VAL_TARGET_MASK_memmap
 
        self.SOURCE_ID_memmap = np.memmap(
          filename=self.SOURCE_ID_PATH, dtype=np.int64, mode="r",shape=(self.input_length,512) 
        )
 
        self.SOURCE_MASK_memmap = np.memmap(
          filename=self.SOURCE_MASK_PATH, dtype=np.int64, mode="r", shape=(self.input_length,512,512)
        )

        self.VAL_SOURCE_CROSS_MASK_memmap = np.memmap(
          filename=self.SOURCE_CROSS_MASK_PATH, dtype=np.int64, mode="r", shape=(self.input_length,512)
        )
 
        self.TARGET_ID_memmap = np.memmap(
          filename=self.TARGET_ID_PATH, dtype=np.int64, mode="r", shape=(self.input_length,100)
        )
 
        self.ARGET_MASK_memmap = np.memmap(
          filename=self.TARGET_MASK_PATH, dtype=np.int64, mode="r",shape=(self.input_length,100)
        )
 
        self._build()
  
    def __len__(self):
        return self.list_index
  
    def __getitem__(self, index):
        return {"source_ids": torch.from_numpy(np.array(self.SOURCE_ID_memmap[index])).squeeze(), "source_mask": torch.from_numpy(np.array(self.SOURCE_MASK_memmap[index])).squeeze(), "cross_attention_mask":torch.from_numpy(np.array(self.VAL_SOURCE_CROSS_MASK_memmap[index])).squeeze(),
                "target_ids": torch.from_numpy(np.array(self.TARGET_ID_memmap[index])).squeeze(), "target_mask": torch.from_numpy(np.array(self.ARGET_MASK_memmap[index])).squeeze()}
        # source_ids = self.inputs[index]["input_ids"].squeeze()
        # target_ids = self.targets[index]["input_ids"].squeeze()
 
        # source_mask = self.inputs[index]["attention_mask"].squeeze()
        # target_mask = self.targets[index]["attention_mask"].squeeze()
 
        
 
        # return {"source_ids": source_ids, "source_mask": source_mask, 
        #         "target_ids": target_ids, "target_mask": target_mask}
 
    def _make_record(self, answer, input, target):
        # ニュースタイトル生成タスク用の入出力形式に変換する。
        input = f"{answer}{input}"
        target = f"{target}"
        return input, target
 
    def _file_controller(self):
      for index, line in enumerate(open(self.file_path, 'r')):
        yield index, line
        # with open(self.file_path, 'r') as f:
        #     for index, line in enumerate(f):
        #       yield index, line
  
    def _build(self):
        with open(self.file_path, 'r') as f:
            self.list_index = 0
            for index, line in enumerate(f):
            # for index, line in self._file_controller():
                # line = f.readline()
                line = line.strip()
                jo = json.loads(line, object_pairs_hook=OrderedDict)
                assert len(jo['inGraph']['g_adj']) > 0

                answers = jo['answers']
                normalized_answers = ""
                for x in answers:
                    normalized_answers += (x + " ")
                target = jo['outSeq']
                n = len(jo['inGraph']['g_node_names'])
                adj_matrix = {}
                adj_matrix2 = {}
                graph = {'node_name_id2word':{}}
                for idx, nid in enumerate(jo['inGraph']['g_node_names']):
                    # replace answer token to "<answer> answer <answer>"
                    if jo['inGraph']['g_node_names'][nid] in answers:
                      graph['node_name_id2word'][nid] = "<answer> "+ jo['inGraph']['g_node_names'][nid] + " <answer>"
                    else:
                      graph['node_name_id2word'][nid] = jo['inGraph']['g_node_names'][nid]

                    # create adj_matrix with 0 value
                    for idy, nid2 in enumerate(jo['inGraph']['g_node_names']):
                      if jo['inGraph']['g_node_names'][nid2] in answers:
                        adj_matrix2["<answer> "+ jo['inGraph']['g_node_names'][nid2] + " <answer>"] = 0
                      else:
                        adj_matrix2[jo['inGraph']['g_node_names'][nid2]] = 0
                    if jo['inGraph']['g_node_names'][nid] in answers:
                      adj_matrix["<answer> "+ jo['inGraph']['g_node_names'][nid] + " <answer>"] = adj_matrix2.copy()
                    else:
                      adj_matrix[jo['inGraph']['g_node_names'][nid]] = adj_matrix2.copy()
                for nid, val in jo['inGraph']['g_adj'].items():
                    for nid2, edge in val.items():
                        edge = edge.split('/')[-1]
                        adj_matrix[graph["node_name_id2word"][nid]][graph["node_name_id2word"][nid2]] = 1
                input_knowledge_graph = ""
                for i,ids in enumerate(list(adj_matrix.keys())):
                  input_knowledge_graph += (ids + " ")

                input, target = input_knowledge_graph, target

                tokenizer = T5Tokenizer.from_pretrained("t5-small", is_fast=True)
                special_tokens_dict = {'sep_token': '<sep>'}
                tokenizer.add_special_tokens(special_tokens_dict)

                special_tokens_dict = {'sep_token': '<answer>'}
                tokenizer.add_special_tokens(special_tokens_dict)

                special_tokens_dict = {'sep_token': '<SEP>'}
                tokenizer.add_special_tokens(special_tokens_dict)

                special_tokens_dict = {'sep_token': '<subject>'}
                tokenizer.add_special_tokens(special_tokens_dict)

                special_tokens_dict = {'sep_token': '<relation>'}
                tokenizer.add_special_tokens(special_tokens_dict)

                special_tokens_dict = {'sep_token': '<object>'}
                tokenizer.add_special_tokens(special_tokens_dict)
                tokenized_inputs = tokenizer.batch_encode_plus_self_attention(
                    [input_knowledge_graph], adjacancy_matrix=adj_matrix, max_length=512, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                tokenized_targets = tokenizer.batch_encode_plus(
                    [target], max_length=100, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                tokenized_row_input = ""
                tokenized_row_input = self.tokenizer.tokenize(input)
                if len(tokenized_row_input) > 512:
                  continue
                
                SOURCE_ID_memmap = np.memmap(
                  filename=self.SOURCE_ID_PATH, dtype=np.int64, mode="r+",shape=(self.input_length,512) 
                )
 
                SOURCE_MASK_memmap = np.memmap(
                  filename=self.SOURCE_MASK_PATH, dtype=np.int64, mode="r+", shape=(self.input_length,512,512)
                )

                SOURCE_CROSS_ATTENTION_MASK_memmap = np.memmap(
                  filename=self.SOURCE_CROSS_MASK_PATH, dtype=np.int64, mode="r+", shape=(self.input_length,512)
                )
 
                TARGET_ID_memmap = np.memmap(
                  filename=self.TARGET_ID_PATH, dtype=np.int64, mode="r+", shape=(self.input_length,100)
                )
 
                TARGET_MASK_memmap = np.memmap(
                  filename=self.TARGET_MASK_PATH, dtype=np.int64, mode="r+",shape=(self.input_length,100)
                )
 
                SOURCE_ID_memmap[self.list_index] = tokenized_inputs["input_ids"].squeeze().numpy()
                SOURCE_MASK_memmap[self.list_index] = tokenized_inputs["attention_mask"].squeeze().numpy()
                SOURCE_CROSS_ATTENTION_MASK_memmap[self.list_index] = tokenized_inputs["cross_attention_mask"].squeeze().numpy()
 
                TARGET_ID_memmap[self.list_index] = tokenized_targets["input_ids"].squeeze().numpy()
                TARGET_MASK_memmap[self.list_index] = tokenized_targets["attention_mask"].squeeze().numpy()
 
                del SOURCE_ID_memmap
                del SOURCE_MASK_memmap
                del SOURCE_CROSS_ATTENTION_MASK_memmap
                del TARGET_ID_memmap
                del TARGET_MASK_memmap
 
                self.list_index += 1