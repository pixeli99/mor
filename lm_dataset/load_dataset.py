import os
import json
import warnings

import torch
from datasets import load_dataset, interleave_datasets

from lm_dataset.language_modeling_dataset import LanguageModelingDataset
from lm_dataset.tokenized_dataset import TokenizedCorpusDataset
from lm_dataset.data_preprocessing import AddLabels, RemoveIndex
from paths import DATA_DIR

num_proc = 24

# Local fineweb-edu sample-10BT parquet shards. Shard 013 is the held-out val
# split (fineweb_test); it must never appear in the training stream.
FINEWEB_EDU_10BT = "/jfs/auto.prod.sz/data/ann/vlf/data/pengxiang.li/fineweb_edu/sample/10BT"

# Local allenai/dolma3_mix-150B-1025: 6081 jsonl.zst shards grouped by source
# (common_crawl-<topic>-*, olmocr, stack_edu, arxiv, finemath, wiki). Shards are
# NOT pre-mixed across sources; the training loader round-robins 32 sub-streams
# of a seed-shuffled file order so batches mix sources at the document level.
# Per source family, every 600th shard is held out as dolma3_test and must
# never appear in the training stream.
DOLMA3_MIX_150B = "/jfs/auto.prod.sz/data/ann/vlf/data/pengxiang.li/dolma3_mix_150b/data"


def _dolma3_splits():
    from glob import glob
    files = sorted(glob(f"{DOLMA3_MIX_150B}/*/shard_*.jsonl.zst"))
    assert len(files) == 6081, f"dolma3 mix: expected 6081 shards, found {len(files)} (jfs mount?)"
    # hold out every 600th shard *within each source family* so the val mix
    # tracks the training mix (a flat stride over-weights the many-small-file
    # sources) and every family contributes at least one shard
    by_src = {}
    for f in files:
        by_src.setdefault(f.rsplit("/", 2)[1].split("-")[0], []).append(f)
    held_out = sorted(f for group in by_src.values() for f in group[::600])
    return [f for f in files if f not in set(held_out)], held_out


_DOLMA3_TRAIN_FILES, _DOLMA3_TEST_FILES = _dolma3_splits()

# arguments for the load_dataset function
LM_DATASETS = {
    "slimpajama": {"path": f"{DATA_DIR}/slimpajama", "split": "train"},
    "slimpajama_chunk1": {"path": "json", "data_files": f"{DATA_DIR}/slimpajama_chunk1/*.jsonl", "split": "train"},
    "cosmopedia": {"path": f"{DATA_DIR}/cosmopedia-v2", "split": "train"},
    "fineweb_edu": {"path": "parquet", "data_files": [f"{FINEWEB_EDU_10BT}/{i:03d}_00000.parquet" for i in range(13)], "split": "train"},
    "fineweb_test": {"path": "parquet", "data_files": [f"{FINEWEB_EDU_10BT}/013_00000.parquet"], "split": "train"},
    "dolma3_mix": {"path": "json", "data_files": _DOLMA3_TRAIN_FILES, "split": "train"},
    "dolma3_test": {"path": "json", "data_files": _DOLMA3_TEST_FILES, "split": "train"},
    "python_edu": {"path": f"{DATA_DIR}/python-edu", "split": "train"},
    "open_web_math": {"path": f"{DATA_DIR}/open-web-math", "split": "train"}, 
    "math_code_pile": {"path": f"{DATA_DIR}/math-code-pile", "split": "train"}, 
    "starcoderdata": {"path": f"{DATA_DIR}/starcoderdata", "split": "train"},  # "data_dir": "python", 
    "finemath": {"path": f"{DATA_DIR}/finemath", "split": "train"},  # "name": "finemath-4plus", 
}

# tokenizer used for pre-tokenization
TOKENIZED_DATASETS = {
    "pythia_pile": "pythia",  
}


def load_dataset_from_config(cfg, tokenizer):
    dataset_name = cfg.dataset.split(',')
    dataset_name = [ds.strip() for ds in dataset_name]
    if len(dataset_name) > 1:
        assert all(ds in LM_DATASETS for ds in dataset_name), "Only LM datasets can be combined"
        assert "weights" in cfg, "When combining datasets, weights must be provided"
        assert len(dataset_name) == len(cfg.weights.split(',')), "Number of weights must match number of datasets"
    
    if all(ds in LM_DATASETS for ds in dataset_name):
        dataset_type = "lm"
        # if "redpajama" in cfg.dataset and cfg.get("redpajama_path"):
        #     os.environ["RED_PAJAMA_DATA_DIR"] = cfg.redpajama_path
        # if "dolma" in cfg.dataset and cfg.get("dolma_path"):
        #     os.environ["DATA_DIR"] = cfg.dolma_path
        
        train_dataset = []
        for ds in dataset_name:
            if ds == "dolma3_mix":
                # shards are topic-grouped and a single sequential stream feeds
                # each global batch from ONE topic shard for ~40 steps; round-
                # robin over 32 sub-streams of a seed-shuffled file order so
                # every batch mixes ~32 shards at the document level
                import random as _random
                files = list(_DOLMA3_TRAIN_FILES)
                _random.Random(42).shuffle(files)
                parts = [load_dataset("json", data_files=files[k::32], split="train",
                                      streaming=True).select_columns(["text"])
                         for k in range(32)]
                train_dataset.append(interleave_datasets(parts))
                continue
            _dataset = load_dataset(**LM_DATASETS[ds], streaming=True)
            if ds == "starcoderdata":
                # train_dataset.append(load_dataset(**LM_DATASETS[ds], num_proc=num_proc))
                # train_dataset[-1] = train_dataset[-1].map(download_contents, input_columns="blob_id", num_proc=num_proc)
                # train_dataset[-1] = train_dataset[-1].filter(lambda x: x["download_success"], num_proc=num_proc)
                _dataset.rename_column("content", "text")
            if ds.startswith("dolma3"):
                # sources carry different metadata schemas (cc lacks `source`,
                # arxiv adds `doc`/`attributes`); project to text so streaming
                # never has to reconcile them across file boundaries
                _dataset = _dataset.select_columns(["text"])
            # if ds == "python_edu":
            #     dataset_text_field.append("blob_id")
            train_dataset.append(_dataset)
        
        if len(train_dataset) == 1:
            train_dataset = train_dataset[0]
        else:
            train_dataset = interleave_datasets(train_dataset, probabilities=cfg.weights.split(','), seed=42)
        
    elif all(ds in TOKENIZED_DATASETS for ds in dataset_name):
        dataset_type = "token"
        # check if tokenizer used by dataset is compatible with the one specified in config
        if "tokenizer" in cfg:
            tokenizer_used = TOKENIZED_DATASETS[cfg.dataset]
            if cfg.tokenizer != tokenizer_used:
                raise ValueError(f"Tokenizer {cfg.tokenizer} is not compatible with dataset {cfg.dataset}")

        # load corpus
        if cfg.dataset == "pythia_pile":
            from lm_dataset.tokenized_dataset import PythiaPileTokenizedCorpus
            corpus = PythiaPileTokenizedCorpus(os.path.join(DATA_DIR, "pythia_pile_idxmaps"))

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    
    transforms = [
        AddLabels(),
        RemoveIndex(),
    ]
    
    if dataset_type == "lm":
        return LanguageModelingDataset(train_dataset, tokenizer, 
                                       max_length=cfg.max_length,
                                       transforms=transforms, 
                                       global_shuffling=cfg.get("global_shuffling", False),
                                       local_shuffling=cfg.get("local_shuffling", False),
                                       add_bos_token=cfg.get("add_bos_token", False),)
    
    elif dataset_type == "token":
        if cfg.dataloader_num_workers <= 1:
            warnings.warn(f"Using cfg.dataloader_num_workers={cfg.dataloader_num_workers} with TokenizedCorpusDataset."
                          f"You may want to increase this number to speed up data loading.")
        return TokenizedCorpusDataset(corpus, length=cfg.max_length, eos_token=tokenizer.eos_token_id,
                                      add_bos_token=cfg.get("add_bos_token", False),
                                      bos_token=tokenizer.bos_token_id, transforms=transforms,)    
      
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")