# Copyright (c) Meta Platforms, Inc. and affiliates.
import re
from logging import getLogger
from pathlib import Path
from typing import Any, Generator

import pyarrow as pa

# pyarrow needs the initialization from this import
import pyarrow.dataset  # pyright: ignore
from pydantic import BaseModel, ConfigDict

from bytelatent import ByteLatentError
from bytelatent.data.data_types import BltExample
from bytelatent.data.iterators.abstract_iterator import IteratorState, StatefulIterator

logger = getLogger(__name__)


class ArrowFileIteratorState(BaseModel, IteratorState):
    model_config = ConfigDict(extra="forbid")
    file_path: str
    row_num: int
    num_workers: int
    worker_id: int
    preprocess_dir: str
    dataset_files: list[str]
    entropy_model_name: str
    arrow_batch_size: int = 100

    def build(self) -> "ArrowFileIterator":
        arrow_file = ArrowFileIterator(
            file_path=self.file_path,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            preprocess_dir=self.preprocess_dir,
            entropy_model_name=self.entropy_model_name,
            arrow_batch_size=self.arrow_batch_size,
            dataset_files=self.dataset_files,
        )
        if self.row_num != 0:
            arrow_file._set_row_num(self.row_num)
        return arrow_file


def shard_sort_key(file: str ):
    match = re.search(r".+\.shard_([0-9]+)\.arrow", str(file))
    shard_number = int(match.group(1))
    return shard_number


class ArrowFileIterator(StatefulIterator):
    def __init__(
        self,
        *,
        file_path: str ,
        worker_id: int,
        num_workers: int,
        preprocess_dir: str ,
        entropy_model_name: str ,
        arrow_batch_size: int,
        dataset_files: list[str] = None,
    ):
        assert 0 <= worker_id < num_workers, (worker_id, num_workers)
        if file_path is None and dataset_files is None:
            raise ByteLatentError("file_path and dataset_files cannot both be None")
        self.row_num = 0
        self.iter_id = 0
        self.batch_iterator = None
        self.batch_to_consume = None
        self.dataset = None
        self.file_path = file_path
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.preprocess_dir = preprocess_dir
        self.entropy_model_name = entropy_model_name
        self.arrow_batch_size = arrow_batch_size
        if dataset_files is None:
            # Prepare arrow shards
            jsonl_file = Path(file_path)
            parts = re.match(r"(.+)\.chunk\.[0-9]+\.jsonl", jsonl_file.name)
            assert parts is not None
            dataset = parts.group(1)
            data_dir = Path(preprocess_dir) / dataset / entropy_model_name
            shard_files = list(data_dir.glob(f"{jsonl_file.name}.shard_*.arrow"))
            for s in shard_files:
                if not (data_dir / f"{s.name}.complete").exists():
                    raise ValueError(f"Missing .complete for input file: {s}")

            shard_files = sorted(shard_files, key=shard_sort_key)
            if len(shard_files) == 0:
                raise ByteLatentError(
                    f"Zero shard_files found corresponding to: {file_path} using preprocess_dir={preprocess_dir} and entropy_model_name={entropy_model_name}, so the search path is data_dir={data_dir} for matches to {jsonl_file.name}.shard_*.arrow"
                )
            self.dataset_files = [str(f) for f in shard_files]
        else:
            self.preprocess_dir = None
            self.dataset_files = dataset_files

    def get_state(self) -> ArrowFileIteratorState:
        return ArrowFileIteratorState(
            file_path=self.file_path,
            row_num=self.row_num,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            preprocess_dir=self.preprocess_dir,
            entropy_model_name=self.entropy_model_name,
            arrow_batch_size=self.arrow_batch_size,
            dataset_files=self.dataset_files,
        )

    def create_iter(
        self,
    ) -> Generator[BltExample, Any, None]:
        if self.dataset is None:
            self.dataset = pa.dataset.dataset(self.dataset_files, format="arrow")
            self.batch_iterator = self.dataset.to_batches(
                batch_size=self.arrow_batch_size
            )
        self.iter_id += 1
        if self.batch_to_consume is not None:
            batch_columns: dict[str, list] = self.batch_to_consume
            self.batch_to_consume = None
            sample_ids = batch_columns["sample_id"]
            texts = batch_columns["text"]
            entropies = batch_columns["entropies"]
            for i in range(len(sample_ids)):
                out = BltExample(
                    sample_id=sample_ids[i],
                    entropies=entropies[i],
                    text=texts[i],
                    tokens=None,
                    mask=None,
                    patch_lengths=None,
                )
                self.row_num += 1
                if (self.row_num - 1) % self.num_workers == self.worker_id:
                    yield out

        for batch in self.batch_iterator:
            batch_columns = batch.to_pydict()
            sample_ids = batch_columns["sample_id"]
            texts = batch_columns["text"]
            entropies = batch_columns["entropies"]
            for i in range(len(sample_ids)):
                out = BltExample(
                    sample_id=sample_ids[i],
                    entropies=entropies[i],
                    text=texts[i],
                    tokens=None,
                    mask=None,
                    patch_lengths=None,
                )
                self.row_num += 1
                if (self.row_num - 1) % self.num_workers == self.worker_id:
                    yield out

    def _set_row_num(self, target_row_num: int):
        logger.info(
            f"Setting arrow position to {target_row_num} for {self.dataset_files}"
        )
        if target_row_num is None or target_row_num == 0:
            self.row_num = 0
            self.dataset = None
            self.batch_iterator = None
            self.batch_to_consume = None
        else:
            self.dataset = pa.dataset.dataset(self.dataset_files, format="arrow")
            self.batch_iterator = self.dataset.to_batches(
                batch_size=self.arrow_batch_size
            )
            curr_remaining = target_row_num
            for batch in self.batch_iterator:
                if len(batch) > curr_remaining:
                    batch_columns: dict[str, list] = batch.to_pydict()
                    batch_columns["sample_id"] = batch_columns["sample_id"][
                        curr_remaining:
                    ]
                    batch_columns["entropies"] = batch_columns["entropies"][
                        curr_remaining:
                    ]
                    batch_columns["text"] = batch_columns["text"][curr_remaining:]
                    self.batch_to_consume = batch_columns
                    break
                elif len(batch) == curr_remaining:
                    # We are exactly at the end of the batch,
                    # so the next batch is the right spot
                    break
                else:
                    curr_remaining -= len(batch)
            self.row_num = target_row_num
        logger.info(
            f"Finished setting arrow position to {target_row_num} for {self.dataset_files}"
        )


TRAIN_DATA_FILE_PATTERN = "*.chunk.*.jsonl"


def find_and_sanitize_chunks(
    dataset_path: str, world_size: int, file_pattern: str = TRAIN_DATA_FILE_PATTERN
):
    dataset_chunks = [str(p) for p in Path(dataset_path).glob(file_pattern)]
    n_chunks = len(dataset_chunks)

    if n_chunks > world_size:
        n_discard = n_chunks - world_size
        dataset_chunks = dataset_chunks[:world_size]
    else:
        assert (
            world_size % n_chunks == 0
        ), "World size should be a multiple of number of chunks"

    assert n_chunks > 0, f"No valid chunks in {dataset_path}"

    return dataset_chunks
