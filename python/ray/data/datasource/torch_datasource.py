import math

import torch

from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.datasource import Datasource, Reader, ReadTask, WriteResult


class TorchDatasource(Datasource):
    def create_reader(self, dataset: torch.utils.data.Dataset):
        return _TorchDatasourceReader(dataset)


class _TorchDatasourceReader(Reader):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self._dataset = dataset

    def get_read_tasks(self, parallelism):
        rows = len(self._dataset)
        rows_per_worker = math.ceil(rows / parallelism)
        shard_start = 0
        read_tasks = []
        for _ in range(parallelism):
            shard_end = min(shard_start + rows_per_worker, rows)
            meta = BlockMetadata(
                num_rows=shard_end - shard_start,
                size_bytes=None,
                schema=None,
                input_files=None,
                exec_stats=None,
            )
            read_tasks.append(
                ReadTask(
                    lambda ds=self._dataset, start=shard_start, end=shard_end: _read_shard(
                        ds, start, end
                    ),
                    metadata=meta,
                ),
            )
            shard_end += rows_per_worker

        return read_tasks

    def estimate_inmemory_data_size(self):
        return 0


def _read_shard(dataset, start, end):
    shard = torch.utils.data.Subset(dataset, range(start, end))
    data_loader = torch.utils.data.DataLoader(
        # default_collate does not accept `PIL.Image.Image`s
        shard, collate_fn=lambda x: x, batch_size=5
    )

    for _, batch in enumerate(data_loader):
        builder = DelegatingBlockBuilder()
        for item in batch:
            builder.add({"item": item})
        block = builder.build()
        yield block

