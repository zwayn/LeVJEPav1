"""Native video stores: one Lance row per video, every frame decoded at its own size.

Written by the jepav1 native builder (schema metadata key `jepav1_native`, layout 1), e.g. SSv2 as
`ssv2_native/{train,validation,test}.lance`. The columns read here:

| column           | type         | content                                                     |
|------------------|--------------|-------------------------------------------------------------|
| num_valid_frames | int32        | N, decoded frames of the video                              |
| height, width    | int32        | decoded frame size (aspect ratio kept)                      |
| pixels           | large_binary | raw (N, height, width, 3) uint8, frames in display order    |
| label_idx        | int32        | class (not used by pretraining)                             |

The store also holds the raw AV1 block data (motion_vectors (N, h, w, 4) int16 in 1/8 pel, reference_map
(N, h, w, 2) int8, frame_numbers (N,) int32, ref_numbers (N, 8) int32, on the 4x4 block grid mv_height x mv_width),
which pretraining does not read yet.

A batch request is a (B, 1 + T) int64 array: column 0 is the row (video) of each clip, columns 1: the indices of its T
frames inside that video. NativeBatchSampler draws them, NativeVideoDataset reads the rows and returns the same batch
as data.WindowedLanceDataset, {"clip": (B, T, height, width, 3) uint8}, so ViewBuilder, the model and the loss are
unchanged. Keeping the frame choice in the sampler keeps the dataset deterministic, which the cached validation
batches (data.CachedBatches, keyed by the first row of a request) rely on.
"""
import json
import warnings

import lance
import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F

NATIVE_METADATA_KEY = b"jepav1_native"
NATIVE_LAYOUT_VERSION = 1
SAMPLINGS = ("stride", "segment")


def select_frames(num_frames, length, stride=1, sampling="stride", rng=None):
    """(B, length) frame indices of one clip in each video of `num_frames` (B,) frames.

    stride:  `length` frames `stride` apart, from a random start (rng) or the centred window (rng=None). A video
             shorter than the window starts at 0 and repeats its last frame on the stride, as data.Episodes pads.
    segment: TSN / SSv2 style, the video is cut into `length` equal segments and one frame is taken in each, at a
             random position (rng) or the segment's middle (rng=None), so a clip always spans the whole video.
    """
    n = np.asarray(num_frames, dtype=np.int64)
    steps = np.arange(length)
    if sampling == "stride":
        room = n - ((length - 1) * stride + 1)   # number of extra start positions; < 0: the window is padded
        if rng is None:
            start = np.maximum(room, 0) // 2
        else:
            start = np.floor(rng.random(len(n)) * (np.maximum(room, 0) + 1)).astype(np.int64)
        last = start + (n - 1 - start) // stride * stride   # last frame on the stride inside the video
        return np.minimum(start[:, None] + steps * stride, last[:, None])
    if sampling == "segment":
        offset = 0.5 if rng is None else rng.random((len(n), length))
        frames = np.floor((steps + offset) * (n[:, None] / length)).astype(np.int64)
        return np.minimum(frames, n[:, None] - 1)
    raise ValueError(f"sampling must be one of {SAMPLINGS}, got {sampling!r}")


def _binary_views(column):
    """Zero-copy memoryviews of the values of a (large_)binary Arrow column: a native row is ~14 MB, and to_pylist()
    would copy every one of them into a Python bytes object before a single frame is used."""
    views = []
    for chunk in column.chunks:
        offset_type = np.int64 if pa.types.is_large_binary(chunk.type) else np.int32
        _, offsets, data = chunk.buffers()
        offsets = np.frombuffer(offsets, dtype=offset_type)[chunk.offset:chunk.offset + len(chunk) + 1]
        data = memoryview(data) if data is not None else memoryview(b"")
        views += [data[offsets[i]:offsets[i + 1]] for i in range(len(chunk))]
    return views


def resize_frames(frames, height, width):
    """uint8 (T, H, W, 3) -> uint8 (T, height, width, 3) tensor: antialiased bilinear resize, as data.decode (the
    aspect ratio is not kept; the crops of ViewBuilder take their own aspect ratio afterwards)."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2)   # channels-last strides, what the uint8 kernel is fast on
    if x.shape[-2:] != (height, width):
        x = F.interpolate(x, size=(height, width), mode="bilinear", antialias=True, align_corners=False)
    return x.permute(0, 2, 3, 1)


class NativeVideoDataset:
    """Clips of a native video store. __getitems__ takes a (B, 1 + T) request (see the module docstring); every row
    of a request is read once, also when several of its clips are in the batch (clips_per_video > 1).

    rows_per_take bounds a worker's memory: rows are read that many at a time (~14 MB each for SSv2)."""

    def __init__(self, path, height, width, rows_per_take=8):
        self.path = str(path)
        self.height, self.width = height, width
        self.rows_per_take = rows_per_take
        ds = lance.dataset(self.path)
        check_native_store(ds)
        meta = ds.to_table(columns=["num_valid_frames", "height", "width"])
        self.num_frames = meta.column("num_valid_frames").to_numpy().astype(np.int64)
        self.frame_height = meta.column("height").to_numpy().astype(np.int64)
        self.frame_width = meta.column("width").to_numpy().astype(np.int64)
        assert len(self.num_frames) > 0, f"{self.path} is empty"
        self.ds = None   # opened lazily in every process (dataloader worker) that reads

    def __len__(self):
        return len(self.num_frames)

    def __getstate__(self):
        return {**self.__dict__, "ds": None}

    def __getitems__(self, request):
        request = np.asarray(request)
        rows, frames = request[:, 0], request[:, 1:]
        assert np.all(frames < self.num_frames[rows][:, None]), "frame index past the end of its video"
        # decode straight into shared memory, as WindowedLanceDataset: the DataLoader then hands the storage over as is
        clip = torch.empty((len(rows), frames.shape[1], self.height, self.width, 3), dtype=torch.uint8).share_memory_()
        unique = np.unique(rows)   # sorted: neighbouring rows share Lance pages
        if self.ds is None:
            self.ds = lance.dataset(self.path)
        for start in range(0, len(unique), self.rows_per_take):
            chunk = unique[start:start + self.rows_per_take]
            table = self.ds.take(chunk.tolist(), columns=["pixels"])
            for row, raw in zip(chunk, _binary_views(table.column("pixels"))):
                pixels = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.num_frames[row], self.frame_height[row], self.frame_width[row], 3)
                for b in np.flatnonzero(rows == row):
                    clip[b] = resize_frames(pixels[frames[b]], self.height, self.width)
            del table   # release the chunk's Arrow buffers before the next take
        return {"clip": clip}


def check_native_store(ds):
    meta = (ds.schema.metadata or {}).get(NATIVE_METADATA_KEY)
    if meta is None:
        warnings.warn(f"{ds.uri} has no {NATIVE_METADATA_KEY.decode()} metadata; reading it as a native store anyway")
        return
    layout = json.loads(meta).get("layout")
    assert layout == NATIVE_LAYOUT_VERSION, f"{ds.uri}: native layout {layout}, this code reads {NATIVE_LAYOUT_VERSION}"


class NativeBatchSampler:
    """Training batches of a native store, as (B, 1 + T) requests.

    Rows come from one endless stream of permutations of the store, shared by all ranks (rank r takes every
    world_size-th slice), so every video is seen once per pass over the data whatever batches_per_epoch is, and an
    epoch's batches only depend on (seed, epoch, rank): resuming continues the same stream. Each row gives
    clips_per_video clips with their own frames (and their own crops on the GPU); reading a row is most of a worker's
    time, so this multiplies the clips a worker delivers, at the price of correlated clips in a batch.

    Like data.VariableLengthBatchSampler, the clip length T is drawn per batch from `probs` (the same on every rank).
    """

    def __init__(self, num_frames, batch_size, probs, batches_per_epoch, stride=1, sampling="stride",
                 clips_per_video=1, seed=0, rank=0, world_size=1):
        assert sampling in SAMPLINGS, f"sampling must be one of {SAMPLINGS}, got {sampling!r}"
        assert batch_size % clips_per_video == 0, "batch_size must be a multiple of clips_per_video"
        self.num_frames = np.asarray(num_frames)
        self.batch_size = batch_size
        self.rows_per_batch = batch_size // clips_per_video
        assert self.rows_per_batch <= len(self.num_frames), "fewer videos in the store than rows per batch"
        self.lengths = list(probs)
        self.probs = np.array([probs[length] for length in self.lengths], dtype=float)
        self.probs /= self.probs.sum()
        self.batches_per_epoch = batches_per_epoch
        self.stride, self.sampling, self.clips_per_video = stride, sampling, clips_per_video
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch = 0
        self._perm = (None, None)

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def sampler(self):
        # Lightning only calls set_epoch on `loader.sampler` / `loader.batch_sampler.sampler`
        return self

    def __len__(self):
        return self.batches_per_epoch

    def permutation(self, j):
        if self._perm[0] != j:
            self._perm = (j, np.random.default_rng((self.seed, j)).permutation(len(self.num_frames)))
        return self._perm[1]

    def stream(self, start, count):
        """Positions [start, start + count) of the endless row stream."""
        out = []
        while count > 0:
            j, offset = divmod(start, len(self.num_frames))
            take = self.permutation(j)[offset:offset + count]
            out.append(take)
            start, count = start + len(take), count - len(take)
        return np.concatenate(out)

    def __iter__(self):
        rng_length = np.random.default_rng((self.seed, self.epoch))
        rng = np.random.default_rng((self.seed, self.epoch, self.rank, 1))
        m = self.rows_per_batch
        for i in range(self.batches_per_epoch):
            length = int(rng_length.choice(self.lengths, p=self.probs))
            step = self.epoch * self.batches_per_epoch + i
            rows = np.repeat(self.stream((step * self.world_size + self.rank) * m, m), self.clips_per_video)
            frames = select_frames(self.num_frames[rows], length, self.stride, self.sampling, rng)
            yield np.concatenate([rows[:, None], frames], axis=1)


def native_val_batches(num_frames, videos, length, batch_size, stride=1, sampling="stride", seed=0, rank=0,
                       world_size=1):
    """This rank's fixed validation batches: `videos` rows drawn once (the same on every rank), the centred clip of
    each (rng=None in select_frames), as (B, 1 + T) requests. Every rank gets the same number of batches."""
    num_frames = np.asarray(num_frames)
    rows = np.sort(np.random.default_rng(seed).choice(len(num_frames), size=min(videos, len(num_frames)),
                                                      replace=False))
    requests = np.concatenate([rows[:, None], select_frames(num_frames[rows], length, stride, sampling)], axis=1)
    batches = [requests[i:i + batch_size] for i in range(0, len(requests) - batch_size + 1, batch_size)]
    per_rank = len(batches) // world_size
    assert per_rank > 0, f"{len(requests)} validation clips give fewer than one batch of {batch_size} per rank"
    return batches[rank * per_rank:(rank + 1) * per_rank]
