# -*- coding: utf-8 -*-
"""Memory-conscious pickle I/O used by the DESI inference pipeline.

DESI preprocessing jobs in the original workspace store either one large
top-level list or several consecutively pickled list chunks.  The custom
unpickler below yields rows from both layouts without requiring the original
``desi_recall/recall.py`` module at runtime.
"""

from __future__ import annotations

import pickle
import struct
import types
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np


_PRUNED_PICKLE_MEMO = object()


def _should_keep_pickle_memo_value(value: Any, root_list: Optional[list], depth: int = 0) -> bool:
    if value is root_list or value is _PRUNED_PICKLE_MEMO:
        return True
    if isinstance(value, (str, int, float, complex, bool, type(None), np.generic)):
        return True
    if isinstance(value, bytes):
        return len(value) <= 1024
    if isinstance(
        value,
        (
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            types.FunctionType,
            type,
        ),
    ):
        return True
    if isinstance(value, np.dtype):
        return True
    if isinstance(value, tuple):
        if depth >= 2 or len(value) > 8:
            return False
        return all(_should_keep_pickle_memo_value(item, root_list, depth + 1) for item in value)
    return False


class _PauseTopLevelListLoad(Exception):
    pass


class TopLevelListStreamingUnpickler(pickle._Unpickler):
    """Incrementally decode a pickle whose root object is a Python list."""

    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, file_obj):
        super().__init__(file_obj)
        self.root_list: Optional[list] = None
        self.done = False
        self._runtime_ready = False
        self._chunk_buffer: list[Any] = []
        self._next_memo_scan = 0

    def _ensure_runtime(self) -> None:
        if self._runtime_ready:
            return
        self._unframer = pickle._Unframer(self._file_read, self._file_readline)
        self.read = self._unframer.read
        self.readinto = self._unframer.readinto
        self.readline = self._unframer.readline
        self.metastack = []
        self.stack = []
        self.append = self.stack.append
        self.proto = 0
        self._runtime_ready = True

    def _mark_root_list_if_needed(self, obj: Any) -> None:
        if self.root_list is None and not self.metastack and len(self.stack) == 1 and self.stack[0] is obj:
            self.root_list = obj

    def _prune_pickle_memo(self) -> None:
        max_key = max(self.memo.keys(), default=-1)
        if max_key < self._next_memo_scan:
            return
        for key in range(self._next_memo_scan, max_key + 1):
            if key not in self.memo:
                continue
            value = self.memo[key]
            if not _should_keep_pickle_memo_value(value, self.root_list):
                self.memo[key] = _PRUNED_PICKLE_MEMO
        self._next_memo_scan = max_key + 1

    def _buffer_root_rows(self, rows: list[Any]) -> None:
        if not rows:
            return
        self._chunk_buffer.extend(rows)
        self._prune_pickle_memo()
        raise _PauseTopLevelListLoad

    def load_empty_list(self):
        value = []
        self.append(value)
        self._mark_root_list_if_needed(value)

    def load_list(self):
        value = self.pop_mark()
        self.append(value)
        self._mark_root_list_if_needed(value)

    def load_append(self):
        value = self.stack.pop()
        list_obj = self.stack[-1]
        if list_obj is self.root_list:
            self._buffer_root_rows([value])
        else:
            list_obj.append(value)

    def load_appends(self):
        values = self.pop_mark()
        list_obj = self.stack[-1]
        if list_obj is self.root_list:
            self._buffer_root_rows(values)
            return
        try:
            list_obj.extend(values)
        except AttributeError:
            for value in values:
                list_obj.append(value)

    def load_get(self):
        key = int(self.readline()[:-1])
        value = self.memo[key]
        if value is _PRUNED_PICKLE_MEMO:
            raise pickle.UnpicklingError(f"pickle memo entry {key} was pruned too early")
        self.append(value)

    def load_binget(self):
        key = self.read(1)[0]
        value = self.memo[key]
        if value is _PRUNED_PICKLE_MEMO:
            raise pickle.UnpicklingError(f"pickle memo entry {key} was pruned too early")
        self.append(value)

    def load_long_binget(self):
        (key,) = struct.unpack("<I", self.read(4))
        value = self.memo.get(key, _PRUNED_PICKLE_MEMO)
        if value is _PRUNED_PICKLE_MEMO:
            raise pickle.UnpicklingError(f"pickle memo entry {key} was pruned too early")
        self.append(value)

    def load_next_chunk(self) -> Optional[list[Any]]:
        if self.done:
            return None
        self._ensure_runtime()
        self._chunk_buffer = []
        try:
            while True:
                key = self.read(1)
                if not key:
                    raise EOFError
                self.dispatch[key[0]](self)
        except _PauseTopLevelListLoad:
            chunk = self._chunk_buffer
            self._chunk_buffer = []
            return chunk
        except pickle._Stop as stop:
            self.done = True
            chunk = self._chunk_buffer
            self._chunk_buffer = []
            if chunk:
                return chunk
            if self.root_list is None:
                return [stop.value]
            return None


TopLevelListStreamingUnpickler.dispatch[pickle.EMPTY_LIST[0]] = TopLevelListStreamingUnpickler.load_empty_list
TopLevelListStreamingUnpickler.dispatch[pickle.LIST[0]] = TopLevelListStreamingUnpickler.load_list
TopLevelListStreamingUnpickler.dispatch[pickle.APPEND[0]] = TopLevelListStreamingUnpickler.load_append
TopLevelListStreamingUnpickler.dispatch[pickle.APPENDS[0]] = TopLevelListStreamingUnpickler.load_appends
TopLevelListStreamingUnpickler.dispatch[pickle.GET[0]] = TopLevelListStreamingUnpickler.load_get
TopLevelListStreamingUnpickler.dispatch[pickle.BINGET[0]] = TopLevelListStreamingUnpickler.load_binget
TopLevelListStreamingUnpickler.dispatch[pickle.LONG_BINGET[0]] = TopLevelListStreamingUnpickler.load_long_binget


def _yield_rows_from_loaded_object(obj: Any) -> Iterator[Any]:
    if isinstance(obj, list):
        for index in range(len(obj)):
            yield obj[index]
            obj[index] = None
        return
    yield obj


def iter_rows_from_pickle(filepath: str | Path) -> Iterator[Any]:
    """Yield rows from a large-list or concatenated-object pickle file."""

    with Path(filepath).open("rb") as handle:
        while True:
            try:
                unpickler = TopLevelListStreamingUnpickler(handle)
                first_chunk = unpickler.load_next_chunk()
            except EOFError:
                break

            if unpickler.root_list is not None:
                chunk = first_chunk
                while chunk is not None:
                    for index in range(len(chunk)):
                        yield chunk[index]
                        chunk[index] = None
                    chunk = unpickler.load_next_chunk()
                continue

            if first_chunk is None:
                continue
            for obj in first_chunk:
                yield from _yield_rows_from_loaded_object(obj)


class PickleStreamWriter:
    """Write consecutive pickle chunks that ``iter_rows_from_pickle`` can read."""

    def __init__(self, output_path: str | Path, overwrite: bool = False):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {self.output_path}. Pass --overwrite to replace it."
            )
        self._mode = "wb"
        self.rows_written = 0

    def write_rows(self, rows: list[Any]) -> None:
        if not rows:
            return
        with self.output_path.open(self._mode) as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._mode = "ab"
        self.rows_written += len(rows)


__all__ = [
    "PickleStreamWriter",
    "TopLevelListStreamingUnpickler",
    "iter_rows_from_pickle",
]
