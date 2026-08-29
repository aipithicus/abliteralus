"""Transactional, append-oriented JSONL stores."""

from .canonical import canonical_bytes, decode_record, encode_record
from .engine import (
    TRANSACTION_SCHEMA_ID,
    CommitReceipt,
    JsonlStore,
    JsonlTransaction,
    PartitionRange,
    StorePrefixScan,
    StoreRepairReceipt,
    TransactionState,
)
from .errors import (
    JsonlCorruptionError,
    JsonlFormatError,
    JsonlStoreError,
    UncommittedTailError,
)
from .index import JSOI_VERSION, Jidx, jidx_path_for, read_index

__version__ = "0.1.0"

__all__ = [
    "CommitReceipt",
    "JsonlCorruptionError",
    "JsonlFormatError",
    "JsonlStore",
    "JsonlStoreError",
    "JsonlTransaction",
    "Jidx",
    "JSOI_VERSION",
    "PartitionRange",
    "StorePrefixScan",
    "StoreRepairReceipt",
    "TRANSACTION_SCHEMA_ID",
    "TransactionState",
    "UncommittedTailError",
    "canonical_bytes",
    "decode_record",
    "encode_record",
    "jidx_path_for",
    "read_index",
]
