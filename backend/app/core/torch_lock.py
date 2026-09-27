"""One process-wide lock for every PyTorch forward pass.

Apple's MPS backend is not safe under concurrent use. Two threads running
inference at the same time - even on *different* models - trip a Metal
assertion and abort the process:

    failed assertion _status < MTLCommandBufferStatusCommitted

Per-model locks are not enough, because the conflict is at the device level:
the embedding bi-encoder and the re-ranking cross-encoder share one GPU. So the
lock has to be global, and every forward pass in the app acquires it.

This costs less than it sounds like. The GPU executes one kernel at a time
regardless; serialising at the Python level just makes that explicit. All the
genuinely concurrent work in a request - API calls, fusion, SQLite writes -
happens outside this lock and still overlaps freely.

CUDA does tolerate concurrent inference, so the lock is a no-op contention-wise
on an NVIDIA box; it is retained unconditionally because correctness on the
developer's laptop is worth more than the nanoseconds it costs elsewhere.
"""
from __future__ import annotations

import threading

# Re-entrant so a future nested call (an encode inside a re-rank, say) cannot
# deadlock against itself.
TORCH_INFERENCE_LOCK = threading.RLock()
