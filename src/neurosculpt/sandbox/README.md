# Sandbox

Planned (P0b, spec §6.1): one local container or WSL2 environment used identically by every LLM arm
and model. No network, read-only filesystem, CPU/time/memory limits, and a CPU-time budget on code
execution. It holds only the arm's presented weight view and exposes `evaluate`, `evaluate_batch` and
(where allowed) `forward` as RPCs to the host-side meter. The adversarial leakage test targets that RPC surface.
