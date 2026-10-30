"""Chronos-Bolt, used zero-shot (no training on M5)."""
import os

import numpy as np

_PIPES = {}
GRID = [round(0.1 * k, 1) for k in range(1, 10)]   # the quantiles Chronos-Bolt was trained on


def load(cfg):
    key = (cfg["model_id"], cfg["device"])
    if key not in _PIPES:
        import torch
        from chronos import BaseChronosPipeline
        if cfg["device"] == "cpu":
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
        _PIPES[key] = BaseChronosPipeline.from_pretrained(cfg["model_id"], device_map=cfg["device"],
                                                          torch_dtype=torch.float32)
    return _PIPES[key]


def forecast(data, idx, origin, h, qs, cfg):
    import torch
    pipe = load(cfg)
    ctx = [torch.tensor(np.nan_to_num(data.Y[i, max(data.first_sale[i] - 1, origin - cfg["context_days"]):origin]),
                        dtype=torch.float32) for i in idx]
    Qs, bs = [], cfg["batch_size"]
    with torch.inference_mode():
        for b in range(0, len(ctx), bs):
            # Bolt's "mean" output is really its median, which biases sums low for skewed counts.
            # The point forecast is the average of its 9 trained quantiles instead.
            q, _ = pipe.predict_quantiles(ctx[b:b + bs], prediction_length=h, quantile_levels=GRID)
            Qs.append(q.float().cpu().numpy())
    A = np.clip(np.concatenate(Qs), 0, None)
    Q = np.sort(A[..., [GRID.index(round(x, 1)) for x in qs]], axis=2)
    return {"point": A.mean(axis=2), "Q": Q}
