"""Chronos-Bolt, used zero-shot (no training on M5)."""
import os

import numpy as np

_PIPES = {}


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
    Qs, Ms, bs = [], [], cfg["batch_size"]
    with torch.inference_mode():
        for b in range(0, len(ctx), bs):
            q, m = pipe.predict_quantiles(ctx[b:b + bs], prediction_length=h, quantile_levels=list(qs))
            Qs.append(q.float().cpu().numpy())
            Ms.append(m.float().cpu().numpy())
    Q = np.clip(np.sort(np.concatenate(Qs), axis=2), 0, None)
    return {"point": np.clip(np.concatenate(Ms), 0, None), "Q": Q}
