from collections import defaultdict

import numpy as np
import torch


class GaussianPostprocessor(object):
    def __init__(self, cfg):
        self._score_threshold = cfg["detector"]["postprocessor"]["score_threshold"]
        self._scales = cfg["detector"]["postprocessor"]["scales"]

    def _decode_single(self, hm):
        score = hm.max().item()
        if score < self._score_threshold:
            return [], []

        h, w = hm.shape
        device = hm.device
        weights = hm * (hm >= self._score_threshold)
        if weights.sum().item() <= 0:
            weights = hm

        gy, gx = torch.meshgrid(
            torch.arange(h, dtype=hm.dtype, device=device),
            torch.arange(w, dtype=hm.dtype, device=device),
            indexing="ij",
        )
        total = weights.sum() + 1e-6
        cx = (weights * gx).sum() / total
        cy = (weights * gy).sum() / total
        return [torch.stack([cx, cy])], [score]

    def run(self, preds, affine_mats):
        results = defaultdict(lambda: defaultdict(dict))
        for scale in self._scales:
            preds_ = preds[scale]
            affine_mats_ = affine_mats[scale]
            hms_ = preds_.sigmoid()

            b, s, _, _ = hms_.shape
            for i in range(b):
                affine_mat = affine_mats_[i].to(device=hms_.device, dtype=hms_.dtype)
                affine_mat_np = affine_mats_[i].detach().cpu().numpy()
                for j in range(s):
                    xys_, scores_ = self._decode_single(hms_[i, j])
                    xys_t_ = []
                    for xy_ in xys_:
                        xy_h = torch.cat([xy_, torch.ones(1, device=xy_.device, dtype=xy_.dtype)])
                        xy_t = torch.matmul(affine_mat, xy_h)
                        xys_t_.append(xy_t.detach().cpu().numpy())

                    results[i][j][scale] = {
                        "xys": xys_t_,
                        "scores": scores_,
                        "hm": hms_[i, j].detach().cpu().numpy(),
                        "trans": affine_mat_np,
                    }

        return results
