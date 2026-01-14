# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# -------------------- Normals Metrics --------------------


def compute_cosine_error(pred_norm, gt_norm, masked=False):
    if len(pred_norm.shape) == 4:
        pred_norm = pred_norm.squeeze(0)
    if len(gt_norm.shape) == 4:
        gt_norm = gt_norm.squeeze(0)

    # shape must be [3,H,W]
    assert (gt_norm.shape[0] == 3) and (
        pred_norm.shape[0] == 3
    ), "Channel dim should be the first dimension!"
    # mask out the zero vectors, otherwise torch.cosine_similarity computes 90° as error
    if masked:
        ch, h, w = gt_norm.shape

        mask = torch.norm(gt_norm, dim=0) > 0

        pred_norm = pred_norm[:, mask.view(h, w)]
        gt_norm = gt_norm[:, mask.view(h, w)]

    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=0)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi  # (H, W)

    return (
        pred_error.view(-1).detach().cpu().numpy()
    )  # flatten so can directly input to compute_normal_metrics()


def mean_angular_error(cosine_error):
    return round(np.average(cosine_error), 4)


def median_angular_error(cosine_error):
    return round(np.median(cosine_error), 4)


def rmse_angular_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(np.sqrt(np.sum(cosine_error * cosine_error) / num_pixels), 4)


def sub5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 5) / num_pixels), 4)


def sub7_5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 7.5) / num_pixels), 4)


def sub11_25_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 11.25) / num_pixels), 4)


def sub22_5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 22.5) / num_pixels), 4)


def sub30_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 30) / num_pixels), 4)
