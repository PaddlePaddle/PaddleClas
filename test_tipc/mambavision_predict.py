#!/usr/bin/env python3
"""Run PaddleClas deployment with Paddle 2.6 legacy-model compatibility."""

from __future__ import annotations

import os
from pathlib import Path
import runpy

from paddle.inference import Config as PaddleInferenceConfig
import paddleclas.deploy.utils.predictor as predictor_module


class LegacyAwareConfig:
    """Select the file-based constructor for legacy pdmodel exports."""

    Precision = PaddleInferenceConfig.Precision

    def __new__(cls, model_dir, model_prefix):
        model_file = os.path.join(model_dir, f"{model_prefix}.pdmodel")
        params_file = os.path.join(model_dir, f"{model_prefix}.pdiparams")
        if os.path.isfile(model_file) and os.path.isfile(params_file):
            return PaddleInferenceConfig(model_file, params_file)
        return PaddleInferenceConfig(model_dir, model_prefix)


predictor_module.Config = LegacyAwareConfig
repo_root = Path(__file__).resolve().parents[1]
runpy.run_path(str(repo_root / "deploy" / "python" / "predict_cls.py"),
               run_name="__main__")
