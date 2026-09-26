"""Q-WAM: 4-bit post-training quantization of World Action Models.

Shared, model-agnostic pieces live directly in this package:

  qwam.quant           symmetric group-wise fake quantization and the SmoothQuant factor
  qwam.hadamard        block-diagonal orthonormal Hadamard rotation (fast Walsh-Hadamard)
  qwam.robotwin_calib  the RoboTwin 2.0 calibration set shared by all models

Model integrations live in qwam.fastwam, qwam.imagewam and qwam.lingbot_va.
"""
