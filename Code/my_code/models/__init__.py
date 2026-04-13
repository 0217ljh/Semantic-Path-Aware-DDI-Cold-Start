# Model implementations by name: my_code/models/<name>/ (model.py, ...).
# Import all method subpackages so they register with pipeline; add new methods here.
from __future__ import annotations

from my_code.pipeline.registry import register
from my_code.models.GCN import GCNMethod
from my_code.models.mnist_cnn import CNNMethod

register("main_GCN")(GCNMethod)
register("main_CNN")(CNNMethod)