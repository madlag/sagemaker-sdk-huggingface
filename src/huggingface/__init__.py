from __future__ import absolute_import

# from sagemaker.pytorch.estimator import HuggingFace  # noqa: F401
from huggingface.estimator import HuggingFace
from sagemaker.pytorch.model import PyTorchModel as HuggingFaceModel  # noqa: F401

# from datasets import *
# import transformers
