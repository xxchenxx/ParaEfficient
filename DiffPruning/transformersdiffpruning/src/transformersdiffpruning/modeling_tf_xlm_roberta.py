# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" TF 2.0  XLM-RoBERTa model. """


import logging

from .configuration_xlm_roberta import XLMRobertaConfig
from .file_utils import add_start_docstrings
from .modeling_tf_roberta import (
    TFRobertaForMaskedLM,
    TFRobertaForSequenceClassification,
    TFRobertaForTokenClassification,
    TFRobertaModel,
)


logger = logging.getLogger(__name__)

TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {}


XLM_ROBERTA_START_DOCSTRING = r"""

    .. note::

        TF 2.0 models accepts two formats as inputs:

            - having all inputs as keyword arguments (like PyTorch models), or
            - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :obj:`tf.keras.Model.fit()` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with input_ids only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({'input_ids': input_ids, 'token_type_ids': token_type_ids})`

    Parameters:
        config (:class:`~transformersdiffpruning.XLMRobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformersdiffpruning.PreTrainedModel.from_pretrained` method to load the model weights.
"""


@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
class TFXLMRobertaModel(TFRobertaModel):
    """
    This class overrides :class:`~transformersdiffpruning.TFRobertaModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model with a `language modeling` head on top. """, XLM_ROBERTA_START_DOCSTRING,
)
class TFXLMRobertaForMaskedLM(TFRobertaForMaskedLM):
    """
    This class overrides :class:`~transformersdiffpruning.TFRobertaForMaskedLM`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class TFXLMRobertaForSequenceClassification(TFRobertaForSequenceClassification):
    """
    This class overrides :class:`~transformersdiffpruning.TFRobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class TFXLMRobertaForTokenClassification(TFRobertaForTokenClassification):
    """
    This class overrides :class:`~transformersdiffpruning.TFRobertaForTokenClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
