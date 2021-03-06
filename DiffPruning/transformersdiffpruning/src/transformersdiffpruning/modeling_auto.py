# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Auto Model class. """


import logging
from collections import OrderedDict

from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    FlaubertConfig,
    GPT2Config,
    OpenAIGPTConfig,
    RobertaConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
)
from .configuration_utils import PretrainedConfig
from .modeling_albert import (
    ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    AlbertForMaskedLM,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertModel,
)
from .modeling_bart import BART_PRETRAINED_MODEL_ARCHIVE_MAP, BartForMaskedLM, BartForSequenceClassification, BartModel
from .modeling_bert import (
    BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    BertForMaskedLM,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
)
from .modeling_camembert import (
    CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    CamembertForMaskedLM,
    CamembertForSequenceClassification,
    CamembertForTokenClassification,
    CamembertModel,
)
from .modeling_ctrl import CTRL_PRETRAINED_MODEL_ARCHIVE_MAP, CTRLLMHeadModel, CTRLModel
from .modeling_distilbert import (
    DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    DistilBertForMaskedLM,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertModel,
)
from .modeling_flaubert import (
    FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    FlaubertForQuestionAnswering,
    FlaubertForSequenceClassification,
    FlaubertModel,
    FlaubertWithLMHeadModel,
)
from .modeling_gpt2 import GPT2_PRETRAINED_MODEL_ARCHIVE_MAP, GPT2LMHeadModel, GPT2Model
from .modeling_openai import OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP, OpenAIGPTLMHeadModel, OpenAIGPTModel
from .modeling_roberta import (
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    RobertaForMaskedLM,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from .modeling_t5 import T5_PRETRAINED_MODEL_ARCHIVE_MAP, T5Model, T5WithLMHeadModel
from .modeling_transfo_xl import TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP, TransfoXLLMHeadModel, TransfoXLModel
from .modeling_xlm import (
    XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLMForQuestionAnswering,
    XLMForSequenceClassification,
    XLMModel,
    XLMWithLMHeadModel,
)
from .modeling_xlm_roberta import (
    XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLMRobertaForMaskedLM,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaModel,
)
from .modeling_xlnet import (
    XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLNetForQuestionAnswering,
    XLNetForSequenceClassification,
    XLNetForTokenClassification,
    XLNetLMHeadModel,
    XLNetModel,
)


logger = logging.getLogger(__name__)


ALL_PRETRAINED_MODEL_ARCHIVE_MAP = dict(
    (key, value)
    for pretrained_map in [
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BART_PRETRAINED_MODEL_ARCHIVE_MAP,
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
        CTRL_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        T5_PRETRAINED_MODEL_ARCHIVE_MAP,
        FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    ]
    for key, value, in pretrained_map.items()
)

MODEL_MAPPING = OrderedDict(
    [
        (T5Config, T5Model),
        (DistilBertConfig, DistilBertModel),
        (AlbertConfig, AlbertModel),
        (CamembertConfig, CamembertModel),
        (XLMRobertaConfig, XLMRobertaModel),
        (BartConfig, BartModel),
        (RobertaConfig, RobertaModel),
        (BertConfig, BertModel),
        (OpenAIGPTConfig, OpenAIGPTModel),
        (GPT2Config, GPT2Model),
        (TransfoXLConfig, TransfoXLModel),
        (XLNetConfig, XLNetModel),
        (FlaubertConfig, FlaubertModel),
        (XLMConfig, XLMModel),
        (CTRLConfig, CTRLModel),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        (T5Config, T5WithLMHeadModel),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (BartConfig, BartForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (BertConfig, BertForPreTraining),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        (T5Config, T5WithLMHeadModel),
        (DistilBertConfig, DistilBertForMaskedLM),
        (AlbertConfig, AlbertForMaskedLM),
        (CamembertConfig, CamembertForMaskedLM),
        (XLMRobertaConfig, XLMRobertaForMaskedLM),
        (BartConfig, BartForMaskedLM),
        (RobertaConfig, RobertaForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        (GPT2Config, GPT2LMHeadModel),
        (TransfoXLConfig, TransfoXLLMHeadModel),
        (XLNetConfig, XLNetLMHeadModel),
        (FlaubertConfig, FlaubertWithLMHeadModel),
        (XLMConfig, XLMWithLMHeadModel),
        (CTRLConfig, CTRLLMHeadModel),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, DistilBertForSequenceClassification),
        (AlbertConfig, AlbertForSequenceClassification),
        (CamembertConfig, CamembertForSequenceClassification),
        (XLMRobertaConfig, XLMRobertaForSequenceClassification),
        (BartConfig, BartForSequenceClassification),
        (RobertaConfig, RobertaForSequenceClassification),
        (BertConfig, BertForSequenceClassification),
        (XLNetConfig, XLNetForSequenceClassification),
        (FlaubertConfig, FlaubertForSequenceClassification),
        (XLMConfig, XLMForSequenceClassification),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        (DistilBertConfig, DistilBertForQuestionAnswering),
        (AlbertConfig, AlbertForQuestionAnswering),
        (RobertaConfig, RobertaForQuestionAnswering),
        (BertConfig, BertForQuestionAnswering),
        (XLNetConfig, XLNetForQuestionAnswering),
        (FlaubertConfig, FlaubertForQuestionAnswering),
        (XLMConfig, XLMForQuestionAnswering),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (DistilBertConfig, DistilBertForTokenClassification),
        (CamembertConfig, CamembertForTokenClassification),
        (XLMRobertaConfig, XLMRobertaForTokenClassification),
        (RobertaConfig, RobertaForTokenClassification),
        (BertConfig, BertForTokenClassification),
        (XLNetConfig, XLNetForTokenClassification),
    ]
)


class AutoModel(object):
    r"""
        :class:`~transformersdiffpruning.AutoModel` is a generic model class
        that will be instantiated as one of the base model classes of the library
        when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        or the `AutoModel.from_config(config)` class methods.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModel.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Args:
            config (:class:`~transformersdiffpruning.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformersdiffpruning.DistilBertModel` (DistilBERT model)
                - isInstance of `roberta` configuration class: :class:`~transformersdiffpruning.RobertaModel` (RoBERTa model)
                - isInstance of `bert` configuration class: :class:`~transformersdiffpruning.BertModel` (Bert model)
                - isInstance of `openai-gpt` configuration class: :class:`~transformersdiffpruning.OpenAIGPTModel` (OpenAI GPT model)
                - isInstance of `gpt2` configuration class: :class:`~transformersdiffpruning.GPT2Model` (OpenAI GPT-2 model)
                - isInstance of `ctrl` configuration class: :class:`~transformersdiffpruning.CTRLModel` (Salesforce CTRL  model)
                - isInstance of `transfo-xl` configuration class: :class:`~transformersdiffpruning.TransfoXLModel` (Transformer-XL model)
                - isInstance of `xlnet` configuration class: :class:`~transformersdiffpruning.XLNetModel` (XLNet model)
                - isInstance of `xlm` configuration class: :class:`~transformersdiffpruning.XLMModel` (XLM model)
                - isInstance of `flaubert` configuration class: :class:`~transformersdiffpruning.FlaubertModel` (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModel.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the base model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: :class:`~transformersdiffpruning.T5Model` (T5 model)
            - contains `distilbert`: :class:`~transformersdiffpruning.DistilBertModel` (DistilBERT model)
            - contains `albert`: :class:`~transformersdiffpruning.AlbertModel` (ALBERT model)
            - contains `camembert`: :class:`~transformersdiffpruning.CamembertModel` (CamemBERT model)
            - contains `xlm-roberta`: :class:`~transformersdiffpruning.XLMRobertaModel` (XLM-RoBERTa model)
            - contains `roberta`: :class:`~transformersdiffpruning.RobertaModel` (RoBERTa model)
            - contains `bert`: :class:`~transformersdiffpruning.BertModel` (Bert model)
            - contains `openai-gpt`: :class:`~transformersdiffpruning.OpenAIGPTModel` (OpenAI GPT model)
            - contains `gpt2`: :class:`~transformersdiffpruning.GPT2Model` (OpenAI GPT-2 model)
            - contains `transfo-xl`: :class:`~transformersdiffpruning.TransfoXLModel` (Transformer-XL model)
            - contains `xlnet`: :class:`~transformersdiffpruning.XLNetModel` (XLNet model)
            - contains `xlm`: :class:`~transformersdiffpruning.XLMModel` (XLM model)
            - contains `ctrl`: :class:`~transformersdiffpruning.CTRLModel` (Salesforce CTRL  model)
            - contains `flaubert`: :class:`~transformersdiffpruning.Flaubert` (Flaubert  model)

            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
            To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformersdiffpruning.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and :func:`~transformersdiffpruning.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = AutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModel.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_MAPPING.keys())
            )
        )


class AutoModelForPreTraining(object):
    r"""
        :class:`~transformersdiffpruning.AutoModelForPreTraining` is a generic model class
        that will be instantiated as one of the model classes of the library -with the architecture used for pretraining this model??? when created with the `AutoModelForPreTraining.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForPreTraining is designed to be instantiated "
            "using the `AutoModelForPreTraining.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForPreTraining.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Args:
            config (:class:`~transformersdiffpruning.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformersdiffpruning.DistilBertModelForMaskedLM` (DistilBERT model)
                - isInstance of `roberta` configuration class: :class:`~transformersdiffpruning.RobertaModelForMaskedLM` (RoBERTa model)
                - isInstance of `bert` configuration class: :class:`~transformersdiffpruning.BertForPreTraining` (Bert model)
                - isInstance of `openai-gpt` configuration class: :class:`~transformersdiffpruning.OpenAIGPTLMHeadModel` (OpenAI GPT model)
                - isInstance of `gpt2` configuration class: :class:`~transformersdiffpruning.GPT2ModelLMHeadModel` (OpenAI GPT-2 model)
                - isInstance of `ctrl` configuration class: :class:`~transformersdiffpruning.CTRLModelLMHeadModel` (Salesforce CTRL  model)
                - isInstance of `transfo-xl` configuration class: :class:`~transformersdiffpruning.TransfoXLLMHeadModel` (Transformer-XL model)
                - isInstance of `xlnet` configuration class: :class:`~transformersdiffpruning.XLNetLMHeadModel` (XLNet model)
                - isInstance of `xlm` configuration class: :class:`~transformersdiffpruning.XLMWithLMHeadModel` (XLM model)
                - isInstance of `flaubert` configuration class: :class:`~transformersdiffpruning.FlaubertWithLMHeadModel` (Flaubert model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForPreTraining.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_PRETRAINING_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the model classes of the library -with the architecture used for pretraining this model??? from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: :class:`~transformersdiffpruning.T5ModelWithLMHead` (T5 model)
            - contains `distilbert`: :class:`~transformersdiffpruning.DistilBertForMaskedLM` (DistilBERT model)
            - contains `albert`: :class:`~transformersdiffpruning.AlbertForMaskedLM` (ALBERT model)
            - contains `camembert`: :class:`~transformersdiffpruning.CamembertForMaskedLM` (CamemBERT model)
            - contains `xlm-roberta`: :class:`~transformersdiffpruning.XLMRobertaForMaskedLM` (XLM-RoBERTa model)
            - contains `roberta`: :class:`~transformersdiffpruning.RobertaForMaskedLM` (RoBERTa model)
            - contains `bert`: :class:`~transformersdiffpruning.BertForPreTraining` (Bert model)
            - contains `openai-gpt`: :class:`~transformersdiffpruning.OpenAIGPTLMHeadModel` (OpenAI GPT model)
            - contains `gpt2`: :class:`~transformersdiffpruning.GPT2LMHeadModel` (OpenAI GPT-2 model)
            - contains `transfo-xl`: :class:`~transformersdiffpruning.TransfoXLLMHeadModel` (Transformer-XL model)
            - contains `xlnet`: :class:`~transformersdiffpruning.XLNetLMHeadModel` (XLNet model)
            - contains `xlm`: :class:`~transformersdiffpruning.XLMWithLMHeadModel` (XLM model)
            - contains `ctrl`: :class:`~transformersdiffpruning.CTRLLMHeadModel` (Salesforce CTRL model)
            - contains `flaubert`: :class:`~transformersdiffpruning.FlaubertWithLMHeadModel` (Flaubert model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~transformersdiffpruning.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and :func:`~transformersdiffpruning.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.
            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely received file. Attempt to resume the download if such a file exists.
            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = AutoModelForPreTraining.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForPreTraining.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForPreTraining.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_PRETRAINING_MAPPING.keys())
            )
        )


class AutoModelWithLMHead(object):
    r"""
        :class:`~transformersdiffpruning.AutoModelWithLMHead` is a generic model class
        that will be instantiated as one of the language modeling model classes of the library
        when created with the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelWithLMHead is designed to be instantiated "
            "using the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelWithLMHead.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Args:
            config (:class:`~transformersdiffpruning.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformersdiffpruning.DistilBertModelForMaskedLM` (DistilBERT model)
                - isInstance of `roberta` configuration class: :class:`~transformersdiffpruning.RobertaModelForMaskedLM` (RoBERTa model)
                - isInstance of `bert` configuration class: :class:`~transformersdiffpruning.BertModelForMaskedLM` (Bert model)
                - isInstance of `openai-gpt` configuration class: :class:`~transformersdiffpruning.OpenAIGPTLMHeadModel` (OpenAI GPT model)
                - isInstance of `gpt2` configuration class: :class:`~transformersdiffpruning.GPT2ModelLMHeadModel` (OpenAI GPT-2 model)
                - isInstance of `ctrl` configuration class: :class:`~transformersdiffpruning.CTRLModelLMHeadModel` (Salesforce CTRL  model)
                - isInstance of `transfo-xl` configuration class: :class:`~transformersdiffpruning.TransfoXLLMHeadModel` (Transformer-XL model)
                - isInstance of `xlnet` configuration class: :class:`~transformersdiffpruning.XLNetLMHeadModel` (XLNet model)
                - isInstance of `xlm` configuration class: :class:`~transformersdiffpruning.XLMWithLMHeadModel` (XLM model)
                - isInstance of `flaubert` configuration class: :class:`~transformersdiffpruning.FlaubertWithLMHeadModel` (Flaubert model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelWithLMHead.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_WITH_LM_HEAD_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_WITH_LM_HEAD_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the language modeling model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: :class:`~transformersdiffpruning.T5ModelWithLMHead` (T5 model)
            - contains `distilbert`: :class:`~transformersdiffpruning.DistilBertForMaskedLM` (DistilBERT model)
            - contains `albert`: :class:`~transformersdiffpruning.AlbertForMaskedLM` (ALBERT model)
            - contains `camembert`: :class:`~transformersdiffpruning.CamembertForMaskedLM` (CamemBERT model)
            - contains `xlm-roberta`: :class:`~transformersdiffpruning.XLMRobertaForMaskedLM` (XLM-RoBERTa model)
            - contains `roberta`: :class:`~transformersdiffpruning.RobertaForMaskedLM` (RoBERTa model)
            - contains `bert`: :class:`~transformersdiffpruning.BertForMaskedLM` (Bert model)
            - contains `openai-gpt`: :class:`~transformersdiffpruning.OpenAIGPTLMHeadModel` (OpenAI GPT model)
            - contains `gpt2`: :class:`~transformersdiffpruning.GPT2LMHeadModel` (OpenAI GPT-2 model)
            - contains `transfo-xl`: :class:`~transformersdiffpruning.TransfoXLLMHeadModel` (Transformer-XL model)
            - contains `xlnet`: :class:`~transformersdiffpruning.XLNetLMHeadModel` (XLNet model)
            - contains `xlm`: :class:`~transformersdiffpruning.XLMWithLMHeadModel` (XLM model)
            - contains `ctrl`: :class:`~transformersdiffpruning.CTRLLMHeadModel` (Salesforce CTRL model)
            - contains `flaubert`: :class:`~transformersdiffpruning.FlaubertWithLMHeadModel` (Flaubert model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method
            config: (`optional`) instance of a class derived from :class:`~transformersdiffpruning.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and :func:`~transformersdiffpruning.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.
            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely received file. Attempt to resume the download if such a file exists.
            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = AutoModelWithLMHead.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelWithLMHead.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelWithLMHead.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_WITH_LM_HEAD_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_WITH_LM_HEAD_MAPPING.keys())
            )
        )


class AutoModelForSequenceClassification(object):
    r"""
        :class:`~transformersdiffpruning.AutoModelForSequenceClassification` is a generic model class
        that will be instantiated as one of the sequence classification model classes of the library
        when created with the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForSequenceClassification is designed to be instantiated "
            "using the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForSequenceClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Args:
            config (:class:`~transformersdiffpruning.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformersdiffpruning.DistilBertModelForSequenceClassification` (DistilBERT model)
                - isInstance of `albert` configuration class: :class:`~transformersdiffpruning.AlbertModelForSequenceClassification` (ALBERT model)
                - isInstance of `camembert` configuration class: :class:`~transformersdiffpruning.CamembertModelForSequenceClassification` (CamemBERT model)
                - isInstance of `xlm roberta` configuration class: :class:`~transformersdiffpruning.XLMRobertaModelForSequenceClassification` (XLM-RoBERTa model)
                - isInstance of `roberta` configuration class: :class:`~transformersdiffpruning.RobertaModelForSequenceClassification` (RoBERTa model)
                - isInstance of `bert` configuration class: :class:`~transformersdiffpruning.BertModelForSequenceClassification` (Bert model)
                - isInstance of `xlnet` configuration class: :class:`~transformersdiffpruning.XLNetModelForSequenceClassification` (XLNet model)
                - isInstance of `xlm` configuration class: :class:`~transformersdiffpruning.XLMModelForSequenceClassification` (XLM model)
                - isInstance of `flaubert` configuration class: :class:`~transformersdiffpruning.FlaubertForSequenceClassification` (Flaubert model)


        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: :class:`~transformersdiffpruning.DistilBertForSequenceClassification` (DistilBERT model)
            - contains `albert`: :class:`~transformersdiffpruning.AlbertForSequenceClassification` (ALBERT model)
            - contains `camembert`: :class:`~transformersdiffpruning.CamembertForSequenceClassification` (CamemBERT model)
            - contains `xlm-roberta`: :class:`~transformersdiffpruning.XLMRobertaForSequenceClassification` (XLM-RoBERTa model)
            - contains `roberta`: :class:`~transformersdiffpruning.RobertaForSequenceClassification` (RoBERTa model)
            - contains `bert`: :class:`~transformersdiffpruning.BertForSequenceClassification` (Bert model)
            - contains `xlnet`: :class:`~transformersdiffpruning.XLNetForSequenceClassification` (XLNet model)
            - contains `flaubert`: :class:`~transformersdiffpruning.FlaubertForSequenceClassification` (Flaubert model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaining positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformersdiffpruning.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and :func:`~transformersdiffpruning.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForSequenceClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )


class AutoModelForQuestionAnswering(object):
    r"""
        :class:`~transformersdiffpruning.AutoModelForQuestionAnswering` is a generic model class
        that will be instantiated as one of the question answering model classes of the library
        when created with the `AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForQuestionAnswering is designed to be instantiated "
            "using the `AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForQuestionAnswering.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Args:
            config (:class:`~transformersdiffpruning.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformersdiffpruning.DistilBertModelForQuestionAnswering` (DistilBERT model)
                - isInstance of `albert` configuration class: :class:`~transformersdiffpruning.AlbertModelForQuestionAnswering` (ALBERT model)
                - isInstance of `bert` configuration class: :class:`~transformersdiffpruning.BertModelForQuestionAnswering` (Bert model)
                - isInstance of `xlnet` configuration class: :class:`~transformersdiffpruning.XLNetModelForQuestionAnswering` (XLNet model)
                - isInstance of `xlm` configuration class: :class:`~transformersdiffpruning.XLMModelForQuestionAnswering` (XLM model)
                - isInstance of `flaubert` configuration class: :class:`~transformersdiffpruning.FlaubertForQuestionAnswering` (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_QUESTION_ANSWERING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: :class:`~transformersdiffpruning.DistilBertForQuestionAnswering` (DistilBERT model)
            - contains `albert`: :class:`~transformersdiffpruning.AlbertForQuestionAnswering` (ALBERT model)
            - contains `bert`: :class:`~transformersdiffpruning.BertForQuestionAnswering` (Bert model)
            - contains `xlnet`: :class:`~transformersdiffpruning.XLNetForQuestionAnswering` (XLNet model)
            - contains `xlm`: :class:`~transformersdiffpruning.XLMForQuestionAnswering` (XLM model)
            - contains `flaubert`: :class:`~transformersdiffpruning.FlaubertForQuestionAnswering` (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformersdiffpruning.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and :func:`~transformersdiffpruning.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForQuestionAnswering.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForQuestionAnswering.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_QUESTION_ANSWERING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys()),
            )
        )


class AutoModelForTokenClassification:
    r"""
        :class:`~transformersdiffpruning.AutoModelForTokenClassification` is a generic model class
        that will be instantiated as one of the token classification model classes of the library
        when created with the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Args:
            config (:class:`~transformersdiffpruning.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformersdiffpruning.DistilBertModelForTokenClassification` (DistilBERT model)
                - isInstance of `xlm roberta` configuration class: :class:`~transformersdiffpruning.XLMRobertaModelForTokenClassification` (XLMRoberta model)
                - isInstance of `bert` configuration class: :class:`~transformersdiffpruning.BertModelForTokenClassification` (Bert model)
                - isInstance of `xlnet` configuration class: :class:`~transformersdiffpruning.XLNetModelForTokenClassification` (XLNet model)
                - isInstance of `camembert` configuration class: :class:`~transformersdiffpruning.CamembertModelForTokenClassification` (Camembert model)
                - isInstance of `roberta` configuration class: :class:`~transformersdiffpruning.RobertaModelForTokenClassification` (Roberta model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForTokenClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: :class:`~transformersdiffpruning.DistilBertForTokenClassification` (DistilBERT model)
            - contains `xlm-roberta`: :class:`~transformersdiffpruning.XLMRobertaForTokenClassification` (XLM-RoBERTa?Para model)
            - contains `camembert`: :class:`~transformersdiffpruning.CamembertForTokenClassification` (Camembert model)
            - contains `bert`: :class:`~transformersdiffpruning.BertForTokenClassification` (Bert model)
            - contains `xlnet`: :class:`~transformersdiffpruning.XLNetForTokenClassification` (XLNet model)
            - contains `roberta`: :class:`~transformersdiffpruning.RobertaForTokenClassification` (Roberta model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path:
                Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformersdiffpruning.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformersdiffpruning.PreTrainedModel.save_pretrained` and :func:`~transformersdiffpruning.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForTokenClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForTokenClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )
