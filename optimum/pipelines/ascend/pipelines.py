"""Pipelines running different backends."""

from typing import Any, Dict, Optional, Union

from transformers import (AutoConfig, FeatureExtractionPipeline,
                          FillMaskPipeline, Pipeline, PreTrainedTokenizer,
                          PreTrainedTokenizerFast, QuestionAnsweringPipeline,
                          SequenceFeatureExtractor, SummarizationPipeline,
                          Text2TextGenerationPipeline,
                          TextClassificationPipeline, TextGenerationPipeline,
                          TokenClassificationPipeline, TranslationPipeline,
                          ZeroShotClassificationPipeline)
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.onnx.utils import get_preprocessor
from transformers.pipelines import \
    SUPPORTED_TASKS as TRANSFORMERS_SUPPORTED_TASKS

from ...ascend.utils import is_acl_available

if is_acl_available():
    from ...ascend import (ORTModelForCausalLM, ORTModelForFeatureExtraction,
                           ORTModelForMaskedLM, ORTModelForQuestionAnswering,
                           ORTModelForSeq2SeqLM,
                           ORTModelForSequenceClassification,
                           ORTModelForTokenClassification)
    from ...ascend.modeling_base import AscendModel

    ORT_SUPPORTED_TASKS = {
        "feature-extraction": {
            "impl": FeatureExtractionPipeline,
            "class": (ORTModelForFeatureExtraction,),
            "default": "distilbert-base-cased",
            "type": "text",  # feature extraction is only supported for text at the moment
        },
        "fill-mask": {
            "impl": FillMaskPipeline,
            "class": (ORTModelForMaskedLM,),
            "default": "bert-base-cased",
            "type": "text",
        },
        "question-answering": {
            "impl": QuestionAnsweringPipeline,
            "class": (ORTModelForQuestionAnswering,),
            "default": "distilbert-base-cased-distilled-squad",
            "type": "text",
        },
        "text-classification": {
            "impl": TextClassificationPipeline,
            "class": (ORTModelForSequenceClassification,),
            "default": "distilbert-base-uncased-finetuned-sst-2-english",
            "type": "text",
        },
        "text-generation": {
            "impl": TextGenerationPipeline,
            "class": (ORTModelForCausalLM,),
            "default": "distilgpt2",
            "type": "text",
        },
        "token-classification": {
            "impl": TokenClassificationPipeline,
            "class": (ORTModelForTokenClassification,),
            "default": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "type": "text",
        },
        "zero-shot-classification": {
            "impl": ZeroShotClassificationPipeline,
            "class": (ORTModelForSequenceClassification,),
            "default": "facebook/bart-large-mnli",
            "type": "text",
        },
        "summarization": {
            "impl": SummarizationPipeline,
            "class": (ORTModelForSeq2SeqLM,),
            "default": "t5-base",
            "type": "text",
        },
        "translation": {
            "impl": TranslationPipeline,
            "class": (ORTModelForSeq2SeqLM,),
            "default": "t5-small",
            "type": "text",
        },
        "text2text-generation": {
            "impl": Text2TextGenerationPipeline,
            "class": (ORTModelForSeq2SeqLM,),
            "default": "t5-small",
            "type": "text",
        },
    }
else:
    ORT_SUPPORTED_TASKS = {}


def load_ascend_pipeline(
    model,
    targeted_task,
    load_tokenizer,
    tokenizer,
    feature_extractor,
    load_feature_extractor,
    SUPPORTED_TASKS,
    subfolder: str = "",
    token: Optional[Union[bool, str]] = None,
    revision: str = "main",
    model_kwargs: Optional[Dict[str, Any]] = None,
    config: AutoConfig = None,
    **kwargs,
):
    if model_kwargs is None:
        model_kwargs = {}

    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]["default"]
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(
            model_id, export=True
        )
    elif isinstance(model, str):
        from ..onnxruntime.modeling_seq2seq import (
            ENCODER_ONNX_FILE_PATTERN, ORTModelForConditionalGeneration)

        model_id = model
        ort_model_class = SUPPORTED_TASKS[targeted_task]["class"][0]

        if issubclass(ort_model_class, ORTModelForConditionalGeneration):
            pattern = ENCODER_ONNX_FILE_PATTERN
        else:
            pattern = ".+?.onnx"

        onnx_files = find_files_matching_pattern(
            model,
            pattern,
            glob_pattern="**/*.onnx",
            subfolder=subfolder,
            use_auth_token=token,
            revision=revision,
        )
        export = len(onnx_files) == 0
        model = ort_model_class.from_pretrained(model, export=export, **model_kwargs)
    elif isinstance(model, AscendModel):
        if tokenizer is None and load_tokenizer:
            for preprocessor in model.preprocessors:
                if isinstance(
                    preprocessor, (PreTrainedTokenizer, PreTrainedTokenizerFast)
                ):
                    tokenizer = preprocessor
                    break
            if tokenizer is None:
                raise ValueError(
                    "Could not automatically find a tokenizer for the ORTModel, you must pass a tokenizer explictly"
                )
        if feature_extractor is None and load_feature_extractor:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, SequenceFeatureExtractor):
                    feature_extractor = preprocessor
                    break
            if feature_extractor is None:
                raise ValueError(
                    "Could not automatically find a feature extractor for the ORTModel, you must pass a "
                    "feature_extractor explictly"
                )
        model_id = None
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or ORTModel.
            You can also provide non model then a default one will be used"""
        )
    return model, model_id, tokenizer, feature_extractor


MAPPING_LOADING_FUNC = {
    "ascend": load_ascend_pipeline,
}


def pipeline(
    task: str = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    accelerator: Optional[str] = "ascend",
    revision: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    *model_kwargs,
    **kwargs,
) -> Pipeline:
    targeted_task = "translation" if task.startswith("translation") else task

    if accelerator == "ascend":
        if targeted_task not in list(ORT_SUPPORTED_TASKS.keys()):
            raise ValueError(
                f"Task {targeted_task} is not supported for the ONNX Runtime pipeline. Supported tasks are { list(ORT_SUPPORTED_TASKS.keys())}"
            )
    else:
        raise ValueError(
            f"Accelerator {accelerator} is not supported. Supported accelerators are ['ascend']"
        )

    # copied from transformers.pipelines.__init__.py
    hub_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": None,
    }

    config = kwargs.get("config", None)
    if config is None and isinstance(model, str):
        config = AutoConfig.from_pretrained(
            model, _from_pipeline=task, **hub_kwargs, **kwargs
        )
        hub_kwargs["_commit_hash"] = config._commit_hash

    supported_tasks = (
        ORT_SUPPORTED_TASKS if accelerator == "ort" else TRANSFORMERS_SUPPORTED_TASKS
    )

    no_feature_extractor_tasks = set()
    no_tokenizer_tasks = set()
    for _task, values in supported_tasks.items():
        if values["type"] == "text":
            no_feature_extractor_tasks.add(_task)
        elif values["type"] in {"image", "video"}:
            no_tokenizer_tasks.add(_task)
        elif values["type"] in {"audio"}:
            no_tokenizer_tasks.add(_task)
        elif values["type"] not in ["multimodal", "audio", "video"]:
            raise ValueError(
                f"SUPPORTED_TASK {_task} contains invalid type {values['type']}"
            )

    # copied from transformers.pipelines.__init__.py l.609
    if targeted_task in no_tokenizer_tasks:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False
    else:
        load_tokenizer = True

    if targeted_task in no_feature_extractor_tasks:
        load_feature_extractor = False
    else:
        load_feature_extractor = True

    model, model_id, tokenizer, feature_extractor = MAPPING_LOADING_FUNC[accelerator](
        model,
        targeted_task,
        load_tokenizer,
        tokenizer,
        feature_extractor,
        load_feature_extractor,
        SUPPORTED_TASKS=supported_tasks,
        config=config,
        hub_kwargs=hub_kwargs,
        token=token,
        *model_kwargs,
        **kwargs,
    )

    if tokenizer is None and load_tokenizer:
        tokenizer = get_preprocessor(model_id)
    if feature_extractor is None and load_feature_extractor:
        feature_extractor = get_preprocessor(model_id)

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        use_fast=use_fast,
        **kwargs,
    )
