from typing import Optional

from optimum.exporters import TasksManager
from optimum.exporters.onnx import OnnxConfig
from transformers import AutoConfig


try:
    import acl
except ImportError:
    acl = None


def is_acl_available():
    if acl is None:
        return False
    else:
        return True


def get_soc_name():
    if not is_acl_available():
        raise ImportError(
            "ACL is not available. Auto detection of SoC name is not possible."
        )
    name = acl.get_soc_name()
    return name


def infer_onnx_config(
    model_id,
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> OnnxConfig:
    config = AutoConfig.from_pretrained(
        model_id,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        **kwargs,
    )
    library_name = TasksManager.infer_library_from_model(
        model_id,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
    )
    # TODO: Hardcoded for now, need to find a way to infer this from the model
    if library_name == "sentence_transformers":
        model_type = "transformer"
    else:
        model_type = config.model_type
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        "onnx",
        model_type=model_type,
        library_name=library_name,
    )
    onnx_config = onnx_config_constructor(config)
    return onnx_config
