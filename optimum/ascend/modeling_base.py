import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from optimum.exporters import TasksManager
from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx import main_export as onnx_main_export
from optimum.modeling_base import OptimizedModel
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import GenerationConfig, PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.generation import GenerationMixin

from ..exporters.ascend import main_export as ascend_main_export
from ..exporters.ascend.constants import OM_WEIGHTS_NAME
from .acl_model import ACLModel
from .utils import infer_onnx_config


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Base AscendModel class.
    """,
)
class AscendBaseModel(OptimizedModel):
    def __init__(
        self,
        model: ACLModel,
        config: PretrainedConfig = None,
        device: str = "0",
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        onnx_config: OnnxConfig = None,
        **kwargs,
    ):
        self.config = config
        self._device = device.upper()
        self.preprocessors = kwargs.get("preprocessors", [])

        self.onnx_config = onnx_config

        self.model = model

        self.model_save_dir = model_save_dir

        self.generation_config = (
            GenerationConfig.from_model_config(config) if self.can_generate() else None
        )

    @staticmethod
    def load_model(file_name: Union[str, Path], **kwargs):
        """
        Loads the model.

        Arguments:
            file_name (`str` or `Path`):
                The path of the model .om file.
        """

        if isinstance(file_name, str):
            file_name = Path(file_name)

        if file_name.suffix != ".om":
            raise ValueError(
                f"Model file should have .om extension, got {file_name.suffix} instead."
            )

        model = ACLModel(file_name.as_posix(), **kwargs)
        return model

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the .om format so that it can be re-loaded using the
        [`~optimum.ascend.modeling.AscendModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """
        original_file = os.path.join(self.model_save_dir, OM_WEIGHTS_NAME)
        target_file = os.path.join(save_directory, OM_WEIGHTS_NAME)

        if Path(original_file) != Path(target_file):
            shutil.copy(original_file, target_file)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = True,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            use_auth_token (`str` or `bool`):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private
                repository.
            revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name (`str`, *optional*):
                The file name of the model to load. Overwrites the default file name and allows one to load the model
                with a different name.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        model_path = Path(model_id)
        default_file_name = OM_WEIGHTS_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )
        acl_model = cls.load_model(model_cache_path)
        onnx_config = infer_onnx_config(
            model_id,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            **kwargs,
        )
        model = cls(
            acl_model,
            config=config,
            model_save_dir=model_cache_path.parent,
            onnx_config=onnx_config,
            **kwargs,
        )
        return model

    @staticmethod
    def _cached_file(
        model_path: Union[Path, str],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
    ):
        # locates a file in a local folder and repo, downloads and cache it if necessary.
        model_path = Path(model_path)
        if model_path.is_dir():
            model_cache_path = model_path / file_name
        else:
            file_name = Path(file_name)
            if file_name.suffix != ".onnx":
                model_file_names = [file_name.with_suffix(".bin"), file_name]
            else:
                model_file_names = [file_name]
            for file_name in model_file_names:
                model_cache_path = hf_hub_download(
                    repo_id=model_path.as_posix(),
                    filename=file_name.as_posix(),
                    subfolder=subfolder,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            model_cache_path = Path(model_cache_path)

        return model_cache_path

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        """
        if isinstance(self, GenerationMixin):
            return True
        return False

    def to(self, device: Optional[Union[int, str]] = 0):
        """
        Moves the model to the specified device.
        """
        if not isinstance(device, int):
            try:
                device = int(device)
            except (ValueError, TypeError):
                logging.warning(
                    f"Ascend device should be an integer or a string representing an integer, got {device} instead. Using default device."
                )
                device = 0
        self.model.set_device(device)

    @classmethod
    def _from_transformers(
        cls,
        model_id: Union[str, Path],
        task: str = "auto",
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "OptimizedModel":
        """
        Export a vanilla Transformers model into an Ascend om model using ATC tools.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.            save_dir (`str` or `Path`):
                The directory where the exported ONNX model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            task: (`str`, *optional*):
                The task to perform. If not provided, it will be inferred from the model's type.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the export function and model constructor.
        """
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        if task == "auto":
            try:
                task = TasksManager.infer_task_from_model(
                    model_id,
                    subfolder=subfolder,
                    revision=revision,
                )
            except RequestsConnectionError as e:
                raise RequestsConnectionError(
                    f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                )
            except Exception as e:
                raise ValueError(
                    f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                )

        onnx_main_export(
            model_id,
            output=save_dir_path,
            task=task,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            **kwargs,
        )
        onnx_config = infer_onnx_config(
            model_id,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            **kwargs,
        )

        ascend_main_export(
            save_dir_path,
            onnx_config,
            output=save_dir_path,
            task=task,
            **kwargs,
        )
        return cls._from_pretrained(save_dir_path, **kwargs)
