import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from transformers import GenerationConfig, PretrainedConfig
from transformers.file_utils import add_start_docstrings

from ..exporters.ascend import main_export
from ..exporters.ascend.constants import OM_DECODER_NAME, OM_DECODER_WITH_PAST_NAME, OM_ENCODER_NAME
from .acl_model import ACLModel
from .modeling_base import AscendBaseModel


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Base AscendModelForSeq2SeqLM class.
    """,
)
class AscendBaseModelForSeq2SeqLM(AscendBaseModel):
    export_feature = "text2text-generation"

    def __init__(
        self,
        encoder: ACLModel,
        decoder: ACLModel,
        decoder_with_past: ACLModel = None,
        config: PretrainedConfig = None,
        device: int = 0,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.config = config
        self.use_cache = decoder_with_past is not None
        self.model_save_dir = model_save_dir
        self.device = device
        self.preprocessors = kwargs.get("preprocessors", [])

        self.encoder_model = encoder
        self.decoder_model = decoder
        self.decoder_with_past_model = decoder_with_past
        self.generation_config = (
            GenerationConfig.from_model_config(config) if self.can_generate() else None
        )

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the Ascend OM format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """
        src_files = [self.encoder_model, self.decoder_model]
        dst_file_names = [OM_ENCODER_NAME, OM_DECODER_NAME]
        if self.use_cache:
            src_files.append(self.decoder_with_past_model)
            dst_file_names.append(OM_DECODER_WITH_PAST_NAME)

        for src_file, dst_file_name in zip(src_files, dst_file_names):
            dst_path = os.path.join(save_directory, dst_file_name)
            shutil.copy(src_file, dst_path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        encoder_file_name: Optional[str] = None,
        decoder_file_name: Optional[str] = None,
        decoder_with_past_file_name: Optional[str] = None,
        local_files_only: bool = False,
        use_cache: bool = True,
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
            revision (`str`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            encoder_file_name(`str`, *optional*):
                The encoder model file name. Overwrites the default file name openvino_encoder_model.xml and allows one to
                load the encoder model with a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name. Overwrites the default file name openvino_decoder_model.xml and allows one to
                load the decoder model with a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name
                openvino_decoder_with_past_model.xml, allowing to load the decoder model with a different name.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        default_encoder_file_name = OM_ENCODER_NAME
        default_decoder_file_name = OM_DECODER_NAME
        default_decoder_with_past_file_name = OM_DECODER_WITH_PAST_NAME
        encoder_file_name = encoder_file_name or default_encoder_file_name
        decoder_file_name = decoder_file_name or default_decoder_file_name
        decoder_with_past_file_name = (
            decoder_with_past_file_name or default_decoder_with_past_file_name
        )
        decoder_with_past = None
        # Load model from a local directory
        if os.path.isdir(model_id):
            encoder = cls.load_model(os.path.join(model_id, encoder_file_name))
            decoder = cls.load_model(os.path.join(model_id, decoder_file_name))
            if use_cache:
                decoder_with_past = cls.load_model(
                    os.path.join(model_id, decoder_with_past_file_name)
                )

            model_save_dir = Path(model_id)

        # Load model from hub
        else:
            model_file_names = {
                "encoder": encoder_file_name,
                "decoder": decoder_file_name,
            }
            if use_cache:
                model_file_names["decoder_with_past"] = decoder_with_past_file_name

            file_names = model_file_names.copy()
            for name, file_name in model_file_names.items():
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_name,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                file_names[name] = model_cache_path

            model_save_dir = Path(model_cache_path).parent
            encoder = cls.load_model(file_names["encoder"])
            decoder = cls.load_model(file_names["decoder"])
            if use_cache:
                decoder_with_past = cls.load_model(file_names["decoder_with_past"])

        return cls(
            encoder=encoder,
            decoder=decoder,
            decoder_with_past=decoder_with_past,
            config=config,
            model_save_dir=model_save_dir,
            **kwargs,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        task: Optional[str] = None,
        use_cache: bool = True,
        trust_remote_code: bool = False,
        load_in_8bit: Optional[bool] = None,
        **kwargs,
    ):
        """
        Export a vanilla Transformers model into an ONNX model using `transformers.onnx.export_onnx`.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            save_dir (`str` or `Path`):
                The directory where the exported ONNX model should be saved, defaults to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        if task is None:
            task = cls.export_feature

            if use_cache:
                task = task + "-with-past"

        compression_option = None
        if load_in_8bit is not None:
            compression_option = "fp32"
        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            compression_option=compression_option,
        )

        config.save_pretrained(save_dir_path)
        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            load_in_8bit=load_in_8bit,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError
