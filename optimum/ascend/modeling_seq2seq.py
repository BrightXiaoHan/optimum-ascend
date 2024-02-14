import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from .acl_model import ACLModel
from .modeling_base_seq2seq import AscendBaseModelForSeq2SeqLM


logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "AutoTokenizer"

INPUTS_DOCSTRING = r"""
    Arguments:
        encoder (`optimum.ascend.ACLModel`):
            The OpenVINO Runtime model associated to the encoder.
        decoder (`openvino.runtime.Model`):
            The OpenVINO Runtime model associated to the decoder.
        decoder_with_past (`openvino.runtime.Model`, *optional*):
            The OpenVINO Runtime model associated  to the decoder with past key values.
        config (`transformers.PretrainedConfig`):
            [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is an instance of the configuration associated to the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
"""

ENCODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`np.ndarray`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`np.ndarray`, *optional*):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
"""


DECODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`np.ndarray`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_hidden_states (`np.ndarray`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        encoder_attention_mask (`np.ndarray`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder `input_ids`.
        past_key_values (`tuple(tuple(np.ndarray), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

SEQ2SEQ_MODEL_DOCSTRING = r"""
    Arguments:
        input_ids (`np.ndarray`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`np.ndarray`, *optional*):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
        decoder_input_ids (`np.ndarray`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`np.ndarray`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(np.ndarray), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""


TRANSLATION_EXAMPLE = r"""
    Example of text generation:
    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> text = "He never went out without a book under his arm, and he often came back with two."
    >>> inputs = tokenizer(text, return_tensors="pt")
    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
    >>> text = "He never went out without a book under his arm, and he often came back with two."
    >>> outputs = pipe(text)
    ```
"""


@add_start_docstrings(
    """
    Sequence-to-sequence model with a language modeling head for OpenVINO inference.
    """,
    INPUTS_DOCSTRING,
)
class AscendModelForSeq2SeqLM(AscendBaseModelForSeq2SeqLM, GenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM
    main_input_name = "input_ids"
    export_feature = "text2text-generation"

    def __init__(
        self,
        encoder: ACLModel,
        decoder: ACLModel,
        decoder_with_past: ACLModel = None,
        config: transformers.PretrainedConfig = None,
        **kwargs,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            decoder_with_past=decoder_with_past,
            config=config,
            **kwargs,
        )
        self.device = torch.device("cpu")
        self.decoder_with_past = None
        enable_compilation = kwargs.get("compile", True)
        self.encoder = AscendEncoder(self.encoder_model, parent_model=self)
        self.decoder = AscendDecoder(self.decoder_model, parent_model=self)

        if self.use_cache:
            self.decoder_with_past = AscendDecoder(
                self.decoder_with_past_model, parent_model=self
            )
        if enable_compilation:
            self.compile()

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        try:
            self.auto_model_class.register(AutoConfig, self.__class__)
        except AttributeError:
            pass

    def to(self, device: str):
        if isinstance(device, str):
            self._device = device.upper()
            self.encoder._device = self._device
            self.decoder._device = self._device
            if self.use_cache:
                self.decoder_with_past._device = self._device
            self.clear_requests()
        else:
            logger.warning(
                f"device must be of type {str} but got {type(device)} instead"
            )

        return self

    @add_start_docstrings_to_model_forward(
        SEQ2SEQ_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + TRANSLATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="AscendModelForSeq2SeqLM",
            checkpoint="echarlaix/t5-small-openvino",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encode if needed : first prediction pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        # Decode
        if past_key_values is None or self.decoder_with_past is None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
        else:
            decoder_outputs = self.decoder_with_past(
                input_ids=decoder_input_ids[
                    :, -1:
                ],  # Cut decoder_input_ids if past is used
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values or kwargs.get("past", None),
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def get_encoder(self):
        return self.encoder

    # Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    @staticmethod
    def _reorder_cache(past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        reordered_past = ()
        for layer_past in past:
            # Cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past

    def reshape(self, batch_size: int, sequence_length: int):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            batch_size (`int`):
                The batch size.
            sequence_length (`int`):
                The sequence length.
        """
        super().reshape(batch_size, sequence_length)
        self.clear_requests()
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        super().half()
        self.clear_requests()
        return self

    def clear_requests(self):
        self.encoder.request = None
        self.decoder.request = None
        if self.use_cache:
            self.decoder_with_past.request = None

    def compile(self):
        self.encoder._compile()
        self.decoder._compile()
        if self.use_cache:
            self.decoder_with_past._compile()


class AscendEncoder:
    """
    Encoder model for OpenVINO inference.

    Arguments:
        request (`openvino.runtime.ie_api.InferRequest`):
            The OpenVINO inference request associated to the encoder.
    """

    def __init__(self, model: ACLModel, parent_model: AscendBaseModelForSeq2SeqLM):
        self.model = model
        self.parent_model = parent_model
        self._device = self.parent_model._device
        self.device = torch.device("cpu")
        self.input_names = {
            key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)
        }
        self.main_input_name = self.parent_model.main_input_name or "input_ids"
        self.request = None

    @add_start_docstrings_to_model_forward(ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        **kwargs,
    ) -> BaseModelOutput:
        self._compile()

        # Model inputs
        inputs = {
            self.main_input_name: input_ids
            if input_ids is not None
            else kwargs.get(self.main_input_name)
        }

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        # Run inference
        last_hidden_state = torch.from_numpy(
            self.request(inputs, share_inputs=True, share_outputs=True)[
                "last_hidden_state"
            ]
        ).to(self.device)

        return BaseModelOutput(last_hidden_state=last_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class AscendDecoder:
    """
    Decoder model for OpenVINO inference.

    Arguments:
        request (`openvino.runtime.ie_api.InferRequest`):
            The OpenVINO inference request associated to the decoder.
        device (`torch.device`):
            The device type used by this process.
    """

    def __init__(self, model: ACLModel, parent_model: AscendModelForSeq2SeqLM):
        self.model = model
        self.parent_model = parent_model
        self._device = self.parent_model._device
        self.device = torch.device("cpu")
        self.input_names = {
            key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)
        }
        self.key_value_input_names = [
            key for key in self.input_names if "key_values" in key
        ]
        self.output_names = {
            key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)
        }
        self.key_value_output_names = [
            key for key in self.output_names if "key_values" in key or "present" in key
        ]
        is_legacy = any(
            "past_key_values" in key.get_any_name() for key in self.model.outputs
        )

        if len(self.key_value_input_names) > 0 and not is_legacy:
            self.use_past = True
            self.num_pkv = 2
        else:
            self.use_past = False
            self.num_pkv = 4

        self.request = None

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        self._compile()
        # Model inputs
        inputs = {}

        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(
                past_key_value
                for pkv_per_layer in past_key_values
                for past_key_value in pkv_per_layer
            )

            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        inputs["input_ids"] = input_ids

        # Add the encoder_attention_mask inputs when needed
        if (
            "encoder_attention_mask" in self.input_names
            and encoder_attention_mask is not None
        ):
            inputs["encoder_attention_mask"] = encoder_attention_mask

        # Add the encoder_hidden_states inputs when needed
        if (
            "encoder_hidden_states" in self.input_names
            and encoder_hidden_states is not None
        ):
            inputs["encoder_hidden_states"] = encoder_hidden_states

        if (
            "decoder_attention_mask" in self.input_names
            and decoder_attention_mask is not None
        ):
            inputs["decoder_attention_mask"] = decoder_attention_mask
        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(
            self.device
        )

        # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
        # self-attention layer and 2 to the cross-attention layer)
        out_past_key_values = tuple(
            self.request.get_tensor(key).data for key in self.key_value_output_names
        )

        # Tuple of tuple of length `n_layers`, with each tuple of length equal to:
        # * 4 for the decoder without cache (k/v of self-attention + k/v of cross-attention)
        # * 2 for the decoder with cache (k/v of self-attention as cross-attention cache is constant)
        if self.use_past is False:
            out_past_key_values = tuple(
                out_past_key_values[i : i + self.num_pkv]
                for i in range(0, len(out_past_key_values), self.num_pkv)
            )
        else:
            # grab the cross attention key/values from the inputs
            out_past_key_values = tuple(
                out_past_key_values[i : i + self.num_pkv]
                + past_key_values[2 * i + 2 : 2 * i + 2 + self.num_pkv]
                for i in range(0, len(out_past_key_values), self.num_pkv)
            )

        return Seq2SeqLMOutput(logits=logits, past_key_values=out_past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
