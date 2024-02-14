import os
import shutil
import subprocess

from optimum.exporters.onnx.base import OnnxConfig
from optimum.utils.constant import ONNX_WEIGHTS_NAME

from .constants import OM_WEIGHTS_NAME, ONNX_WEIGHTS_NAME


MIN_BATCH_SIZE = 1
MIN_SEQUENCE_LENGTH = 32


def export(
    model_path: str,
    onnx_config: OnnxConfig,
    output_path: str,
    task: str = "auto",
    soc_version: str = "Ascend310P3",
    max_batch_size: int = 8,
    max_sequence_length: int = 512,
    max_output_sequence_length: int = 512,
    **kwargs,
):
    input_shape_list = []
    for name, shape in onnx_config.inputs.items():
        shape_str = f"{name}:{','.join(['-1'] * len(shape))}"
        input_shape_list.append(shape_str)

    # input_shape e.g.1> "input_ids:-1,-1;attention_mask:-1,-1"
    input_shape = ";".join(input_shape_list)

    dynamic_dims_list = set()
    cur_bsz = 1
    cur_seq_len = MIN_SEQUENCE_LENGTH
    cur_output_seq_len = MIN_SEQUENCE_LENGTH
    while cur_bsz <= max_batch_size:
        while cur_seq_len <= max_sequence_length:
            while cur_output_seq_len <= max_output_sequence_length:
                spl = []  # spl = shape list
                for name, shape in onnx_config.inputs.items():
                    for i in range(len(shape)):
                        dim_n = shape[i]
                        if "batch_size" in dim_n:
                            spl.append(f"{cur_bsz}")
                        elif "decoder_sequence_length" in dim_n:
                            spl.append(f"{cur_seq_len}")
                        elif "sequence_length" in dim_n:
                            spl.append(f"{cur_output_seq_len}")

                dynamic_dims_list.add(",".join(spl))
                cur_output_seq_len *= 2
            cur_seq_len *= 2
            cur_output_seq_len = MIN_SEQUENCE_LENGTH
        cur_bsz *= 2
        cur_seq_len = MIN_SEQUENCE_LENGTH

    # dynamic_dims e.g.1> "1,64,1,64;2,64,2,64;4,64,4,64"
    dynamic_dims = ";".join(dynamic_dims_list)

    subprocess.run(
        f"atc --model={model_path} --framework=5 --output={output_path} --soc_version={soc_version} "
        f"--input_shape='{input_shape}' --dynamic_dims='{dynamic_dims}' "
        "--precision_mode=allow_fp32_to_fp16 --output_type FP32 --input_format=ND",
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main_export(
    onnx_model_path: str,
    onnx_config: OnnxConfig,
    output: str = None,
    task: str = "auto",
    **kwargs,
):
    if not shutil.which("atc"):
        raise RuntimeError(
            "Ascend ATC command not found. Please make sure you have correctly installed the CANN with the atc command available."
        )

    model_path = os.path.join(onnx_model_path, ONNX_WEIGHTS_NAME)
    output_path = os.path.join(output, OM_WEIGHTS_NAME.replace(".om", ""))

    export(
        model_path,
        onnx_config,
        output_path,
        task,
        **kwargs,
    )
