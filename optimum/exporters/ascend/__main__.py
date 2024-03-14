import os
import shutil
import subprocess

import onnxruntime as ort

from .constants import OM_NAME_LIST, ONNX_NAME_LIST


MIN_BATCH_SIZE = 1
MIN_SEQUENCE_LENGTH = 32


def parse_onnx_inputs_outputs(onnx_model_path: str):
    model = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    input_dict = {}
    for input in model.get_inputs():
        input_dict[input.name] = input.shape

    output_dict = {}
    for output in model.get_outputs():
        output_dict[output.name] = output.shape

    return input_dict, output_dict


def export(
    model_path: str,
    output_path: str,
    soc_version: str = "Ascend310P3",
    max_batch_size: int = 8,
    max_sequence_length: int = 512,
    max_output_sequence_length: int = 512,
    **kwargs,
):
    input_dict, output_dict = parse_onnx_inputs_outputs(model_path)
    input_shape_list = []
    for name, shapes in input_dict.items():
        dims = []
        for shapes in shapes:
            if isinstance(shapes, int):
                dims.append(str(shapes))
            else:
                dims.append("-1")
        shape_str = f"{name}:{','.join(dims)}"
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
                for name, shapes in input_dict.items():
                    for i in range(len(shapes)):
                        dim_n = shapes[i]
                        if isinstance(dim_n, int):
                            spl.append(str(dim_n))
                        elif "batch_size" in dim_n:
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
    output: str = None,
    task: str = "auto",
    **kwargs,
):
    if not shutil.which("atc"):
        raise RuntimeError(
            "Ascend ATC command not found. Please make sure you have correctly installed the CANN with the atc command available."
        )

    # from optimum.exporters.onnx.base import OnnxConfig
    # onnx_config = infer_onnx_config(
    #     self.args.model,
    # )
    success = False

    for onnx_name, om_name in zip(ONNX_NAME_LIST, OM_NAME_LIST):
        model_path = os.path.join(onnx_model_path, onnx_name)
        if not os.path.exists(model_path):
            continue
        success = True
        output_path = os.path.join(output, om_name.replace(".om", ""))

        export(
            model_path,
            output_path,
            **kwargs,
        )

    if not success:
        raise RuntimeError(
            "No ONNX model found in the input directory. Please make sure you have correctly exported the ONNX model."
        )
