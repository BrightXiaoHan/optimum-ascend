from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List

import numpy as np


ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


@dataclass
class TensorMetadata:
    name: str  # 张量名称
    buffer: int  # 设备内存地址指针
    size: int  # 内存字节大小
    dimCount: int  # 张量维度
    dims: List[int]  # 张量维度大小
    index: int  # 张量索引
    dims_gear_position: List[int] = field(default_factory=list)  # 动态档位中的位置
    buffer_host: int = -1  # 宿主机内存地址指针, 默认为-1为不占用宿主机内存


class ACLModel:
    def __init__(
        self,
        model_path: str,
        device: int = 0,
        pad_id: int = 0,
    ):
        try:
            import acl
        except ImportError:
            raise ImportError(
                "Can't find the acl module. Please check if you have installed the CANN Toolkit correctly. "
            )
        self.acl = acl
        self.model_path = model_path
        self.device = device
        self.pad_id = pad_id
        ret = self.acl.init()
        assert ret == ACL_SUCCESS, "acl init failed. ret = {}".format(ret)

        self._init_memory()

        gear_count, ret = self.acl.mdl.get_input_dynamic_gear_count(self.model_desc, -1)
        assert (
            ret == ACL_SUCCESS
        ), "get input dynamic gear count failed. ret = {}".format(ret)
        self.dim_gears, ret = self.acl.mdl.get_input_dynamic_dims(
            self.model_desc, -1, gear_count
        )
        assert ret == ACL_SUCCESS, "get input dynamic dims failed. ret = {}".format(ret)

    def _init_memory(self):
        self.context, ret = self.acl.rt.create_context(self.device)
        assert ret == ACL_SUCCESS, "set device failed. ret = {}".format(ret)

        ret = self.acl.rt.set_context(self.context)
        assert ret == ACL_SUCCESS, "set context failed. ret = {}".format(ret)

        # 加载离线模型文件，返回标识模型的ID
        self.model_id, ret = self.acl.mdl.load_from_file(self.model_path)

        assert ret == ACL_SUCCESS, "load model from file failed. ret = {}".format(ret)

        # 创建空白模型描述信息，获取模型描述信息的指针地址
        self.model_desc = self.acl.mdl.create_desc()

        # 通过模型的ID，将模型的描述信息填充到model_desc
        ret = self.acl.mdl.get_desc(self.model_desc, self.model_id)
        # 创建aclmdlDataset类型的数据，描述模型推理的输入。
        self.load_input_dataset = self.acl.mdl.create_dataset()
        # 获取模型输入张量数。
        input_size = self.acl.mdl.get_num_inputs(self.model_desc)
        self.input_data: Dict[str, TensorMetadata] = {}
        gear_index = 0
        # 循环为每个输入申请内存，并将每个输入添加到aclmdlDataset类型的数据中。
        for i in range(input_size):
            buffer_size = self.acl.mdl.get_input_size_by_index(self.model_desc, i)
            metadata, ret = self.acl.mdl.get_input_dims_v2(self.model_desc, i)
            assert ret == ACL_SUCCESS, "get input dims failed. ret = {}".format(ret)
            # 申请输入内存。
            buffer, ret = self.acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == ACL_SUCCESS, "malloc failed. ret = {}".format(ret)
            data = self.acl.create_data_buffer(buffer, buffer_size)
            _, ret = self.acl.mdl.add_dataset_buffer(self.load_input_dataset, data)
            assert ret == ACL_SUCCESS, "add dataset buffer failed. ret = {}".format(ret)
            dims_gear_position = [-1] * metadata["dimCount"]
            for j, dim in enumerate(metadata["dims"]):
                if dim == -1:
                    dims_gear_position[j] = gear_index
                    gear_index += 1

            self.input_data[metadata["name"]] = TensorMetadata(
                buffer=buffer,
                size=buffer_size,
                **metadata,
                dims_gear_position=dims_gear_position,
                index=i,
            )

        # 2.准备模型推理的输出数据集。
        # 创建aclmdlDataset类型的数据，描述模型推理的输出。
        self.load_output_dataset = self.acl.mdl.create_dataset()
        # 获取模型输出的数量。
        output_size = self.acl.mdl.get_num_outputs(self.model_desc)
        self.output_data: Dict[str, TensorMetadata] = {}
        # 循环为每个输出申请内存，并将每个输出添加到aclmdlDataset类型的数据中。
        for i in range(output_size):
            buffer_size = self.acl.mdl.get_output_size_by_index(self.model_desc, i)
            metadata, ret = self.acl.mdl.get_output_dims(self.model_desc, i)
            assert ret == ACL_SUCCESS, "get output dims failed. ret = {}".format(ret)
            # 申请输出内存。
            buffer, ret = self.acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            assert ret == ACL_SUCCESS, "malloc failed. ret = {}".format(ret)

            data = self.acl.create_data_buffer(buffer, buffer_size)
            _, ret = self.acl.mdl.add_dataset_buffer(self.load_output_dataset, data)
            assert ret == ACL_SUCCESS, "add dataset buffer failed. ret = {}".format(ret)

            buffer_host, ret = self.acl.rt.malloc_host(buffer_size)
            assert ret == ACL_SUCCESS, "malloc host failed. ret = {}".format(ret)
            self.output_data[metadata["name"].split(":")[-1]] = TensorMetadata(
                buffer=buffer,
                buffer_host=buffer_host,
                size=buffer_size,
                **metadata,
                index=i,
            )

        # create stream
        self.stream, ret = self.acl.rt.create_stream()
        assert ret == ACL_SUCCESS, "create stream failed. ret = {}".format(ret)

    def _free(self):
        ret = self.acl.rt.destroy_context(self.context)
        assert ret == ACL_SUCCESS, "destroy context failed. ret = {}".format(ret)

    def set_device(self, device: int):
        if self.device != device:
            self.device = device
            self._free()
            self._init_memory()

    def _model_set_dynamic_info(self, dims: List[int]) -> List[int]:
        index, ret = self.acl.mdl.get_input_index_by_name(
            self.model_desc, "ascend_mbatch_shape_data"
        )
        assert ret == ACL_SUCCESS, "get input index by name failed. ret = {}".format(
            ret
        )
        dim_gear = self._find_best_gear(dims)
        ret = self.acl.mdl.set_input_dynamic_dims(
            self.model_id,
            self.load_input_dataset,
            index,
            dim_gear,
        )
        assert ret == ACL_SUCCESS, "set input dynamic dims failed. ret = {}".format(ret)
        return dim_gear["dims"]

    def _find_best_gear(self, given_dims: List[int]):
        best_match = None
        min_diff = float("inf")

        for gear in self.dim_gears:
            dims = gear["dims"]

            if len(dims) != len(given_dims):
                raise ValueError(
                    f"Given dims {given_dims} has invalid length {len(given_dims)}. Expected {len(dims)}"
                )
            diff = 0
            valid = True
            for i in range(len(dims)):
                if dims[i] < given_dims[i]:
                    valid = False
                    break
                diff += dims[i] - given_dims[i]
            if not valid:
                continue
            if diff < min_diff:
                min_diff = diff
                best_match = gear

        if best_match is None:
            raise ValueError(f"No gear found for given dims {given_dims}")

        return best_match

    def _pad_to_shape(self, array: np.ndarray, shape: List[int]) -> np.ndarray:
        if len(array.shape) != len(shape):
            raise ValueError(
                "Array and target shape must have the same number of dimensions"
            )

        pad_width = [(0, shape[i] - array.shape[i]) for i in range(len(shape))]

        return np.pad(array, pad_width, mode="constant", constant_values=self.pad_id)

    def __call__(self, model_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        ret = self.acl.rt.set_context(self.context)
        assert ret == ACL_SUCCESS, "set context failed. ret = {}".format(ret)
        batch_size = model_inputs["input_ids"].shape[0]
        sequence_length = model_inputs["input_ids"].shape[1]

        input_index_to_dims = {}
        for name, input_ in model_inputs.items():
            if name not in self.input_data:
                continue
            dims = self.input_data[name].dims
            input_index_to_dims[self.input_data[name].index] = [
                input_.shape[i] for i in range(len(dims)) if dims[i] == -1
            ]

        indexes = sorted(input_index_to_dims.keys())
        dynamic_dims = list(
            chain.from_iterable(input_index_to_dims[i] for i in indexes)
        )

        static_dims = self._model_set_dynamic_info(dynamic_dims)

        for name, input_ in model_inputs.items():
            if name not in self.input_data:
                continue
            input_data = self.input_data[name]
            refine_dims = []
            for i, dim in enumerate(input_data.dims):
                if dim == -1:
                    refine_dims.append(static_dims[input_data.dims_gear_position[i]])
                else:
                    refine_dims.append(dim)

            model_inputs[name] = self._pad_to_shape(input_, refine_dims)

        model_inputs_bytes = {
            k: v.tobytes() for k, v in model_inputs.items() if k in self.input_data
        }
        model_inputs_ptrs = {
            k: self.acl.util.bytes_to_ptr(v) for k, v in model_inputs_bytes.items()
        }

        for name, ptr in model_inputs_ptrs.items():
            if name not in self.input_data:
                continue

            ret = self.acl.rt.memcpy(
                self.input_data[name].buffer,
                self.input_data[name].size,
                ptr,
                len(model_inputs_bytes[name]),
                ACL_MEMCPY_HOST_TO_DEVICE,
            )
            assert ret == ACL_SUCCESS, "memcpy failed. ret = {}".format(ret)

        # 3.执行模型推理。
        # self.model_id表示模型ID，在模型加载成功后，会返回标识模型的ID。
        ret = self.acl.mdl.execute(
            self.model_id, self.load_input_dataset, self.load_output_dataset
        )
        assert ret == ACL_SUCCESS, "execute model failed. ret = {}".format(ret)

        model_outputs = {}
        for name, data in self.output_data.items():
            cur_output_dims, _ = self.acl.mdl.get_cur_output_dims(
                self.model_desc, data.index
            )
            cur_output_dims = cur_output_dims["dims"]
            times = 1
            for cur_dim, dim in zip(cur_output_dims, data.dims):
                times *= dim // cur_dim
            cur_output_size = data.size // times
            ret = self.acl.rt.memcpy(
                data.buffer_host,
                data.size,
                data.buffer,
                cur_output_size,
                ACL_MEMCPY_DEVICE_TO_HOST,
            )

            bytes_out = self.acl.util.ptr_to_bytes(data.buffer_host, cur_output_size)
            output_data_type = self.acl.mdl.get_output_data_type(
                self.model_desc, data.index
            )
            # TODO dynamic get output data type, now only support float32
            assert output_data_type == 0, "output data type is not float"
            data = np.frombuffer(bytes_out, dtype=np.float32)
            array = data.reshape(cur_output_dims)
            if len(array.shape) <= 2:
                array = array[:batch_size]
            else:
                array = array[:batch_size, :sequence_length]

            model_outputs[name] = array

        return model_outputs
