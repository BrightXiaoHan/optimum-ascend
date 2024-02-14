"""Defines the command line for the export with Ascend."""

import logging
import sys
from typing import TYPE_CHECKING, Optional

from optimum.commands.base import CommandInfo
from optimum.commands.export.onnx import ONNXExportCommand

from ...ascend.utils import infer_onnx_config
from ...exporters.ascend.__main__ import main_export


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def parse_args_ascend(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Ascend Required arguments")
    required_group.add_argument(
        "--soc-version",
        type=str,
        required=True,
        help="The version of the Ascend SOC to use for the exported model.",
    )

    optional_group = parser.add_argument_group("Ascend Optional arguments")
    optional_group.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="The maximum batch size to use for the exported model.",
    )
    optional_group.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="The maximum sequence length to use for the exported model. If the model is an encoder model or an deocder only model, this will be the maximum context length."
        "if the model is an encoder-decoder model, this will be the maximum input sequence length.",
    )
    optional_group.add_argument(
        "--max-output-sequence-length",
        type=int,
        default=512,
        help="The maximum output sequence length supported by the exported model. This is only used for encoder-decoder models.",
    )


class AscendExportCommand(ONNXExportCommand):
    COMMAND = CommandInfo(
        name="ascend", help="Export PyTorch models to Ascend offline model."
    )

    def __init__(
        self,
        subparsers: "_SubParsersAction",
        args: Optional["Namespace"] = None,
        command: Optional["CommandInfo"] = None,
        from_defaults_factory: bool = False,
        parser: Optional["ArgumentParser"] = None,
    ):
        super().__init__(
            subparsers,
            args=args,
            command=command,
            from_defaults_factory=from_defaults_factory,
            parser=parser,
        )
        self.args_string = " ".join(sys.argv[3:])

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        super(AscendExportCommand, AscendExportCommand).parse_args(parser)
        return parse_args_ascend(parser)

    def run(self):
        super().run()

        onnx_config = infer_onnx_config(
            self.args.model,
        )

        main_export(
            self.args.output,
            onnx_config,
            output=self.args.output,
            task=self.args.task,
            soc_version=self.args.soc_version,
            max_batch_size=self.args.max_batch_size,
            max_sequence_length=self.args.max_sequence_length,
            max_output_sequence_length=self.args.max_output_sequence_length,
        )
