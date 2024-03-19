"""Defines the command line for the export with Ascend."""

import logging
import sys
from typing import TYPE_CHECKING, Optional

from optimum.commands.base import CommandInfo
from optimum.commands.export.onnx import ONNXExportCommand

from ...exporters.ascend.__main__ import main_export


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def parse_args_ascend(parser: "ArgumentParser"):
    # required_group = parser.add_argument_group("Ascend Required arguments")

    optional_group = parser.add_argument_group("Ascend Optional arguments")
    optional_group.add_argument(
        "--soc-version",
        type=str,
        default=None,
        help="The version of the Ascend SOC to use for the exported model. If not set, the SOC version will be auto-detected.",
    )
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
        help="The maximum sequence length to use for the exported model. If the model is an encoder model, this will be the maximum context length."
        "if the model is an encoder-decoder model, this will be the maximum input sequence length.",
    )
    optional_group.add_argument(
        "--max-output-sequence-length",
        type=int,
        default=512,
        help="The maximum generation sequence length to use for the exported model. This is only used for encoder-decoder and decoder models.",
    )
    optional_group.add_argument(
        "--max-prompts-length",
        type=int,
        default=128,
        help="The maximum generation prompts length supported by the exported model. This is only used for encoder-decoder and decoder models.",
    )
    optional_group.add_argument(
        "--from-onnx",
        action="store_true",
        help="If set, the input model is an ONNX model export from 'optimum export onnx' command."
        "Otherwise, the input model is a raw huggingface model.",
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
        if not self.args.from_onnx:
            super().run()

        main_export(
            self.args.output,
            output=self.args.output,
            task=self.args.task,
            soc_version=self.args.soc_version,
            max_batch_size=self.args.max_batch_size,
            max_sequence_length=self.args.max_sequence_length,
            max_output_sequence_length=self.args.max_output_sequence_length,
        )
