from optimum.commands.export import ExportCommand
from ..export.ascend import AscendExportCommand

REGISTER_COMMANDS = [(AscendExportCommand, ExportCommand)]
