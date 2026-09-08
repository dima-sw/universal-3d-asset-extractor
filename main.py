"""Single entry point.

    python main.py            -> GUI
    python main.py gui        -> GUI
    python main.py scan ...   -> CLI (everything after the first arg goes to the CLI)
"""
from __future__ import annotations

import sys

GUI_COMMANDS = {"", "gui", "ui"}


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in GUI_COMMANDS:
        from app.ui.app import main as gui_main
        return gui_main([sys.argv[0]])
    from app.cli import main as cli_main
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
