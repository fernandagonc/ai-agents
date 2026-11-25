"""Allow running the assistant via `python -m calendar_agent`."""

import sys

from calendar_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
