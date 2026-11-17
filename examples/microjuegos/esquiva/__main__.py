import sys

from examples.microjuegos.app import microjuegos

microjuegos.main([__package__, *sys.argv[1:]], prog_name=f"python -m {__package__}")
