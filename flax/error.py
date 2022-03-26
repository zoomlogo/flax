# error: holds the error handling things
import sys
from prompt_toolkit import print_formatted_text, HTML


def error(msg, exit_status=1):
    # error: errors with msg and optional exit_status
    print_formatted_text(HTML("<ansired>" + msg + "</ansired>"), file=sys.stderr)
    exit(exit_status)


def debug(msg, prefix="DEBUG: "):
    # debug: log a debug message to stderr
    print_formatted_text(
        HTML("<ansiyellow>" + prefix + msg + "</ansiyellow>"), file=sys.stderr
    )
