import sys
from prompt_toolkit import print_formatted_text, HTML


def error(msg, exit_status=1):
    print_formatted_text(HTML("<ansired>" + msg + "</ansired>"), file=sys.stderr)
    exit(exit_status)
