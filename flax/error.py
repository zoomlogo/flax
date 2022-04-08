# error: holds the error handling things
import sys

def error(msg, exit_status=1, prefix="ERROR: "):
    # error: errors with msg and optional exit_status
    print(
        "\x1b[0;31m" + prefix + msg + "\x1b[0m", file=sys.stderr
    )
    exit(exit_status)


def debug(msg, prefix="DEBUG: "):
    # debug: log a debug message to stderr
    print(
        "\x1b[0;33m" + prefix + msg + "\x1b[0m", file=sys.stderr
    )
