# error: holds the error handling things
import sys
import flax.common


def error(msg, exit_status=1, prefix="'"):
    """error: errors with msg and optional exit_status"""
    print(prefix + str(msg), file=sys.stderr)
    exit(exit_status)


def debug(msg, prefix='"'):
    """debug: log a debug message to stderr"""
    if flax.common.DEBUG:
        print(prefix + str(msg), file=sys.stderr)
