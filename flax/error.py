# error: holds the error handling things
import sys
import flax.common
from colorama import Fore, Style


def error(msg, exit_status=1, prefix="'"):
    """error: errors with msg and optional exit_status"""
    print(Fore.RED + prefix + str(msg) + Style.RESET_ALL, file=sys.stderr)
    exit(exit_status)


def debug(msg, prefix='"'):
    """debug: log a debug message to stderr"""
    if flax.common.DEBUG:
        print(Fore.YELLOW + prefix + str(msg) + Style.RESET_ALL, file=sys.stderr)
