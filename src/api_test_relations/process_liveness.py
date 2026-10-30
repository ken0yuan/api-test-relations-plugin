"""Conservative Linux producer checks shared by post-exit evidence operations."""
from pathlib import Path


_PROC = Path('/proc')


def process_may_be_running(pid, start_ticks=None):
    """Zombies cannot write; unknown state still blocks recovery or disposal.

    A recorded start time can disambiguate a reused child PID. Without that
    evidence, any non-zombie process at the PID remains a possible writer.
    """
    try:
        stat = (_PROC / str(int(pid)) / 'stat').read_text(encoding='utf-8')
    except FileNotFoundError:
        return False
    except OSError:
        return True
    fields = stat.rpartition(')')[2].split()
    if len(fields) < 20:
        return True
    if fields[0] == 'Z':
        return False
    if start_ticks is not None and fields[19] != str(start_ticks):
        return False
    return True
