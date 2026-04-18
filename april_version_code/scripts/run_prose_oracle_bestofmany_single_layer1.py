
from __future__ import annotations

from _script_utils import run_script_main
import run_prose_oracle_bestofmany as base_run


def main() -> None:
    import sys
    run_script_main('run_prose_oracle_bestofmany', base_run.main, ['--single-layer1-attempt', *sys.argv[1:]])


if __name__ == '__main__':
    main()
