"""CLI entry point khi gọi `python -m stage1_compat ...`"""

import sys
from stage1_compat.cli import main

if __name__ == "__main__":
    from stage1_compat.budget import BudgetExhausted
    try:
        sys.exit(main())
    except BudgetExhausted as error:
        print(f"PAUSED_QUOTA: {error}")
        sys.exit(75)
