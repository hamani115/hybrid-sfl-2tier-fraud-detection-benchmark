#!/usr/bin/env python3
from paper_analysis import main
if __name__ == "__main__":
    import sys
    sys.argv += ["--benchmark", "main"] if "--benchmark" not in sys.argv else []
    main()
