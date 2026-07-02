"""Console colors and logging helper."""

DIM = "\033[2m"
YOU = "\033[96m"
AI = "\033[92m"
SYS = "\033[93m"
RESET = "\033[0m"


def log(msg: str, color: str = SYS) -> None:
    print(f"{color}{msg}{RESET}", flush=True)
