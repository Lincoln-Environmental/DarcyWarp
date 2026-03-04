from DARCY_WARP_PACKAGE.bench_and_plot import main as _bench_main


def main(argv: list[str] | None = None) -> int:
    return int(_bench_main(argv))


if __name__ == "__main__":
    argv = ['--run_mf', '--run_warp']
    raise SystemExit(main(argv))
