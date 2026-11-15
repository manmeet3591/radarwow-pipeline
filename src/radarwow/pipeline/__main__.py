"""Command line: ``python -m radarwow.pipeline run config.toml`` (or ``check`` to validate only)."""

import sys

from .config import ConfigError, Pipeline


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("run", "check"):
        print("usage: python -m radarwow.pipeline run|check config.toml")
        return 2
    try:
        p = Pipeline.from_toml(argv[1])
    except (ConfigError, OSError) as e:
        print(f"config error: {e}")
        return 1
    print(p.describe())
    if argv[0] == "run":
        res = p.run()
        print("\n".join(f"{k}: {v}" for k, v in res["files"].items()))
        if res["scores"] is not None:
            print(res["scores"].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
