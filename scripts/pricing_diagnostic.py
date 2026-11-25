"""Plot actual model-price outputs for explicit synthetic inputs, not forecasts."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from black76 import black76_price


def main() -> None:
    """Save a Black-76 call/put curve and all input parameters."""
    root = Path(__file__).resolve().parents[1]
    strikes = list(range(70, 131, 5))
    parameters = dict(F=100, T=0.5, sigma=0.25, D=0.98)
    calls = [black76_price(K=k, is_call=True, **parameters) for k in strikes]
    puts = [black76_price(K=k, is_call=False, **parameters) for k in strikes]
    (root / "results").mkdir(exist_ok=True)
    (root / "results/pricing_diagnostic.json").write_text(
        json.dumps(
            {
                "kind": "Synthetic-input model diagnostic; no market forecast",
                "parameters": parameters,
                "strikes": strikes,
                "calls": calls,
                "puts": puts,
            },
            indent=2,
        )
        + "\n"
    )
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(strikes, calls, label="Call", color="#176b60")
    ax.plot(strikes, puts, label="Put", color="#547da8")
    ax.set(
        xlabel="Strike",
        ylabel="Model price",
        title="Black-76 diagnostic · explicit synthetic inputs",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    (root / "assets").mkdir(exist_ok=True)
    fig.savefig(root / "assets/results.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
