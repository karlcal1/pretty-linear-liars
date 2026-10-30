"""Table 1: probe loss, intervention loss, variance and angle to DAS for OLS, ridge and DAS on both testbeds."""
import json

import numpy as np

from pll.paths import RESULTS_DIR


def load_json(name):
    with open(RESULTS_DIR / name) as handle:
        return json.load(handle)


def hmm_rows():
    sweep = load_json("hmm_sweep.json")
    das = load_json("hmm_das.json")
    ols = sweep["lambdas"].index(0.0)
    ridge = sweep["lambdas"].index(das["selected_lambda"])
    curve = das["rotation_curve"]
    return [
        ("Mess3 HMM", "OLS probe", sweep["probe_loss"][ols], sweep["interv_mse"][ols], sweep["variance"][ols],
         curve["angle_to_DAS"][curve["lambdas"].index(0.0)]),
        ("", f"Ridge probe (lambda = {das['selected_lambda']:g})", sweep["probe_loss"][ridge], sweep["interv_mse"][ridge],
         sweep["variance"][ridge], das["angle_ridge_to_DAS"]),
        ("", "DAS", das["probe_loss"], das["interv_mse"], das["variance"], 0.0),
    ]


def llm_rows():
    sweep = load_json("llm_sweep.json")
    ridge = sweep["ridge"][sweep["lambdas"].index(sweep["selected_lambda"])]
    angle = lambda cosine: float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))  # noqa: E731
    return [
        ("Llama-2-7B", "OLS probe", sweep["OLS"]["probe_loss"], sweep["OLS"]["interv_mse"], sweep["OLS"]["variance"],
         angle(sweep["OLS"]["cos_DAS"])),
        ("(birth year)", f"Ridge probe (lambda = {sweep['selected_lambda']:.0e})", ridge["probe_loss"], ridge["interv_mse"],
         ridge["variance"], angle(ridge["cos_DAS"])),
        ("", "DAS", sweep["DAS"]["probe_loss"], sweep["DAS"]["interv_mse"], sweep["DAS"]["variance"], 0.0),
    ]


def main():
    rows = []
    for loader in (hmm_rows, llm_rows):
        try:
            rows.extend(loader())
        except FileNotFoundError as missing:
            print(f"skipping rows, missing input {missing.filename}")
    header = "| Testbed | Direction | Probe loss (MSE) | Int. loss (MSE) | Variance | Angle to DAS |"
    lines = [header, "|---|---|---|---|---|---|"]
    for testbed, direction, probe_loss, interv_loss, variance, angle in rows:
        lines.append(f"| {testbed} | {direction} | {probe_loss:.2e} | {interv_loss:.2e} | {variance:.2e} | {angle:.1f} deg |")
    table = "\n".join(lines)
    print(table)
    with open(RESULTS_DIR / "table1.md", "w") as handle:
        handle.write(table + "\n")
    print("wrote results/table1.md")


if __name__ == "__main__":
    main()
