#!/usr/bin/env python3
"""Render real solver arrays from an explicit, solver-independent figure spec.

No solver, smoothing, interpolation or gradient normalization is implemented.
Dependencies: numpy, matplotlib; Pillow only for optional GIF exports.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import operator
from pathlib import Path
import re
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "savefig.dpi": 300})


def array(data, key):
    value = np.asarray(data[key])
    if value.dtype.kind not in "fi" or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{key}: require a nonempty finite real array")
    return value


def field(data, panel):
    value = array(data, panel["key"])
    if "frame" in panel:
        index = operator.index(panel["frame"])
        if index < 0 or index >= value.shape[0]:
            raise ValueError("history frame outside the saved range")
        value = value[index]
    if panel["kind"] == "map" and value.ndim != 2:
        raise ValueError("map requires a 2-D field (or an explicit history frame)")
    if panel["kind"] == "points":
        x, y = array(data, panel["x_key"]), array(data, panel["y_key"])
        if value.ndim != 1 or x.shape != value.shape or y.shape != value.shape:
            raise ValueError("point coordinates and field must be matching vectors")
    return value


def limits(values, panel):
    raw = np.concatenate([v.ravel() for v in values])
    percentile = float(panel.get("percentile", 100))
    if not 0 < percentile <= 100:
        raise ValueError("display percentile must be in (0,100]")
    if panel.get("symmetric", False):
        bound = float(np.percentile(np.abs(raw), percentile))
        low, high = -bound, bound
    else:
        tail = (100 - percentile) / 2
        low, high = map(float, np.percentile(raw, [tail, 100-tail]))
    # Keep zero/constant data honest and visible; do not invent a gradient.
    if low == high:
        pad = max(abs(low) * .01, 1e-15)
        low, high = low-pad, high+pad
    return low, high


def statistics(value, low=None, high=None):
    result = {"shape": list(value.shape), "min": float(value.min()),
              "max": float(value.max()), "l2": float(np.linalg.norm(value)),
              "count": int(value.size)}
    if low is not None:
        clipped = int(np.count_nonzero((value < low) | (value > high)))
        result.update(display_limits=[low, high], clipped_count=clipped,
                      clipped_fraction=clipped / value.size)
    return result


def safe_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("figure names must be simple file stems")
    return name


def draw_field(ax, data, panel, low, high):
    value = field(data, panel)
    cmap = panel.get("cmap", "RdBu_r" if panel.get("symmetric") else "viridis")
    if panel["kind"] == "map":
        extent = panel["extent"]
        if len(extent) != 4 or not np.isfinite(extent).all():
            raise ValueError("map extent must give four finite physical bounds")
        artist = ax.imshow(value, extent=extent, origin=panel.get("origin", "upper"),
                           aspect=panel.get("aspect", "equal"), cmap=cmap,
                           vmin=low, vmax=high, interpolation="nearest")
    else:
        artist = ax.scatter(array(data, panel["x_key"]), array(data, panel["y_key"]),
                            c=value, s=panel.get("point_size", 5), cmap=cmap,
                            vmin=low, vmax=high, linewidths=0, rasterized=True)
        ax.set_aspect(panel.get("aspect", "equal"))
        if panel.get("invert_y", False):
            ax.invert_yaxis()
    # Independent inset avoids unequal plot-area shrinkage from colourbars.
    cax = ax.inset_axes([1.025, 0, .035, 1])
    ax.figure.colorbar(artist, cax=cax, label=panel["units"])
    return artist, statistics(value, low, high)


def geometry(fig, axes, output, auditor=None):
    fig.canvas.draw()
    boxes = [list(ax.get_position().bounds) for ax in axes]
    result = {"plot_area_bounds_fraction": boxes, "figure_inches": list(fig.get_size_inches()),
              "tolerance_pt": 1.5, "external_alignment_audit": bool(auditor)}
    if auditor:
        spec = importlib.util.spec_from_file_location("figure_alignment_audit", auditor)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.require_matplotlib_panel_alignment(fig, json_out=str(output),
                                                 tolerance_pt=1.5, strict=True)
    else:
        output.write_text(json.dumps(result, indent=2)+"\n")
    return result


def render(data, config, output, alignment_auditor=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "figures": [], "animations": [],
                "transforms": "No smoothing, spatial interpolation, gradient normalization or array exclusion"}
    gallery = ["# Visual report", ""]
    names = [safe_name(item["name"]) for item in config.get("figures", [])]
    if len(names) != len(set(names)):
        raise ValueError("duplicate figure name would overwrite evidence")
    for item in config.get("figures", []):
        name = safe_name(item["name"]); panels = item["panels"]
        if not panels or len(panels) > 12:
            raise ValueError("provide between one and twelve panels per figure")
        columns = min(int(item.get("columns", 2)), len(panels))
        if columns < 1:
            raise ValueError("columns must be positive")
        rows = math.ceil(len(panels)/columns)
        fig, grid = plt.subplots(rows, columns, figsize=(3.8*columns, 3.5*rows), squeeze=False)
        fig.subplots_adjust(left=.10, right=.86, top=.83, bottom=.23,
                            wspace=.70, hspace=.80)
        groups = {}
        for index, panel in enumerate(panels):
            if panel["kind"] not in ("map", "points"):
                continue
            group = panel.get("scale_group", f"panel-{index}")
            signature = (panel["units"], panel.get("symmetric", False), panel.get("percentile", 100),
                         panel.get("cmap", "RdBu_r" if panel.get("symmetric") else "viridis"))
            if group in groups and groups[group][0] != signature:
                raise ValueError("shared scale requires identical units, symmetry and clipping")
            groups.setdefault(group, (signature, [], panel))[1].append(field(data, panel))
        bounds = {key: limits(v[1], v[2]) for key, v in groups.items()}
        records = []
        for index, panel in enumerate(panels):
            ax = grid.flat[index]; kind = panel["kind"]
            if kind in ("map", "points"):
                low, high = bounds[panel.get("scale_group", f"panel-{index}")]
                _, record = draw_field(ax, data, panel, low, high)
            elif kind == "bars":
                value = array(data, panel["key"])
                if value.ndim != 1 or len(panel["categories"]) != value.size:
                    raise ValueError("bar categories must match a vector")
                ax.bar(panel["categories"], value, color="#327C9C", width=.65)
                ax.axhline(0, color=".5", linewidth=.6)
                record = statistics(value)
            elif kind == "curves":
                record = {"series": []}
                for series in panel["series"]:
                    x, y = array(data, series["x"]), array(data, series["y"])
                    if x.ndim != 1 or x.shape != y.shape:
                        raise ValueError("curve x/y must be matching vectors")
                    if (panel.get("xscale") == "log" and np.any(x <= 0)) or (panel.get("yscale") == "log" and np.any(y <= 0)):
                        raise ValueError("log curves require positive data; document any display floor separately")
                    ax.plot(x, y, label=series["label"], linewidth=1.3)
                    record["series"].append({"x": statistics(x), "y": statistics(y)})
                ax.set_xscale(panel.get("xscale", "linear")); ax.set_yscale(panel.get("yscale", "linear"))
                ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.20),
                          ncol=min(2, len(panel["series"])), frameon=False)
                ax.grid(alpha=.15)
            else:
                raise ValueError(f"unknown panel kind: {kind}")
            ax.set_xlabel(panel.get("xlabel", "")); ax.set_ylabel(panel["ylabel"])
            ax.set_title(f"{chr(97+index)}  {textwrap.fill(panel['label'], 32)}", loc="left", fontsize=10, pad=10)
            records.append({"spec": panel, "raw_statistics": record})
        for ax in list(grid.flat)[len(panels):]:
            fig.delaxes(ax)
        fig.suptitle(textwrap.fill(item["title"], 85), y=.97, fontsize=12)
        layout = geometry(fig, list(grid.flat)[:len(panels)], output/f"{name}.alignment.json", alignment_auditor)
        fig.savefig(output/f"{name}.png", dpi=300)
        fig.savefig(output/f"{name}.pdf")
        fig.savefig(output/f"{name}.svg")
        plt.close(fig)
        manifest["figures"].append({"name": name, "caption": item["caption"],
                                    "panels": records, "layout": layout})
        gallery.extend([f"## {item['title']}", "", f"![{item['title']}]({name}.png)", "", item["caption"], ""])
    for item in config.get("animations", []):
        name = safe_name(item["name"]); history = array(data, item["key"])
        times = array(data, item["time_key"]); frames = [operator.index(i) for i in item["frames"]]
        if times.ndim != 1 or history.shape[0] != times.size or len(frames) < 2 or len(set(frames)) != len(frames) or min(frames) < 0 or max(frames) >= times.size:
            raise ValueError("animation needs distinct valid native frame indices and matching times")
        if item.get("fps", 8) <= 0:
            raise ValueError("animation fps must be positive")
        chosen = history[frames]; low, high = limits([chosen], item)
        fig, ax = plt.subplots(figsize=(7.6, 4.2)); fig.subplots_adjust(left=.12, right=.82, bottom=.16, top=.84)
        panel = dict(item, frame=frames[0]); artist, _ = draw_field(ax, data, panel, low, high)
        ax.set_xlabel(item["xlabel"]); ax.set_ylabel(item["ylabel"])
        def update(index):
            values = history[index]
            if item["kind"] == "map": artist.set_data(values)
            else: artist.set_array(values)
            ax.set_title(f"{item['label']} | t = {times[index]:.5g} {item['time_units']}", fontsize=11)
            return artist,
        update(frames[0])
        movie = FuncAnimation(fig, update, frames=frames, blit=False)
        movie.save(output/f"{name}.gif", writer=PillowWriter(fps=item.get("fps", 8)))
        plt.close(fig)
        manifest["animations"].append({"name": name, "spec": item,
                                        "selected_times": times[frames].tolist(),
                                        "raw_statistics": statistics(chosen, low, high)})
        gallery.extend([f"## {item['label']}", "", f"![{item['label']}]({name}.gif)", "", item["caption"], ""])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    (output/"README.md").write_text("\n".join(gallery))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-auditor", type=Path, help="optional installed rendered-alignment module")
    args = parser.parse_args()
    with np.load(args.arrays, allow_pickle=False) as data:
        report = render(data, json.loads(args.spec.read_text()), args.output, args.alignment_auditor)
    report["inputs"] = {"arrays_sha256": hashlib.sha256(args.arrays.read_bytes()).hexdigest(),
                        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                        "arrays": args.arrays.name, "spec": args.spec.name}
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"figures": len(report["figures"]), "animations": len(report["animations"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
