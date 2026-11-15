"""Run radar-wow from a TOML file: radars -> hourly totals -> (mosaic) -> (gauges, merging) -> files and a report.

Example (see ``examples/*.toml``)::

    [run]
    name = "ktlx_20220505"
    start = "2022-05-05 07:00"
    end = "2022-05-05 11:00"
    output_dir = "~/radarwow_runs"

    [[radar]]
    site = "KTLX"
    source = "nexrad"        # nexrad | fmi | ideam
    n_tilts = 1
    blockage = false

    [mosaic]                 # used when there is more than one radar
    mode = "max"             # max | weighted

    [gauges]                 # optional
    source = "hads"          # hads | fmi | ideam
    networks = ["OK_DCP"]    # hads only
    utc_offset_h = 0         # ideam: 5 (its times are local, UTC-5)
    qc = "gross"             # basic | gross

    [merge]                  # optional; needs [gauges]
    method = "conditional merging"

Every key is checked before anything runs, so a typo fails at once instead of after an hour of processing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

SOURCES = ("nexrad", "fmi", "ideam")
GAUGES = ("hads", "fmi", "ideam")
SCHEMA = {
    "run": {"name": str, "start": str, "end": str, "output_dir": str},
    "radar": {"site": str, "source": str, "n_tilts": int, "blockage": bool, "rate": str},
    "mosaic": {"mode": str, "res_km": float},
    "gauges": {"source": str, "networks": list, "utc_offset_h": float, "qc": str, "radius_km": float},
    "merge": {"method": str},
}
REQUIRED = {"run": ("name", "start", "end"), "radar": ("site", "source")}


class ConfigError(ValueError):
    pass


def _check(section: str, d: dict):
    allowed = SCHEMA[section]
    unknown = set(d) - set(allowed)
    if unknown:
        raise ConfigError(f"[{section}]: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    for k in REQUIRED.get(section, ()):
        if k not in d:
            raise ConfigError(f"[{section}]: missing required key '{k}'")
    for k, v in d.items():
        typ = allowed[k]
        if typ is float and isinstance(v, int) and not isinstance(v, bool):
            continue
        if not isinstance(v, typ):
            raise ConfigError(f"[{section}] {k} = {v!r}: expected {typ.__name__}")


@dataclass
class Pipeline:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    radars: list
    output_dir: str = "~/radarwow_runs"
    mosaic: dict = field(default_factory=lambda: {"mode": "max", "res_km": 1.0})
    gauges: dict | None = None
    merge: dict | None = None

    @classmethod
    def from_toml(cls, path: str) -> "Pipeline":
        import tomllib

        with open(os.path.expanduser(path), "rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, cfg: dict) -> "Pipeline":
        from radarwow.merging import METHODS

        unknown = set(cfg) - set(SCHEMA)
        if unknown:
            raise ConfigError(f"unknown section(s) {sorted(unknown)}; allowed: {sorted(SCHEMA)}")
        if "run" not in cfg or "radar" not in cfg:
            raise ConfigError("need [run] and at least one [[radar]]")
        _check("run", cfg["run"])
        radars = cfg["radar"] if isinstance(cfg["radar"], list) else [cfg["radar"]]
        for r in radars:
            _check("radar", r)
            if r["source"] not in SOURCES:
                raise ConfigError(f"[[radar]] source = {r['source']!r}: one of {SOURCES}")
            if r.get("rate", "synthetic") not in ("synthetic", "marshall_palmer"):
                raise ConfigError(f"[[radar]] rate = {r['rate']!r}: 'synthetic' or 'marshall_palmer'")
        mosaic = {"mode": "max", "res_km": 1.0, **cfg.get("mosaic", {})}
        _check("mosaic", mosaic)
        if mosaic["mode"] not in ("max", "weighted"):
            raise ConfigError(f"[mosaic] mode = {mosaic['mode']!r}: 'max' or 'weighted'")
        gauges = cfg.get("gauges")
        if gauges is not None:
            gauges = {"qc": "gross", "utc_offset_h": 0.0, "radius_km": 145.0, **gauges}
            _check("gauges", gauges)
            if gauges.get("source") not in GAUGES:
                raise ConfigError(f"[gauges] source = {gauges.get('source')!r}: one of {GAUGES}")
            if gauges["source"] == "hads" and not gauges.get("networks"):
                raise ConfigError("[gauges] source = 'hads' needs networks, e.g. ['OK_DCP']")
            if gauges["qc"] not in ("basic", "gross"):
                raise ConfigError(f"[gauges] qc = {gauges['qc']!r}: 'basic' or 'gross'")
        merge = cfg.get("merge")
        if merge is not None:
            _check("merge", merge)
            if gauges is None:
                raise ConfigError("[merge] needs a [gauges] section")
            if merge.get("method") not in METHODS:
                raise ConfigError(f"[merge] method = {merge.get('method')!r}: one of {sorted(METHODS)}")
        run = cfg["run"]
        start, end = pd.Timestamp(run["start"]), pd.Timestamp(run["end"])
        if not start < end:
            raise ConfigError("[run] start must be before end")
        return cls(name=run["name"], start=start, end=end, radars=radars, output_dir=run.get("output_dir", "~/radarwow_runs"),
                   mosaic=mosaic, gauges=gauges, merge=merge)

    # -- running ----------------------------------------------------------------------------------------
    def describe(self) -> str:
        lines = [f"{self.name}: {self.start} to {self.end}"]
        lines += [f"  radar {r['site']} ({r['source']}, {r.get('n_tilts', 3)} tilts, blockage {r.get('blockage', True)}, "
                  f"{r.get('rate', 'synthetic')})" for r in self.radars]
        if len(self.radars) > 1:
            lines.append(f"  mosaic: {self.mosaic['mode']} quality, {self.mosaic['res_km']} km")
        if self.gauges:
            lines.append(f"  gauges: {self.gauges['source']} ({self.gauges['qc']} QC)")
        if self.merge:
            lines.append(f"  merge: {self.merge['method']} (scored by leave-one-out)")
        return "\n".join(lines)

    def run(self, cache_dir: str | None = None) -> dict:
        """Process everything; write netCDF files and ``report.md`` to output_dir/name. Returns the paths and scores."""
        from radarwow.mosaic import mosaic_hourly

        from .chain import hourly_totals

        out = os.path.join(os.path.expanduser(self.output_dir), self.name)
        os.makedirs(out, exist_ok=True)
        rads = [hourly_totals(r["site"], self.start, self.end, cache_dir=cache_dir, n_tilts=r.get("n_tilts", 3),
                              blockage=r.get("blockage", True), source=r["source"], rate=r.get("rate", "synthetic"))
                for r in self.radars]
        files = {}
        for r, ds in zip(self.radars, rads):
            p = os.path.join(out, f"radar_{r['site']}.nc")
            ds.to_netcdf(p)
            files[r["site"]] = p
        # one field on a km grid for gauges and merging
        if len(rads) == 1:
            rain = rads[0]
            lat0, lon0 = rain.attrs["radar_lat"], rain.attrs["radar_lon"]
            x, y, field_of = rain["x"].values, rain["y"].values, (lambda t: rain["RAIN"].sel(time=t).values)
            to_km = lambda la, lo: ((lo - lon0) * 111.2 * np.cos(np.deg2rad(lat0)), (la - lat0) * 111.2)  # noqa: E731
            times = rain["time"].values
        else:
            mos = mosaic_hourly(rads, mode=self.mosaic["mode"], res_km=self.mosaic["res_km"])
            p = os.path.join(out, "mosaic.nc")
            mos.to_netcdf(p)
            files["mosaic"] = p
            lat0, lon0 = float(mos["lat"].mean()), float(mos["lon"].mean())
            x = (mos["lon"].values - lon0) * 111.2 * np.cos(np.deg2rad(lat0))
            y = (mos["lat"].values - lat0) * 111.2
            field_of = lambda t: mos["RAIN"].sel(time=t).values  # noqa: E731
            to_km = lambda la, lo: ((lo - lon0) * 111.2 * np.cos(np.deg2rad(lat0)), (la - lat0) * 111.2)  # noqa: E731
            times = mos["time"].values
        report = [f"# {self.name}", "", "```", self.describe(), "```", ""]
        scores = None
        if self.gauges:
            g = self._gauges(rads, cache_dir)
            g["gx"], g["gy"] = to_km(g["lat"].values, g["lon"].values)
            scores, g = self._score(g, times, x, y, field_of)
            report += [f"{g.station.nunique()} gauges; quality control: " + ", ".join(f"{k}: {v}" for k, v in g["qc"].value_counts().items()), "",
                       "Leave-one-out at the gauges (bias = estimate / gauge; MAE in mm):", "", scores.round(2).to_markdown(index=False), ""]
            if self.merge:
                files["merged"] = self._write_merged(out, g, times, x, y, field_of, rads)
        report += ["Files:", ""] + [f"- {k}: `{v}`" for k, v in files.items()]
        with open(os.path.join(out, "report.md"), "w") as f:
            f.write("\n".join(report) + "\n")
        files["report"] = os.path.join(out, "report.md")
        return dict(files=files, scores=scores)

    def _gauges(self, rads, cache_dir):
        from radarwow.verify import fetch_fmi, fetch_hads, fetch_ideam

        cache = os.path.join(cache_dir or os.path.expanduser("~/.cache/radarwow"), "gauges")
        gc = self.gauges
        parts = []
        for r in rads:
            la, lo = r.attrs["radar_lat"], r.attrs["radar_lon"]
            if gc["source"] == "hads":
                parts.append(fetch_hads(gc["networks"], la, lo, gc["radius_km"], self.start, self.end, cache))
            elif gc["source"] == "fmi":
                parts.append(fetch_fmi(la, lo, gc["radius_km"], self.start, self.end, cache))
            else:
                parts.append(fetch_ideam(la, lo, gc["radius_km"], self.start, self.end, cache, utc_offset_h=gc["utc_offset_h"]))
        return pd.concat(parts).drop_duplicates(["station", "time"]).reset_index(drop=True)

    def _score(self, g, times, x, y, field_of):
        from radarwow.merging import leave_one_out, radar_at
        from radarwow.verify import basic_qc, gross_error_qc

        g["radar"] = np.nan
        for t, idx in g.groupby("time").groups.items():
            if np.datetime64(t) in times:
                g.loc[idx, "radar"] = radar_at(field_of(np.datetime64(t)), x, y, g.loc[idx, "gx"].values, g.loc[idx, "gy"].values)
        q = basic_qc(g, radar_col="radar")
        if self.gauges["qc"] == "gross":
            q = gross_error_qc(q, radar_col="radar")
        ok = q[q["qc"] == "ok"]
        methods = ["radar only"] + ([self.merge["method"]] if self.merge and self.merge["method"] != "radar only" else [])
        rows = []
        for m in methods:
            est, obs = [], []
            for t, gt in ok.groupby("time"):
                est += list(leave_one_out(m, field_of(np.datetime64(t)), x, y, gt.gx.values, gt.gy.values, gt.gauge_mm.values))
                obs += list(gt.gauge_mm.values)
            e, o = np.array(est), np.array(obs)
            k = np.isfinite(e)
            rows.append(dict(estimate=m, n=int(k.sum()), bias=e[k].sum() / max(o[k].sum(), 1e-9), MAE=np.mean(np.abs(e[k] - o[k])),
                             corr=np.corrcoef(e[k], o[k])[0, 1] if k.sum() > 2 else np.nan))
        return pd.DataFrame(rows), q

    def _write_merged(self, out, g, times, x, y, field_of, rads):
        from radarwow.merging import merge

        ok = g[g["qc"] == "ok"]
        merged = []
        for t in times:
            gt = ok[ok["time"] == pd.Timestamp(t)]
            f = field_of(t)
            merged.append(merge(self.merge["method"], f, x, y, gt.gx.values, gt.gy.values, gt.gauge_mm.values) if len(gt) >= 3 else f)
        ds = xr.Dataset({"RAIN": (("time", "y", "x"), np.array(merged, dtype="float32"), {"units": "mm", "method": self.merge["method"]})},
                        coords={"time": times, "y": y, "x": x})
        p = os.path.join(out, "merged.nc")
        ds.to_netcdf(p)
        return p
