"""The radar-wow chain from raw NEXRAD volumes to hourly rain totals on a 1 km grid.

Per volume: melting layer from QVPs (tracked, with the HRRR 0 C height as prior) -> brightband VPR correction
(Zhang & Qi 2010) -> for the lowest ``n_tilts`` tilts: beam blockage and NEXRAD dB correction (Bech et al. 2003,
Fulton et al. 1998; optional), KDP (Vulpiani et al. 2012), ZPHI attenuation with drop-size-adaptive alpha
(Wang et al. 2019), synthetic rain rate (Zhang et al. 2020) -> hybrid scan (lowest tilt < 50 % blocked) ->
1 km Cartesian grid. Hourly totals follow storm motion (Anagnostou & Krajewski 1999).

This is the configuration verified in radarwow-verify notebooks 01-05. Rates are 0 where there is no echo and
NaN where no tilt gives a valid rain estimate (above the melting layer, or blocked on every tilt).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr


def hourly_totals(site: str, start, end, cache_dir: str | None = None, n_tilts: int = 3, blockage: bool = True,
                  max_km: float = 150.0, res_km: float = 1.0, min_valid_minutes: float = 48.0) -> xr.Dataset:
    """Hourly rain (mm) for every full hour in [start, end) from NEXRAD ``site``; cached as netCDF in ``cache_dir``.

    Returns a Dataset with RAIN (time, y, x), coordinates x/y (km east/north of the radar) and lat/lon, and the
    per-volume ALPHA used for R(A); attrs hold the radar position.
    """
    from radarwow.attenuation import Testud2000ZPHI
    from radarwow.classification import MeltingLayerTracker, melting_layer_from_volume
    from radarwow.core import load_band, nexrad_aws_files, open_volume
    from radarwow.estimation import (ADAPTIVE_MIN_DPHI, Bech2003BeamBlockage, SyntheticRainRate, ZDRSlopeAlpha, ZhangQi2010VPR,
                                     advection_accumulate, below_melting_layer, copernicus_dem, fulton1998_correction_db,
                                     hybrid_scan, polar_to_cartesian, to_fixed_azimuths, unique_tilts)
    from radarwow.kdp import Vulpiani2012KDP
    from radarwow.verify import hrrr_freezing_level

    start, end = pd.Timestamp(start).floor("h"), pd.Timestamp(end).ceil("h")
    cache_dir = cache_dir or os.path.expanduser("~/.cache/radarwow")
    out_nc = os.path.join(cache_dir, "pipeline", f"{site}_{start:%Y%m%d%H}_{end:%Y%m%d%H}_t{n_tilts}{'b' if blockage else ''}.nc")
    if os.path.exists(out_nc):
        return xr.open_dataset(out_nc).load()

    files = nexrad_aws_files(site, start - pd.Timedelta(minutes=15), end + pd.Timedelta(minutes=5), cache_dir=f"{cache_dir}/nexrad")
    dt0 = open_volume(files[0][1], band="S")
    lat0, lon0, alt = (float(dt0[k].values) for k in ("latitude", "longitude", "altitude"))
    bb = (Bech2003BeamBlockage(dem=copernicus_dem(lat0, lon0, max_km + 10, f"{cache_dir}/dem"), site=(lat0, lon0, alt),
                               beamwidth_deg=0.95) if blockage else None)
    s_band = load_band("S")
    tracker, alpha_est, fz_cache = MeltingLayerTracker(), ZDRSlopeAlpha(), {}
    scans, alphas = [], []
    grid = None
    for t, f in files:
        try:
            dt = open_volume(f, band="S")
        except Exception:  # noqa: BLE001 - damaged archive file
            continue
        hr = pd.Timestamp(t).round("h")
        if hr not in fz_cache:
            fz_cache[hr] = (hrrr_freezing_level(hr, lat0, lon0, f"{cache_dir}/hrrr", max_km) - alt) / 1e3
        ml, _ = melting_layer_from_volume(dt)
        use, _ = tracker.update(t, ml, prior_top_km=fz_cache[hr])
        mlb = use.bottom_km if use else fz_cache[hr] - 0.5
        vdt = ZhangQi2010VPR(freezing_level_km=fz_cache[hr])(dt)
        fields, cbbs = [], []
        for i, (el, name) in enumerate(unique_tilts(vdt)[:n_tilts]):
            ds = vdt[name].to_dataset()
            ds = ds.isel(range=ds["range"].values <= (max_km + 2) * 1e3)
            if bb is not None:
                ds = bb(ds, band="S")
                cbb = ds["CBB"].values
                corr = fulton1998_correction_db(cbb)
            else:
                cbb, corr = np.zeros(ds["DBZH"].shape), np.zeros(ds["DBZH"].shape)
            below = below_melting_layer(ds, ml_bottom_km=mlb)
            if i == 0:
                alpha, _ = alpha_est.update(t, ds, mask=below & (cbb < 0.1))
                alphas.append((pd.Timestamp(t), alpha))
            dsc = ds.assign(DBZH=ds["DBZH_VPR"] + np.nan_to_num(corr))
            cfg = s_band.override(zphi_alpha=alpha, ra_min_dphi=ADAPTIVE_MIN_DPHI)
            rate = SyntheticRainRate(temperature_c=15)(Testud2000ZPHI()(Vulpiani2012KDP()(dsc, band="S"), band=cfg), band=cfg)["RATE"]
            rate = rate.copy(data=np.where(below & np.isfinite(corr), np.nan_to_num(rate.values), np.nan))
            cart = polar_to_cartesian(to_fixed_azimuths(rate), el, max_km=max_km, res_km=res_km)
            grid = cart if grid is None else grid
            fields.append(cart.values)
            cbbs.append(polar_to_cartesian(to_fixed_azimuths(rate.copy(data=cbb)), el, max_km=max_km, res_km=res_km).values)
        field, _ = hybrid_scan(list(zip(fields, cbbs)))
        scans.append((t, grid.copy(data=field.astype("float32"))))
    hours = pd.date_range(start, end, freq="h")[:-1]
    rain = []
    for h in hours:
        acc = advection_accumulate(scans, h, h + pd.Timedelta(hours=1))
        rain.append(np.where(acc["VALID_MINUTES"].values >= min_valid_minutes, acc["ACCUM"].values, np.nan))
    xx, yy = np.meshgrid(grid["x"].values, grid["y"].values)
    rain_ds = xr.Dataset(
        {"RAIN": (("time", "y", "x"), np.array(rain, dtype="float32"), {"units": "mm", "long_name": "rain in the hour starting at time"}),
         "ALPHA": (("volume_time",), np.array([a for _, a in alphas], dtype="float32"))},
        coords={"time": hours, "y": grid["y"].values, "x": grid["x"].values, "volume_time": [t for t, _ in alphas],
                "lat": (("y", "x"), lat0 + yy / 111.2), "lon": (("y", "x"), lon0 + xx / (111.2 * np.cos(np.deg2rad(lat0))))},
        attrs={"site": site, "radar_lat": lat0, "radar_lon": lon0, "radar_alt_m": alt, "n_tilts": n_tilts, "blockage": int(blockage),
               "chain": "QVP ML -> VPR -> blockage/hybrid scan -> KDP -> ZPHI (adaptive alpha) -> synthetic rate -> advection totals"})
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    rain_ds.to_netcdf(out_nc)
    return rain_ds
