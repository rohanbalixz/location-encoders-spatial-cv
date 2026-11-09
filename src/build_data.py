"""Rebuild RANGE-style evaluation tasks from public sources.

The RANGE eval CSVs are not public, so labels are rebuilt from:
  Biome / EcoRegion : RESOLVE Ecoregions 2017 (Dinerstein et al. 2017), the same source as RANGE
  Country           : Natural Earth 10m admin-0 (ADMIN)
  Temperature       : WorldClim 2.1 BIO1 annual mean temperature, 10' (RANGE uses Hooker et al. 2018)
  Elevation         : WorldClim 2.1 elevation, 10' (RANGE uses Rolf et al. 2021)
  Population        : GHS-POP 2020, area-averaged to 5' (~9 km), log1p (RANGE uses Rolf et al. 2021).
                      The first build used 30'' point samples; see the caveats in FINDINGS.md.
  California housing: sklearn fetch_california_housing (same 1990 census data as RANGE)
Global tasks share one set of points sampled uniformly on land (lat > -60).
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)
N_POINTS = 50_000
SEED = 0


def sample_land_points():
    land = gpd.read_file(next((RAW / "ne_land").rglob("ne_10m_land.shp")))
    rng = np.random.default_rng(SEED)
    pts = []
    while sum(len(p) for p in pts) < N_POINTS:
        n = 200_000
        lon = rng.uniform(-180, 180, n)
        lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))  # uniform on the sphere
        keep = lat > -60
        g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon[keep], lat[keep]), crs="EPSG:4326")
        hit = gpd.sjoin(g, land[["geometry"]], how="inner", predicate="within")
        hit = hit[~hit.index.duplicated()]
        pts.append(np.stack([hit.geometry.x.values, hit.geometry.y.values], 1))
    pts = np.concatenate(pts)[:N_POINTS]
    return pd.DataFrame({"lon": pts[:, 0], "lat": pts[:, 1]})


def join_polygons(df, shp, cols):
    poly = gpd.read_file(shp)[cols + ["geometry"]]
    g = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    poly = poly.to_crs("EPSG:4326")
    hit = gpd.sjoin(g, poly, how="left", predicate="within")
    hit = hit[~hit.index.duplicated()]
    return hit[cols].reindex(df.index)


def sample_raster(df, tif):
    with rasterio.open(tif) as src:
        vals = np.array([v[0] for v in src.sample(zip(df.lon.values, df.lat.values))], dtype=float)
        nod = src.nodata
    if nod is not None:
        vals[vals == nod] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    vals[np.abs(vals) > 1e30] = np.nan
    return vals


def sample_raster_avg(df, tif, factor):
    """Area-average the raster by `factor` (30'' -> 5' for GHS-POP), then sample.
    Point samples of 30'' population counts are too noisy (R^2 ~0.3 for every encoder)."""
    from rasterio.enums import Resampling
    with rasterio.open(tif) as src:
        arr = src.read(1, out_shape=(src.height // factor, src.width // factor),
                       resampling=Resampling.average).astype(np.float64)
        tr = src.transform * src.transform.scale(src.width / arr.shape[1], src.height / arr.shape[0])
    arr[arr < 0] = np.nan
    r, c = rasterio.transform.rowcol(tr, df.lon.values, df.lat.values)
    r = np.clip(np.array(r), 0, arr.shape[0] - 1)
    c = np.clip(np.array(c), 0, arr.shape[1] - 1)
    return arr[r, c]


def main():
    pts = sample_land_points()
    print("land points", len(pts))
    eco = join_polygons(pts, next((RAW / "ecoregions").rglob("*.shp")), ["BIOME_NAME", "ECO_NAME"])
    ctry = join_polygons(pts, next((RAW / "ne_countries").rglob("ne_10m_admin_0_countries.shp")), ["ADMIN"])
    temp = sample_raster(pts, next((RAW / "worldclim_bio").rglob("*bio_1.tif")))
    elev = sample_raster(pts, next((RAW / "worldclim_elev").rglob("*elev.tif")))
    pop = sample_raster_avg(pts, next((RAW / "ghs_pop").rglob("*.tif")), factor=10)

    def save(name, y):
        d = pts.copy()
        d["y"] = y
        d = d.dropna(subset=["y"])
        num = pd.api.types.is_numeric_dtype(d["y"])
        if not num:
            d = d[d["y"] != "N/A"]
            d["y"] = d["y"].astype(str)
        d.to_parquet(OUT / f"{name}.parquet")
        print(f"{name:12s} n={len(d):6d}", (f"classes={d.y.nunique()}" if not num else f"mean={d.y.mean():.2f} sd={d.y.std():.2f}"))

    save("biome", eco["BIOME_NAME"].values)
    save("ecoregion", eco["ECO_NAME"].values)
    save("country", ctry["ADMIN"].values)
    save("temperature", temp)
    save("elevation", elev)
    save("population", np.log1p(pop))

    from sklearn.datasets import fetch_california_housing
    ch = fetch_california_housing(data_home=str(RAW / "sklearn"), as_frame=True).frame
    d = pd.DataFrame({"lon": ch.Longitude.values, "lat": ch.Latitude.values, "y": ch.MedHouseVal.values})
    d.to_parquet(OUT / "housing.parquet")
    print(f"{'housing':12s} n={len(d):6d} mean={d.y.mean():.2f}")


if __name__ == "__main__":
    main()
