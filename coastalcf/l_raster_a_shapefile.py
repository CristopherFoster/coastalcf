# coastalcf/raster_a_shapefile.py
import os
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape as shp_from_geojson


def _tifs_in(path: str) -> List[str]:
    """Return a list of .tif files. If `path` is a file, return [path]."""
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".tif")]
    if path.lower().endswith(".tif"):
        return [path]
    raise FileNotFoundError(f"No .tif found at: {path}")


def l_raster_a_shapefile(
    input_path: str,
    output_dir: str,
    target_value: int = 1,
    *,
    output_suffix: str = "_filtered.shp",
    min_area_pixels: int = 0,
    quiet: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """
    Convert binary (or categorical) raster(s) to polygon shapefile(s) for a given target value.

    Parameters
    ----------
    input_path : str
        Path to a single .tif file OR a folder containing .tif files.
    output_dir : str
        Folder where shapefiles will be written. Created if it does not exist.
    target_value : int, default=1
        Pixel value to extract as polygons (e.g., 1 for water mask).
    output_suffix : str, default="_filtered.shp"
        Suffix appended to the base raster name for the output shapefile.
    min_area_pixels : int, default=0
        If > 0, drop polygons smaller than this area (measured in pixels).
        Uses pixel area = |xres * yres| to approximate a pixel-based filter.
    quiet : bool, default=False
        If True, suppress progress prints.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Mapping from input filename (basename) to (xres, yres) pixel size.
        Useful for later setting simplify/buffer tolerances.
    """
    os.makedirs(output_dir, exist_ok=True)
    pixel_sizes: Dict[str, Tuple[float, float]] = {}

    for tiff_path in _tifs_in(input_path):
        base = os.path.splitext(os.path.basename(tiff_path))[0]
        out_path = os.path.join(output_dir, base + output_suffix)

        with rasterio.open(tiff_path) as src:
            band = src.read(1)
            # Build a mask for the target_value
            mask = band == target_value
            xres, yres = src.res
            pixel_sizes[os.path.basename(tiff_path)] = (float(xres), float(yres))

            # Extract polygons for pixels matching target_value
            feats = (
                {"properties": {"value": int(v)}, "geometry": s}
                for s, v in shapes(band, mask=mask, transform=src.transform)
            )
            geoms = list(feats)

            if not geoms:
                if not quiet:
                    print(f"⚠️ No pixels with value {target_value} found in: {os.path.basename(tiff_path)}")
                continue

            gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

            # Optional: drop tiny polygons based on pixel area threshold
            if min_area_pixels > 0:
                pixel_area = abs(xres * yres)
                min_area_units = min_area_pixels * pixel_area
                gdf = gdf[gdf.geometry.area >= min_area_units]

            # Write shapefile
            gdf.to_file(out_path)
            if not quiet:
                print(f"✅ Shapefile saved: {out_path}")

    return pixel_sizes
