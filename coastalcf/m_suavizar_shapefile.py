# coastalcf/suavizado_de_shapefile.py
import os
from typing import Optional
import geopandas as gpd
from shapely.ops import unary_union, linemerge

def m_suavizar_shapefile(
    folder_path: str,
    tolerance: float,
    buffer_val: float,
    *,
    input_suffix: str = "_filtered.shp",
    output_suffix: str = "_smoothed_line.shp",
    dissolve: bool = False,
    quiet: bool = False,
) -> None:
    """
    Smooth polygon shapefiles (simplify + optional in/out buffer) and export as polylines.

    Parameters
    ----------
    folder_path : str
        Folder containing input shapefiles (typically polygons).
    tolerance : float
        Simplification tolerance (map units).
    buffer_val : float
        Buffer distance used as a smooth/regularization step.
        Polygons are buffered by +buffer_val then -buffer_val. Use 0 to skip.
    input_suffix : str, default="_filtered.shp"
        Process only files ending with this suffix.
    output_suffix : str, default="_smoothed_line.shp"
        Suffix for output line shapefile names.
    dissolve : bool, default=False
        If True, dissolve all features to a single merged line per file.
    quiet : bool, default=False
        If True, suppress progress prints.
    """
    for file in os.listdir(folder_path):
        if not file.lower().endswith(".shp"):
            continue
        if input_suffix and not file.endswith(input_suffix):
            continue
        if file.endswith(output_suffix):
            continue

        shp_path = os.path.join(folder_path, file)
        if not quiet:
            print(f"🌀 Smoothing and converting to line: {file}")

        gdf = gpd.read_file(shp_path)

        # Fix invalids early
        gdf = gdf[~gdf.geometry.is_empty]
        gdf["geometry"] = gdf["geometry"].buffer(0)

        # 1) simplify
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=tolerance, preserve_topology=True)

        # 2) optional buffer smooth (+/-)
        if buffer_val != 0:
            gdf["geometry"] = gdf["geometry"].buffer(buffer_val).buffer(-buffer_val)

        # Convert polygon(s) to boundary line(s)
        lines = gdf.boundary
        lines = lines[~lines.is_empty]

        if dissolve:
            # Merge to a single line feature per file
            merged = linemerge(unary_union(lines.values))
            out_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[merged], crs=gdf.crs)
        else:
            out_gdf = gpd.GeoDataFrame(gdf.drop(columns=gdf.columns.difference(["geometry"], sort=False)), geometry=lines, crs=gdf.crs)

        # Output name
        base = file[:-4]
        new_filename = base + output_suffix
        new_path = os.path.join(folder_path, new_filename)

        # Save as ESRI Shapefile
        out_gdf.to_file(new_path)
        if not quiet:
            print(f"✅ Saved as line: {new_path}")

