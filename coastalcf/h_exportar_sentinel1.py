# h_exportar_scaled_tiff.py

import os
import numpy as np
from esa_snappy import GPF, HashMap, jpy, ProductIO
from osgeo import gdal


# ───────────────────────── Helpers ─────────────────────────

def h_scale_to_8bit_0_244(input_tif, output_tif, nodata_in_values=(-9999, -999), set_nodata_out=255):
    """
    Scale a single-band GeoTIFF to 8-bit:
      - Valid data are mapped to [0, 244]
      - NoData is set to 255
      - Input NoData: NaN/Inf and any value in `nodata_in_values`

    Args:
        input_tif (str): Path to input GeoTIFF (float32).
        output_tif (str): Path to output GeoTIFF (uint8).
        nodata_in_values (tuple): Values to treat as NoData in the input.
        set_nodata_out (int): NoData value to set in the output (default 255).

    Returns:
        (vmin, vmax): Min/max of the valid input data used for scaling.

    Raises:
        RuntimeError: If the input cannot be opened.
        ValueError: If there are no valid pixels to scale.
    """
    src = gdal.Open(input_tif, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Could not open: {input_tif}")
    band = src.GetRasterBand(1)
    arr  = band.ReadAsArray().astype(np.float32)

    # Input NoData: NaN/Inf and explicit markers
    mask_nan   = ~np.isfinite(arr)
    mask_nd_in = np.zeros_like(arr, dtype=bool)
    for v in nodata_in_values:
        mask_nd_in |= (arr == v)

    valid = ~(mask_nan | mask_nd_in)
    if not np.any(valid):
        raise ValueError("No valid data to scale.")

    vmin = np.nanmin(arr[valid])
    vmax = np.nanmax(arr[valid])

    scaled = np.empty_like(arr, dtype=np.float32)
    if vmax == vmin:
        # Neutral value if constant image
        scaled[valid]  = 122.0
        scaled[~valid] = float(set_nodata_out)
    else:
        scaled[valid]  = (arr[valid] - vmin) / (vmax - vmin) * 244.0
        scaled[~valid] = float(set_nodata_out)

    # Safety: if 255 appears in valid pixels, move it to 244
    mask_255_valid = (scaled == 255) & valid
    scaled[mask_255_valid] = 244.0

    out_arr = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    out_arr[~valid] = set_nodata_out

    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(
        output_tif, src.RasterXSize, src.RasterYSize, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=IF_SAFER"]
    )
    out.SetGeoTransform(src.GetGeoTransform())
    out.SetProjection(src.GetProjection())

    out_band = out.GetRasterBand(1)
    out_band.WriteArray(out_arr)
    out_band.SetNoDataValue(set_nodata_out)
    out_band.FlushCache()

    out = None
    src = None
    return vmin, vmax


def h_exportar_sentinel1(snap_product, out_base_without_ext):
    """
    SNAP Product -> temporary GeoTIFF (float32) -> 8-bit (0–244 valid; 255 NoData).
    Returns (output_tif_path, vmin, vmax) from the scaling step.

    Notes:
        SNAP's ProductIO.writeProduct expects a base path without extension.

    Args:
        snap_product: SNAP Product instance to export.
        out_base_without_ext (str): Output base path without extension.

    Returns:
        (str, float, float): (output_tif, vmin, vmax)
    """
    # SNAP requires base without extension
    tmp_base = out_base_without_ext + "_tmp"
    ProductIO.writeProduct(snap_product, tmp_base, "GeoTIFF")
    tmp_tif = tmp_base + ".tif"

    out_tif = out_base_without_ext + ".tif"
    vmin, vmax = h_scale_to_8bit_0_244(
        tmp_tif, out_tif, nodata_in_values=(-9999, -999), set_nodata_out=255
    )

    # Cleanup
    try:
        os.remove(tmp_tif)
    except OSError:
        pass

    return out_tif, vmin, vmax
