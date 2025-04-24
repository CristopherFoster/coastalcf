# generate_shapefile_path.py

import os
from .get_sentinel_fecha_id import get_sentinel_fecha_id


def generate_output_path(sentinel_path, output_dir, suffix, extension):
    """
    Genera la ruta de salida para un archivo usando el ID Sentinel-1.

    Args:
        sentinel_path (str): Ruta del producto Sentinel-1.
        output_dir (str): Carpeta donde se guardará el archivo.
        suffix (str): Sufijo personalizado (por ejemplo: "_TIF", "_shp").
        extension (str): Extensión del archivo (por ejemplo: "tif", "shp").

    Returns:
        str: Ruta completa del archivo de salida.
    """
    sentinel_id = get_sentinel_fecha_id(sentinel_path)
    return os.path.join(output_dir, f"{sentinel_id}{suffix}.{extension}")



def generate_shapefile_path(sentinel_path, output_dir):
    """Genera la ruta de salida para el shapefile."""
    return generate_output_path(sentinel_path, output_dir, "_shp", "shp")
