# get_sentinel_fecha_id.py

import os

def a_sentinel1_id(sentinel_path):
    """
    Extrae la fecha (AAAAMMDD) y el ID único del producto Sentinel-1
    y los concatena en el formato: '20240121_8E8F'.

    Args:
        sentinel_path (str): Ruta al producto Sentinel-1.

    Returns:
        str: Cadena con formato 'fecha_id', por ejemplo: '20240121_8E8F'.
    """
    base_name = os.path.basename(sentinel_path)
    parts = base_name.split('_')

    sentinel_id = parts[-1] if len(parts) > 1 else base_name

    fecha = None
    for part in parts:
        if len(part) >= 15 and part[:8].isdigit():
            fecha = part[:8]
            break

    if fecha is None:
        raise ValueError(f"No se pudo extraer la fecha del nombre: {base_name}")

    return f"{fecha}_{sentinel_id}"
