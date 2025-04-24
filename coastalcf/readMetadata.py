# readMetadata.py

import os
from esa_snappy import jpy, ProductIO

def readMetadata(sentinel_1_path, toPrint=True):
    """
    Lee los metadatos de un producto GRD de Sentinel-1.

    Args:
        sentinel_1_path: Ruta al directorio .SAFE del producto Sentinel-1.
        toPrint: Si es True, imprime la información de los metadatos.

    Returns:
        Un objeto Product de SNAP que contiene los metadatos del producto Sentinel-1.
    """
    sentinel_1_metadata = "manifest.safe"  # Nombre del archivo de metadatos.
    s1prd = f"{sentinel_1_path}.SAFE\{sentinel_1_metadata}"  # Ruta completa al archivo de metadatos.
    
    # Cargar el lector de productos para Sentinel-1.
    reader = ProductIO.getProductReader("SENTINEL-1")

    # Leer los metadatos.
    product = reader.readProductNodes(s1prd, None)

    # Extraer la información relevante de los metadatos.
    width, height = product.getSceneRasterWidth(), product.getSceneRasterHeight()
    name = product.getName()
    band_names = list(product.getBandNames())

    # Imprimir los metadatos si se solicita.
    if toPrint:
        print(f"Producto: {name}, {width} x {height} píxeles")
        print(f"Bandas:   {band_names}")

    return product
