# replace_nodata_with_minus1.py

from esa_snappy import Product, ProductData, ProductUtils, ProductIO
import numpy as np

import numpy as np
import os

def replace_nodata_with_minus1(product, output_path, nodata_val=-9999.0, new_val=-1.0):
    """
    Crea un nuevo producto con las bandas corregidas: reemplaza -9999 por -1 y lo guarda como GeoTIFF.
    """
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    band_names = product.getBandNames()

    # Crear nuevo producto editable desde cero
    corrected = Product(product.getName() + "_corrected", product.getProductType(), width, height)
    corrected.setSceneGeoCoding(product.getSceneGeoCoding())
    ProductUtils.copyMetadata(product, corrected)

    for band_name in band_names:
        band_src = product.getBand(band_name)

        # Leer datos
        data = np.zeros(width * height, dtype=np.float32)
        band_src.readPixels(0, 0, width, height, data)
        data[data == nodata_val] = new_val

        # Crear nueva banda en el producto corregido
        band_new = corrected.addBand(band_name, ProductData.TYPE_FLOAT32)
        band_new.writePixels(0, 0, width, height, data)
        band_new.setNoDataValue(new_val)
        band_new.setNoDataValueUsed(True)

    # Guardar como GeoTIFF
    if not output_path.lower().endswith('.tif'):
        output_path += '.tif'

    ProductIO.writeProduct(corrected, output_path, 'GeoTIFF')
    print(f"✅ Producto corregido guardado como: {output_path}")
    return output_path

