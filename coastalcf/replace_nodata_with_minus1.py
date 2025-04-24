# replace_nodata_with_minus1.py

from esa_snappy import Product, ProductData, ProductUtils, ProductIO
import numpy as np

import numpy as np
import os

def replace_nodata_with_minus1(product, output_path, nodata_val=-9999.0, new_val=-1.0):
    """
    Reemplaza -9999 por -1 en todas las bandas de un producto esa_snappy,
    creando una copia editable y guardando como GeoTIFF.

    Args:
        product: Producto SNAP original (solo lectura).
        output_path: Ruta donde se guardará el producto corregido.
        nodata_val: Valor nodata a reemplazar.
        new_val: Valor nuevo con el que se reemplaza el nodata.

    Returns:
        Ruta del producto corregido guardado.
    """
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    band_names = product.getBandNames()

    # Crear un nuevo producto editable
    corrected = Product(product.getName() + "_corrected", product.getProductType(), width, height)
    #corrected.setGeoCoding(product.getGeoCoding())
    corrected.setSceneGeoCoding(product.getSceneGeoCoding())
    ProductUtils.copyMetadata(product, corrected)

    for band_name in band_names:
        band = product.getBand(band_name)
        print(f"Procesando banda: {band_name}")

        data = np.zeros(width * height, dtype=np.float32)
        band.readPixels(0, 0, width, height, data)
        data[data == nodata_val] = new_val

        # Crear nueva banda en el producto editable
        new_band = corrected.addBand(band_name, ProductData.TYPE_FLOAT32)
        new_band.writePixels(0, 0, width, height, data)
        new_band.setNoDataValue(new_val)
        new_band.setNoDataValueUsed(True)

    # Guardar como GeoTIFF
    ProductIO.writeProduct(corrected, output_path, 'GeoTIFF')

    print(f"✅ Producto corregido guardado en: {output_path}")
    return output_path

