# subset.py

import os
from esa_snappy import jpy, GPF

def subset(product, x, y, width, height, subset_bands=None, toPrint=True):
    """
    Recorta un producto Sentinel-1 a una región específica del ráster, opcionalmente con bandas seleccionadas.

    Args:
        product: Objeto SNAP (Product).
        x, y: Coordenadas de la esquina superior izquierda (en píxeles).
        width, height: Dimensiones del recorte (en píxeles).
        subset_bands (list): Lista de nombres de bandas a incluir (opcional).
        toPrint: Si es True, imprime las bandas resultantes.

    Returns:
        Producto SNAP recortado.
    """
    # Si no se especifica región ni bandas, no se aplica subset
    if None in (x, y, width, height):
        print("No se especificó región ni bandas para subset. Se devuelve el producto original.")
        return product
    
    HashMap = jpy.get_type('java.util.HashMap')
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    parameters = HashMap()

    parameters.put('copyMetadata', True)
    parameters.put('region', f"{x},{y},{width},{height}")

    if subset_bands:
        band_str = ','.join(subset_bands)
        parameters.put('sourceBands', band_str)

    subset_product = GPF.createProduct('Subset', parameters, product)

    if subset_product is None or subset_product.getSceneRasterWidth() == 0 or subset_product.getSceneRasterHeight() == 0:
        raise RuntimeError("El producto recortado no contiene datos válidos.")

    if toPrint:
        print(f"Bandas en el subset: {list(subset_product.getBandNames())}")

    return subset_product