# speckleFiltering.py

from esa_snappy import GPF, HashMap

def speckleFiltering(calibrate, toPrint=True):
    """
    Aplica el filtro Lee para reducir el ruido speckle en un producto Sentinel-1.

    Args:
        calibrate: Producto de SNAP como entrada (debe estar calibrado).
        toPrint: Si es True, imprime los nombres de las bandas del resultado.

    Returns:
        Producto de SNAP con el filtro speckle aplicado.
    """
    parameters = HashMap()
    parameters.put('filter', 'Lee')                  # Filtro Lee (preserva bordes)
    parameters.put('filterSizeX', '5')                 # Tamaño de la ventana en X
    parameters.put('filterSizeY',  '5')                 # Tamaño de la ventana en Y
    parameters.put('dampingFactor',  '2')               # Factor de amortiguación (controla suavizado)
    parameters.put('edgeThreshold', 5000.0)          # Umbral para detección de bordes
    parameters.put('estimateENL', True)              # Estimar ENL automáticamente
    parameters.put('enl', 1.0)                       # ENL manual (usado si estimateENL=False)

    speckle = GPF.createProduct('Speckle-Filter', parameters, calibrate)

    if toPrint:
        print("\tBandas del producto filtrado:", list(speckle.getBandNames()))

    return speckle
