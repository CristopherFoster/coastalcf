# radiometricCalibration.py

from esa_snappy import GPF, HashMap

def radiometricCalibration(subset):
    """
    Aplica calibración radiométrica a un producto Sentinel-1.

    Args:
        subset: Producto de SNAP como entrada (normalmente ya recortado).

    Returns:
        Objeto Product de SNAP con la calibración radiométrica aplicada.
    """
    # Crear diccionario de parámetros para el operador de calibración
    parameters = HashMap()
    parameters.put('auxFile', 'Latest Auxiliary File')       # Utiliza el archivo auxiliar más reciente
    parameters.put('outputBetaBand', True)                   # Genera la banda beta0 (relación señal/ruido)
    parameters.put('selectedPolarisations', 'VH')            # Selecciona la polarización VH

    print("\tAplicando calibración radiométrica...")

    # Ejecutar el operador de calibración de SNAP
    calibrated = GPF.createProduct('Calibration', parameters, subset)

    return calibrated