# do_thermal_noise_removal.py

import os
from esa_snappy import GPF, HashMap


def do_thermal_noise_removal(source):
    
    """
    Elimina el ruido térmico de las imágenes Sentinel-1.

    Args:
        source: Producto de SNAP como entrada (por ejemplo, recortado o calibrado).

    Returns:
        Producto de SNAP con el ruido térmico eliminado.

    Raises:
        ValueError: Si el producto de entrada es None.
    """
    # Validación de entrada: asegúrate de que el producto no sea nulo.
    if source is None:
        raise ValueError("El producto de entrada es None. Asegúrate de haber ejecutado correctamente el paso de recorte (subset).")

    print('\tEliminando ruido térmico...')

    # Crear el diccionario de parámetros. Este operador no requiere parámetros adicionales.
    parameters = HashMap()

    try:
        # Ejecutar el operador 'ThermalNoiseRemoval' de SNAP.
        output = GPF.createProduct('ThermalNoiseRemoval', parameters, source)
    except RuntimeError as e:
        # Manejar errores que puedan surgir durante el proceso.
        print("Error durante la eliminación de ruido térmico:", e)
        raise

    # Devolver el producto con el ruido térmico eliminado.
    return output