# perform_terrain_flattening.py

from esa_snappy import GPF, HashMap, jpy

def g_nivelacion_terreno(multilook):
    """
    Aplica la nivelación del terreno (terrain flattening) a imágenes Sentinel-1.
    Este proceso corrige la distorsión causada por la topografía en los valores de retrodispersión SAR.

    Args:
        multilook: Producto de SNAP como entrada (después del multilooking).

    Returns:
        Producto de SNAP con la corrección de nivelación del terreno aplicada.
    """
    
    if multilook is None:
        raise ValueError("El producto de entrada es None. Asegúrate de haber ejecutado el paso de multilooking correctamente.")

    # Crear diccionario de parámetros para el operador
    params = HashMap()

    # Modelo digital de elevación (DEM) utilizado. El SRTM de 1 segundo tiene una resolución de ~30 m.
    params.put('demName', 'SRTM 1Sec HGT')

    # Método de remuestreo del DEM: bicúbico para suavidad.
    params.put('demResamplingMethod', 'BICUBIC_INTERPOLATION')

    # Factor de sobremuestreo para mejorar el alineamiento geométrico.
    params.put('oversamplingMultiple', 1.5)

    # Solapamiento adicional entre tiles para evitar artefactos en bordes.
    params.put('additionalOverlap', 0.3)
    
    # Desactiva el enmascarado de áreas sin datos de elevación
    params.put('nodataValueAtSea', False)


    print('\tAplicando nivelación del terreno...')

    # Ejecutar el operador 'Terrain-Flattening'
    terrain = GPF.createProduct('Terrain-Flattening', params, multilook)
    
    return terrain
