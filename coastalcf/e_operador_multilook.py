# perform_multilook.py

from esa_snappy import GPF, jpy, HashMap


def e_operador_multilook(calibration, range_look_number=1, azimuth_look_number=1, 
                      pixel_size_m=10, output_intensity=False, toPrint=True):
    
    """
    Aplica el operador Multilook para reducir el ruido speckle en imágenes Sentinel-1.

    Args:
        calibration: Producto de SNAP como entrada (debe estar calibrado previamente).
        range_look_number: Número de looks en dirección de rango (horizontal). Por defecto: 1.
        azimuth_look_number: Número de looks en dirección azimutal (vertical). Por defecto: 1.
        pixel_size_m: Tamaño de píxel en metros en la salida. Por defecto: 10.
        output_intensity: Si True, convierte la salida a intensidad. Por defecto: False.
        toPrint: Si True, imprime mensajes informativos.

    Returns:
        Producto de SNAP con multilooking aplicado.

    Raises:
        ValueError: Si el producto de entrada es None.
    """
    
    if calibration is None:
        raise ValueError("El producto de entrada es None. Asegúrate de haber ejecutado la calibración correctamente.")

    # Cargar tipo Integer de Java
    Integer = jpy.get_type('java.lang.Integer')

    # Crear diccionario de parámetros para el operador
    params = HashMap()
    params.put('nRgLooks', Integer(range_look_number))      # Número de looks en rango
    params.put('nAzLooks', Integer(azimuth_look_number))    # Número de looks en azimut
    params.put('mGRSquarePixel', Integer(pixel_size_m))     # Tamaño de píxel en metros
    params.put('outputIntensity', output_intensity)         # Convertir a intensidad si se desea

    if toPrint:
        print(f'\tAplicando Multilooking: RgLooks={range_look_number}, AzLooks={azimuth_look_number}, '
              f'Pixel={pixel_size_m}m, Intensidad={output_intensity}')

    # Ejecutar el operador Multilook
    multilook = GPF.createProduct('Multilook', params, calibration)

    return multilook