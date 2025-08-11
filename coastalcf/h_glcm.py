# glcmOp.py

from esa_snappy import GPF, jpy

class h_glcm:
    """
    Clase para calcular métricas de textura GLCM utilizando el operador GLCM de SNAP en imágenes Sentinel-1.
    """

    def __init__(self):
        """
        Inicializa glcmOp:
        - Carga la clase Java HashMap para la gestión de parámetros.
        - Carga los operadores de SNAP.
        """
        self.HashMap = jpy.get_type('java.util.HashMap')
        GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

    def glcm(self, src_data, para_dict):
        """
        Calcula el GLCM para imágenes Sentinel-1.

        Args:
            src_data: Producto de SNAP como entrada.
            para_dict: Diccionario de parámetros para el operador GLCM.

        Returns:
            Producto de SNAP con el GLCM calculado.
        """
        func_name = 'GLCM'
        parameters = self.HashMap()

        # Parámetros predeterminados para el operador GLCM.
        parameters.put('angleStr', 'ALL')  # Ángulos considerados (todos)
        parameters.put('displacement', '1')  # Desplazamiento de píxeles
        parameters.put('outputMean', True)  # Calcular la media de las métricas
        parameters.put('quantizationLevelsStr', '128')  # Niveles de cuantización
        parameters.put('quantizerStr', 'Probabilistic Quantizer')  # Tipo de cuantificador
        parameters.put('windowSizeStr', '5x5')  # Tamaño de la ventana de análisis
        parameters.put('outputASM', False)              # Suma de cuadrados de la matriz: energía angular o uniformidad.
        parameters.put('outputContrast', False)         # Contraste local entre el píxel y sus vecinos.
        parameters.put('outputCorrelation', False)      # Medida de correlación entre la imagen y su desplazamiento.
        parameters.put('outputDissimilarity', False)    # Disimilitud: diferencia entre pares de píxeles.
        parameters.put('outputEnergy', False)           # Energía: suma de los cuadrados de los valores de la matriz.
        parameters.put('outputEntropy', False)          # Entropía: aleatoriedad o complejidad del patrón de textura.
        parameters.put('outputHomogeneity', False)      # Homogeneidad: similitud entre los valores de la matriz.
        parameters.put('outputMAX', False)              # Valor máximo de la matriz de co-ocurrencia.
        parameters.put('outputVariance', False)         # Varianza: dispersión de los valores de la matriz.
        #parameters.put('Mask out areas without elevation', False)
        parameters.put('nodataValueAtSea', False)


        # Actualiza los parámetros con los valores proporcionados en para_dict.
        print(f'{func_name}: Estableciendo parámetros...')
        for key, value in para_dict.items():
            parameters.put(key, value)

        print(f'{func_name}: Creando producto GLCM...')
        textura = GPF.createProduct('GLCM', parameters, src_data)
        print(textura)  # Imprime los detalles del producto generado.

        return textura