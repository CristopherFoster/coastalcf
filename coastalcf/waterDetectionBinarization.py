import os
from esa_snappy import GPF, HashMap, jpy
import numpy as np
import math

# Métodos de umbralización
from skimage.filters import (
    threshold_otsu,
    threshold_niblack,
    threshold_sauvola,
    threshold_li
)

from .get_sentinel_fecha_id import get_sentinel_fecha_id
from .wbplots import plot_binary_map, plot_histogram
from .plot_thresholds import plot_thresholds  




def waterDetectionBinarization(
    textura,
    sentinel_1_path,
    output_directory,
    threshold_method="sauvola",
    window_size=31,
    k=0.2,     # extra param para Niblack/Sauvola
    r=None     # extra param para Sauvola
):
    """
    Detecta cuerpos de agua utilizando diferentes métodos de umbralización 
    (Sauvola, Otsu, Niblack, Li) y genera una máscara binaria a partir de 
    la banda 'Gamma0_VH_GLCMMean'.
    
    Args:
        textura: Producto SNAP que contiene la banda 'Gamma0_VH_GLCMMean'.
        sentinel_1_path (str): Ruta del archivo Sentinel.
        output_directory (str): Directorio para guardar resultados.
        threshold_method (str): Método de umbralización a usar. 
                                Puede ser "sauvola", "otsu", "niblack" o "li".
                                Por defecto, "sauvola".
        window_size (int): Tamaño de ventana para métodos locales como 
                           Sauvola o Niblack. Por defecto, 21.
    
    Returns:
        Producto SNAP con una nueva banda binaria llamada 'flood'.
    """
    sentinel_id = get_sentinel_fecha_id(sentinel_1_path)
    print("\n--- Iniciando detección de agua ---")
    print(f"Método de umbralización: {threshold_method}")

    # Verificar que la banda requerida esté presente en el producto
    available_bands = list(textura.getBandNames())
    print(f"Bandas disponibles: {available_bands}")

    if 'Gamma0_VH_GLCMMean' not in available_bands:
        raise RuntimeError("La banda 'Gamma0_VH_GLCMMean' no está disponible en el producto.")

    # Convertir la banda a decibeles (dB)
    paramToDB = HashMap()
    paramToDB.put('sourceBands', 'Gamma0_VH_GLCMMean')

    try:
        print("Convirtiendo a escala de decibeles (dB)...")
        img1 = GPF.createProduct("LinearToFromdB", paramToDB, textura)
        img2 = img1.getBand('Gamma0_VH_GLCMMean_db')
        print("Conversión a dB completada.")
    except Exception as e:
        raise RuntimeError("Error al convertir la banda a dB.") from e

    # Leer datos de la banda en un array de NumPy
    print("Lectura de datos de la banda en un array de NumPy")
    w, h = img2.getRasterWidth(), img2.getRasterHeight()
    img1_data = np.zeros(w * h, np.float32)
    img2.readPixels(0, 0, w, h, img1_data)

    # Estadísticas de la imagen en dB
    min_val, max_val = np.min(img1_data), np.max(img1_data)
    print(f"Rango de valores en dB: Min={min_val:.2f}, Max={max_val:.2f}")

    # Reshape del array 1D a 2D
    img1_data = img1_data.reshape((h, w))

    # Marcar valores inválidos (<= -1000) como NaN
    img1_data_masked = np.where(img1_data <= -1000, np.nan, img1_data)

    # Rellenar los NaN con un valor fijo o con el mínimo, a tu criterio.
    # Ejemplo: aquí usamos -1 para no distorsionar estadísticas.
    img1_valid = np.nan_to_num(img1_data_masked, nan=-1)
    
    
    print("Calcular el umbral según el método")
    # ------------------------------------------------------------------
    # A) Calcular el umbral según el método
    # ------------------------------------------------------------------
    if threshold_method.lower() == "sauvola":
        # Devuelve una "imagen" de umbrales locales.
        #https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola
        thresh_local = threshold_sauvola(img1_valid, window_size=window_size, k=k, r=r)
        # Usamos la media de todos los valores locales como umbral global:
        threshold_val = np.mean(thresh_local)
        print(f"(SAUVOLA) Umbral promedio: {threshold_val:.3f}")

    elif threshold_method.lower() == "niblack":
        # Similar a Sauvola, cada píxel tiene su umbral local
        #https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_niblack
        thresh_local = threshold_niblack(img1_valid, window_size=window_size, k=k)
        threshold_val = np.mean(thresh_local)
        print(f"(NIBLACK) Umbral promedio: {threshold_val:.3f}")

    elif threshold_method.lower() == "otsu":
        # threshold_otsu devuelve un único valor de umbral
        #https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
        threshold_val = threshold_otsu(img1_valid)
        print(f"(OTSU) Umbral: {threshold_val:.3f}")

    elif threshold_method.lower() == "li":
        # threshold_li devuelve un único valor de umbral
        #https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_li
        threshold_val = threshold_li(img1_valid)
        print(f"(LI) Umbral: {threshold_val:.3f}")

    else:
        raise ValueError(
            f"Método de umbralización desconocido: '{threshold_method}'. "
            "Use uno de: sauvola, niblack, otsu, li."
        )

    # ------------------------------------------------------------------
    # B) Crear la máscara binaria usando ese valor de umbral
    # ------------------------------------------------------------------
    output_data = np.zeros_like(img1_data)
    output_data[img1_data > threshold_val] = 1.0

    # Calcular la relación de potencia (en escala lineal)
    # La imagen está en dB, por lo que este valor se utiliza para expresión en BandMaths.
    powerratio = math.pow(10, (threshold_val / 10))

    print(f"Min img1_data: {np.min(img1_data)}")
    print(f"Max img1_data: {np.max(img1_data)}")
    print(f"Umbral usado para máscara: {threshold_val:.3f}")
    print(f"Número de pixeles > umbral: {np.sum(img1_data > threshold_val)}")

    # Construir rutas para guardar los gráficos
    # Se sugiere incluir el método en el nombre del archivo
    output_path_mask = os.path.join(output_directory, f"{sentinel_id}_mask_{threshold_method}.png")

    # --- Llamada a la función que dibuja la máscara binaria ---
    plot_binary_map(output_data, output_path_mask)

    # Obtener la banda base para aplicar la máscara
    try:
        band = textura.getBand('Gamma0_VH_GLCMMean')
        print(f"Banda seleccionada: {band.getName()}")
    except Exception as e:
        raise RuntimeError("No se pudo acceder a la banda Gamma0_VH_GLCMMean.") from e

    # Leer los datos originales para mostrar el histograma
    w, h = band.getRasterWidth(), band.getRasterHeight()
    band_data = np.zeros(w * h, np.float32)
    band.readPixels(0, 0, w, h, band_data)

    # Guardar el histograma, también con referencia al método
    output_path_hist = os.path.join(output_directory, f"{sentinel_id}_histogram_{threshold_method}.png")
    plot_histogram(band_data, output_path_hist)

    # Definir expresión lógica para BandMaths (detección de agua)
    expression = f"Gamma0_VH_GLCMMean < {powerratio}"
    print(f"Expresión para la máscara binaria: {expression}")

    # Configurar BandMaths para crear la banda 'flood'
    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    targetBand = BandDescriptor()
    targetBand.name = 'flood'
    targetBand.type = 'Float32'
    targetBand.expression = expression

    targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
    targetBands[0] = targetBand

    parameters = HashMap()
    parameters.put('targetBands', targetBands)

    # Ejecutar BandMaths
    try:
        flood = GPF.createProduct('BandMaths', parameters, textura)
        print("Máscara binaria de agua creada exitosamente.")
    except Exception as e:
        raise RuntimeError("Error al crear la máscara de agua con BandMaths.") from e

    return flood
