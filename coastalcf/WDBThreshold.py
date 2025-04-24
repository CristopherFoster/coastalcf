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
#from .plot_thresholds import plot_thresholds

"""
def WDBThreshold(
    textura,
    sentinel_1_path,
    output_directory,
    window_size=31,
    k=0.2,     # extra param para Niblack/Sauvola
    r=None     # extra param para Sauvola
):
    
    Calcula máscaras de agua usando los cuatro métodos de umbralización
    (Otsu, Niblack, Sauvola y Li). Devuelve un producto SNAP con 4 bandas:
    flood_otsu, flood_niblack, flood_sauvola, flood_li.

    Args:
        textura: Producto SNAP que contiene la banda 'Gamma0_VH_GLCMMean'.
        sentinel_1_path (str): Ruta del archivo Sentinel-1.
        output_directory (str): Directorio para guardar resultados (imágenes).
        window_size (int): Tamaño de ventana para métodos locales (Niblack, Sauvola).
        k (float): Parámetro 'k' para Niblack y Sauvola.
        r (float): Parámetro 'r' para Sauvola, opcional.

    Returns:
        - Producto SNAP con 4 bandas binarias ('flood_otsu', 'flood_niblack', 
          'flood_sauvola', 'flood_li').
    
    sentinel_id = get_sentinel_fecha_id(sentinel_1_path)
    print("\n--- Iniciando detección de agua con TODOS los métodos ---")
    
    # Verificar que la banda requerida esté presente en el producto
    available_bands = list(textura.getBandNames())
    print(f"Bandas disponibles: {available_bands}")
    if 'Gamma0_VH_GLCMMean' not in available_bands:
        raise RuntimeError("La banda 'Gamma0_VH_GLCMMean' no está disponible en el producto.")

    # 1) Convertir la banda a decibeles (dB)
    paramToDB = HashMap()
    paramToDB.put('sourceBands', 'Gamma0_VH_GLCMMean')

    try:
        print("Convirtiendo a escala de decibeles (dB)...")
        producto_db = GPF.createProduct("LinearToFromdB", paramToDB, textura)
        banda_db = producto_db.getBand('Gamma0_VH_GLCMMean_db')
        print("Conversión a dB completada.")
    except Exception as e:
        raise RuntimeError("Error al convertir la banda a dB.") from e

    # 2) Leer datos de la banda en un array de NumPy
    print("Lectura de datos de la banda en un array de NumPy...")
    w, h = banda_db.getRasterWidth(), banda_db.getRasterHeight()
    img_db_data = np.zeros(w * h, np.float32)
    banda_db.readPixels(0, 0, w, h, img_db_data)

    # Estadísticas de la imagen en dB
    min_val, max_val = np.min(img_db_data), np.max(img_db_data)
    print(f"Rango de valores en dB: Min={min_val:.2f}, Max={max_val:.2f}")

    # Reshape del array 1D a 2D
    img_db_data = img_db_data.reshape((h, w))

    # Marcar valores inválidos (<= -1000) como NaN y rellenarlos
    img_db_data_masked = np.where(img_db_data <= -1000, np.nan, img_db_data)
    img_db_valid = np.nan_to_num(img_db_data_masked, nan=-1)  # -1 para no distorsionar

    # 3) Calcular los umbrales para cada método
    print("Calculando umbrales (Otsu, Niblack, Sauvola, Li)...")

    # Otsu => escalar
    otsu_val = threshold_otsu(img_db_valid)

    # Niblack => mapa local, tomamos la media como umbral global
    niblack_local = threshold_niblack(img_db_valid, window_size=window_size, k=k)
    niblack_val = np.mean(niblack_local)

    # Sauvola => mapa local, tomamos la media
    sauvola_local = threshold_sauvola(img_db_valid, window_size=window_size, k=k, r=r)
    sauvola_val = np.mean(sauvola_local)

    # Li => escalar
    li_val = threshold_li(img_db_valid)

    # Imprimirlos
    print(f"  - Otsu    = {otsu_val:.3f}")
    print(f"  - Niblack = {niblack_val:.3f}")
    print(f"  - Sauvola = {sauvola_val:.3f}")
    print(f"  - Li      = {li_val:.3f}")

    # 4) Construir expresiones para las 4 bandas binarias
    #    En SNAP, usaremos la banda 'Gamma0_VH_GLCMMean' (lineal, no dB)
    #    por lo tanto, convertimos los umbrales dB -> lineal
    def dB_to_linear(db_val):
        return math.pow(10, (db_val / 10))

    pow_otsu    = dB_to_linear(otsu_val)
    pow_niblack = dB_to_linear(niblack_val)
    pow_sauvola = dB_to_linear(sauvola_val)
    pow_li      = dB_to_linear(li_val)

    # Usamos la sintaxis: (condicion) ? valor_si_true : valor_si_false
    # para generar 1 o 0. Por ejemplo: (Gamma0_VH_GLCMMean < 0.02) ? 1 : 0
    nodata_val = -9999.0
    expr_otsu = f"(Gamma0_VH_GLCMMean <= -1000) ? {nodata_val} : ((Gamma0_VH_GLCMMean < {pow_otsu}) ? 1 : 0)"
    expr_niblack = f"(Gamma0_VH_GLCMMean <= -1000) ? {nodata_val} : ((Gamma0_VH_GLCMMean < {pow_niblack}) ? 1 : 0)"
    expr_sauvola = f"(Gamma0_VH_GLCMMean <= -1000) ? {nodata_val} : ((Gamma0_VH_GLCMMean < {pow_sauvola}) ? 1 : 0)"
    expr_li      = f"(Gamma0_VH_GLCMMean <= -1000) ? {nodata_val} : ((Gamma0_VH_GLCMMean < {pow_li}) ? 1 : 0)"





    # 5) Crear BandDescriptors para cada máscara
    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')

    bd_otsu = BandDescriptor()
    bd_otsu.name = 'flood_otsu'
    bd_otsu.type = 'Float32'
    bd_otsu.expression = expr_otsu

    bd_niblack = BandDescriptor()
    bd_niblack.name = 'flood_niblack'
    bd_niblack.type = 'Float32'
    bd_niblack.expression = expr_niblack

    bd_sauvola = BandDescriptor()
    bd_sauvola.name = 'flood_sauvola'
    bd_sauvola.type = 'Float32'
    bd_sauvola.expression = expr_sauvola

    bd_li = BandDescriptor()
    bd_li.name = 'flood_li'
    bd_li.type = 'Float32'
    bd_li.expression = expr_li

    # Array de targetBands
    targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 4)
    targetBands[0] = bd_otsu
    targetBands[1] = bd_niblack
    targetBands[2] = bd_sauvola
    targetBands[3] = bd_li

    parameters = HashMap()
    parameters.put('targetBands', targetBands)

    # 6) Crear productos individuales para cada método
    productos_binarios = {}
    descriptores = {
        "flood_otsu": bd_otsu,
        "flood_niblack": bd_niblack,
        "flood_sauvola": bd_sauvola,
        "flood_li": bd_li
    }



    for nombre, descriptor in descriptores.items():
        print(f"Creando producto individual para: {nombre}")
    
        # Crear array con un solo descriptor
        targetBandArray = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
        targetBandArray[0] = descriptor

        params = HashMap()
        params.put('targetBands', targetBandArray)

        try:
            producto = GPF.createProduct('BandMaths', params, textura)
            productos_binarios[nombre] = producto
            print(f"Producto creado: {nombre}")
        except Exception as e:
            raise RuntimeError(f"Error al crear la máscara {nombre} con BandMaths.") from e

        # === 7) Exportar la máscara a imagen PNG y su histograma ===
        band = producto.getBand(nombre)
        w, h = band.getRasterWidth(), band.getRasterHeight()
        mask_data = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, mask_data)
        mask_data = mask_data.reshape((h, w))
        metodo = nombre.replace("flood_", "").capitalize()

        # Imagen de la máscara
        output_path_mask = os.path.join(output_directory, f"{sentinel_id}_{nombre}.png")
        #plot_binary_map(mask_data, output_path_mask, metodo=metodo)

        # Histograma
        output_path_hist = os.path.join(output_directory, f"{sentinel_id}_histogram_{nombre}.png")
        #plot_binary_map(mask_data, output_path_mask, metodo=metodo)


    #return list(productos_binarios.values())
    return productos_binarios
"""
def WDBThreshold(
    textura,
    sentinel_1_path,
    output_directory,
    window_size=31,
    k=0.2,
    r=None
):
    """
    Calcula máscaras de agua usando cuatro métodos de umbralización.
    Agua = 1, Tierra = 2, NoData = -1
    """
    from esa_snappy import GPF, HashMap, jpy
    import numpy as np
    import math
    from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola, threshold_li
    from .get_sentinel_fecha_id import get_sentinel_fecha_id
    import os

    sentinel_id = get_sentinel_fecha_id(sentinel_1_path)
    print("\n--- Iniciando detección de agua con TODOS los métodos ---")

    available_bands = list(textura.getBandNames())
    print(f"Bandas disponibles: {available_bands}")
    if 'Gamma0_VH_GLCMMean' not in available_bands:
        raise RuntimeError("La banda 'Gamma0_VH_GLCMMean' no está disponible en el producto.")

    paramToDB = HashMap()
    paramToDB.put('sourceBands', 'Gamma0_VH_GLCMMean')


    band = textura.getBand("Gamma0_VH_GLCMMean")
    data = np.zeros(band.getRasterWidth() * band.getRasterHeight(), np.float32)
    band.readPixels(0, 0, band.getRasterWidth(), band.getRasterHeight(), data)
    print("Rango de datos lineales:", np.min(data), np.max(data))

    try:
        print("Convirtiendo a escala de decibeles (dB)...")
        producto_db = GPF.createProduct("LinearToFromdB", paramToDB, textura)
        banda_db = producto_db.getBand('Gamma0_VH_GLCMMean_db')
        print("Conversión a dB completada.")
    except Exception as e:
        raise RuntimeError("Error al convertir la banda a dB.") from e

    w, h = banda_db.getRasterWidth(), banda_db.getRasterHeight()
    img_db_data = np.zeros(w * h, np.float32)
    banda_db.readPixels(0, 0, w, h, img_db_data)

    min_val, max_val = np.min(img_db_data), np.max(img_db_data)
    print(f"Rango de valores en dB: Min={min_val:.2f}, Max={max_val:.2f}")

    img_db_data = img_db_data.reshape((h, w))
    img_db_data_masked = np.where(img_db_data <= -1000, np.nan, img_db_data)
    img_db_valid = np.nan_to_num(img_db_data_masked, nan=-1)

    print("Calculando umbrales (Otsu, Niblack, Sauvola, Li)...")
    otsu_val = threshold_otsu(img_db_valid)
    niblack_val = np.mean(threshold_niblack(img_db_valid, window_size=window_size, k=k))
    sauvola_val = np.mean(threshold_sauvola(img_db_valid, window_size=window_size, k=k, r=r))
    li_val = threshold_li(img_db_valid)

    print(f"  - Otsu    = {otsu_val:.3f}")
    print(f"  - Niblack = {niblack_val:.3f}")
    print(f"  - Sauvola = {sauvola_val:.3f}")
    print(f"  - Li      = {li_val:.3f}")

    def dB_to_linear(db_val):
        return math.pow(10, (db_val / 10))

    pow_otsu = dB_to_linear(otsu_val)
    pow_niblack = dB_to_linear(niblack_val)
    pow_sauvola = dB_to_linear(sauvola_val)
    pow_li = dB_to_linear(li_val)
    

    expr_otsu    = f"(Gamma0_VH_GLCMMean < {pow_otsu}) ? 0 : 1"
    expr_niblack = f"(Gamma0_VH_GLCMMean < {pow_niblack}) ? 0 : 1"
    expr_sauvola = f"(Gamma0_VH_GLCMMean < {pow_sauvola}) ? 0 : 1"
    expr_li      = f"(Gamma0_VH_GLCMMean < {pow_li}) ? 0 : 1"

    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')

    def crear_band_descriptor(nombre, expresion):
        bd = BandDescriptor()
        bd.name = nombre
        bd.type = 'Int32'
        bd.expression = expresion
        return bd

    bd_otsu = crear_band_descriptor('flood_otsu', expr_otsu)
    bd_niblack = crear_band_descriptor('flood_niblack', expr_niblack)
    bd_sauvola = crear_band_descriptor('flood_sauvola', expr_sauvola)
    bd_li = crear_band_descriptor('flood_li', expr_li)

    targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 4)
    targetBands[0] = bd_otsu
    targetBands[1] = bd_niblack
    targetBands[2] = bd_sauvola
    targetBands[3] = bd_li

    parameters = HashMap()
    parameters.put('targetBands', targetBands)

    productos_binarios = {}
    descriptores = {
        "flood_otsu": bd_otsu,
        "flood_niblack": bd_niblack,
        "flood_sauvola": bd_sauvola,
        "flood_li": bd_li
    }

    for nombre, descriptor in descriptores.items():
        print(f"Creando producto individual para: {nombre}")
        targetBandArray = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
        targetBandArray[0] = descriptor
        params = HashMap()
        params.put('targetBands', targetBandArray)

        try:
            producto = GPF.createProduct('BandMaths', params, textura)
            productos_binarios[nombre] = producto
            print(f"Producto creado: {nombre}")
        except Exception as e:
            raise RuntimeError(f"Error al crear la máscara {nombre} con BandMaths.") from e
        


    return productos_binarios, {
        "otsu": otsu_val,
        "niblack": niblack_val,
        "sauvola": sauvola_val,
        "li": li_val
    }


    
