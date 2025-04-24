import os
import coastalcf as clf
from esa_snappy import ProductIO, GPF
from esa_snappy import GPF, HashMap, jpy
import numpy as np
from osgeo import gdal, ogr, osr 

from coastalcf.exportar_raster_y_shapefile import exportar_raster_y_shapefile
from coastalcf.exportar_raster_y_shapefile import convert_to_8bit_gdal

from coastalcf.wbplots import histograma_threshold_metrics

"""
---------------------------------------------------------------------------
***************************************************************************
---------------------------------------------------------------------------
//////////////////        PREPARACIÓN DE LOS DATOS        /////////////////
---------------------------------------------------------------------------
***************************************************************************
---------------------------------------------------------------------------
"""



# --- 1. Define la región de interés (ROI) para el subconjunto y la ruta del producto.
# Si no existew substet colocar None
# x = y = width = height = None
#Inicio de la región de interés (ROI) en coordenadas de píxeles.
x=32
y = 14091
#Final de la región de interés (ROI) en coordenadas de píxeles.
width = 1355
height = 897

x=23
y = 13524
#Final de la región de interés (ROI) en coordenadas de píxeles.
width = 1667
height = 1617
#x = y = width = height = None
#23 13524 1667 1617
# Defininir parámetros para el operador GLCM.
para = {'sourceBands': 'VH'}

# Path del Producto de Sentinel 1 y ouput de archivos.

sentinel_1_path = r'C:\Users\c4cfo\OneDrive\CENTROGEO-cfoster\1_ENSOCE\C2_Data\2023\S1A_IW_GRDH_1SDV_20230607T130609_20230607T130634_048880_05E0CE_1909'
sentinel_id = clf.get_sentinel_fecha_id(sentinel_1_path)
output_directory = r'C:\Users\c4cfo\OneDrive\CENTROGEO-cfoster\1_ENSOCE\thresholdmzt\6_junio'
output_directorysh = r'C:\Users\c4cfo\OneDrive\CENTROGEO-cfoster\1_ENSOCE\thresholdmzt\6_junio\shape'


"""
---------------------------------------------------------------------------
***************************************************************************
---------------------------------------------------------------------------
///////////////// APLICACIÓN DE LOS PASOS DE PROCESAMIENTO ////////////////
---------------------------------------------------------------------------
***************************************************************************
---------------------------------------------------------------------------
"""
# --- 1. Lectura de metadatos del producto Sentinel-1 ---
product = clf.readMetadata(sentinel_1_path, toPrint=True)  

# --- 2. Eliminación de ruido térmico : do_thermal_noise_removal [output]---
thermaremoved = clf.do_thermal_noise_removal(product)
if thermaremoved is None:
    raise RuntimeError("Thermal noise removal failed.")
print(f"Bands after ThermalNoiseRemoval: {list(thermaremoved.getBandNames())}")
#**************************************************************************
# --- 3. Calibración radiométrica : radiometricCalibration [calibrated] ---
calibrate = clf.radiometricCalibration(thermaremoved)
if calibrate is None:
    raise RuntimeError("Radiometric calibration failed.")
print(f"Bands after RadiometricCalibration: {list(calibrate.getBandNames())}")

# - Verificar las bandas después de la calibración
valid_bands = list(calibrate.getBandNames())
print(f" Bandas disponibles después de la calibración: {valid_bands}")

# - Asegurar que usamos solo las bandas que existen
subset_bands = [band for band in ['Sigma0_VH', 'Sigma0_VV', 'Beta0_VH', 'Beta0_VV'] if band in valid_bands]
if not subset_bands:
    raise RuntimeError("No existen Bandas válidas para el subset.")
# ---
# --- Paso 3.1: Crear el subset después de calibrar ---

subset_product = clf.subset(calibrate, x, y, width, height, subset_bands)

# - Validar subset antes de continuar
if subset_product is None or subset_product.getSceneRasterWidth() == 0 or subset_product.getSceneRasterHeight() == 0:
    raise RuntimeError("Error: El subset no contiene datos válidos.")

print(f" Subset creado correctamente con bandas: {list(subset_product.getBandNames())}")
print(subset_product.getMetadataRoot().toString())
#**************************************************************************
# --- 4. Aplicación de multilooking : perform_multilook [multilook]
print("Performing multilooking...")
#multilook = perform_multilook(calibrate)
multilook = clf.perform_multilook(subset_product)
if multilook is None:
    raise RuntimeError("Multilooking failed.")
#**************************************************************************
# --- 5. Filtrado de speckle : speckleFiltering [speckle]
print("Applying speckle filtering...")
speckle = clf.speckleFiltering(multilook, toPrint=True)
if speckle is None:
    raise RuntimeError("Speckle filtering failed.")


#**************************************************************************
# --- 6. Nivelación del terreno (Terrain Flattening) : perform_terrain_flattening [terrain]
print("Applying terrain flattening...")
terrain1 = clf.perform_terrain_flattening(speckle)
if terrain1 is None:
    raise RuntimeError("Terrain flattening failed.")

# Guardar el producto completo, con todas sus bandas
output_path_terrain = os.path.join(output_directory, "terrain1")
#ProductIO.writeProduct(terrain1, output_path_terrain, "GeoTIFF")
#******************x********************************************************
# --- 7. Cálculo de texturas GLCM : glcmOp / glcm [textura]
print("Calculating texture...")
textura = clf.glcmOp().glcm(terrain1, para)
if textura is None:
    raise RuntimeError("GLCM texture calculation failed.")
# Guardar el producto completo, con todas sus bandas
output_path_terrain = os.path.join(output_directory, "glcm")
#glcmc = clf.geometricCorrection(textura, toPrint=True)

producto_gc = clf.geometricCorrection(textura, toPrint=True)
tmp_tif   = os.path.join(output_directory, "glcm_tmp.tif")     # float32
final_tif = os.path.join(output_directory, "glcm_8bit.tif")    # uint8
final_shp = os.path.join(output_directory, "glcm.shp")
# --- 2 · Exportar el Product corregido a GeoTIFF (float) ---
print(f"Escribiendo GeoTIFF temporal: {tmp_tif}")
ProductIO.writeProduct(producto_gc, tmp_tif, "GeoTIFF")



# --- 3 · Convertir a 8 bits con tu propia función ---
convert_to_8bit_gdal(tmp_tif, final_tif, nodata_val=-9999.0)
print(f"GeoTIFF 8 bits escrito: {final_tif}")

# --- 4 · Borrar el temporal (opcional) ---
os.remove(tmp_tif)

# --- 5 · Vectorizar a shapefile (opcional) ---

exportar_raster_y_shapefile(producto_gc, raster_path=final_tif, shapefile_path=final_shp)

print("Proceso terminado ✔️")



from coastalcf.exportar_raster_y_shapefile import convert_to_8bit_gdal
#ProductIO.writeProduct(glcmc, output_path_terrain, "GeoTIFF")
#!!!!!!!!!!!!!!!!!!!!!


#**************************************************************************

# 8. Umbralización (Método Sauvola) : waterDetectionBinarization [product]
print("Performing water detection...")
#waterDetection = clf.waterDetectionBinarization(textura, sentinel_1_path, output_directory)
productos_binarios, umbrales = clf.WDBThreshold(
    textura=textura,
    sentinel_1_path=sentinel_1_path,
    output_directory=output_directory,
    window_size=31,
    k=0.2,     # extra param para Niblack/Sauvola
    r=None     # extra param para Sauvola
)

productos_para_graficar = {
    "flood_otsu": (productos_binarios["flood_otsu"], umbrales["otsu"]),
    "flood_niblack": (productos_binarios["flood_niblack"], umbrales["niblack"]),
    "flood_sauvola": (productos_binarios["flood_sauvola"], umbrales["sauvola"]),
    "flood_li": (productos_binarios["flood_li"], umbrales["li"]),
}
#**************************************************************************
#band_names = ["flood_otsu", "flood_niblack", "flood_sauvola", "flood_li"]

# 9. Corrección geométrica: geometricCorrection [corrected]
print("Applying terrain correction to all thresholded products...")


print("Applying terrain correction to all thresholded products...")

productos_binarios_corrected = {}
for nombre, (producto, threshold) in productos_para_graficar.items():
    print(f" -> Corrigiendo: {nombre}")
    corregido = clf.geometricCorrection(producto, toPrint=True)
    if corregido is None:
        print(f"⚠️  No se pudo corregir el producto: {nombre}")
    else:
        productos_binarios_corrected[nombre] = (corregido, threshold)





# 10. Visualización en grid de productos corregidos
from coastalcf.wbplots import aplicar_correccion_y_grid
aplicar_correccion_y_grid(productos_binarios_corrected, output_directory, sentinel_id)


##---------------------------------------
# 1) Abrir el GeoTIFF en modo solo-lectura
tif_path = final_tif
ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
band = ds.GetRasterBand(1)

# 2) Leer la banda a un array float32
glcm = band.ReadAsArray().astype(np.float32)

# 3) Manejar NoData → NaN
nodata = band.GetNoDataValue()      # debería devolver -9999.0
if nodata is not None:
    glcm[glcm == nodata] = np.nan   # así tu histograma ignora nodata

# 4) ¡Listo!  →  Llamar a la función
thresholds, scores = histograma_threshold_metrics(glcm)
#---------------------------------------------------------
    
for nombre, producto in productos_binarios_corrected.items():
    # Generar sufijos personalizados para cada nombre de producto
    suffix_base = f"_{nombre}_corrected"

    # Generar rutas de salida usando tus funciones reutilizables
    raster_path = clf.generate_raster_path(sentinel_1_path, output_directory)
    shapefile_path = clf.generate_shapefile_path(sentinel_1_path, output_directory)

    print(f"Procesando producto corregido: {nombre}")
    
    # Exportar raster y shapefile
    #clf.exportar_raster_y_shapefile(producto, raster_path, shapefile_path)


"""
for nombre, producto in productos_binarios_corrected.items():
    output_path = os.path.join(output_directory, f"{sentinel_id}_{nombre}_corrected.tif")  
    shape_path = os.path.join(output_directory, f"{sentinel_id}_{nombre}_corrected.shp")  
    print(f"Guardando producto corregido: {output_path}")
    #ProductIO.writeProduct(producto, output_path, "GeoTIFF")  # o "GeoTIFF" si prefieres tif
"""
#extent=[-106.52, -106.41, 23.17, 23.27]
