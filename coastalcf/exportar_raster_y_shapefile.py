# exportar_raster_y_shapefile.py

from osgeo import gdal, ogr, osr
from esa_snappy import ProductIO
import os
import numpy as np

def convert_to_8bit_gdal(input_path, output_path, nodata_val=-1):
    """Convierte un GeoTIFF a 8 bits usando GDAL, conservando metadatos y nodata explícito."""
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"No se pudo abrir el archivo: {input_path}")

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    
    # Reemplaza valores menores a 0 por 0
    data[data < -2] = 0

    # Evita escalar valores nodata
    data_valid = data[data != nodata_val]
    if data_valid.size == 0:
        raise ValueError("No hay datos válidos para escalar.")

    min_val = np.nanmin(data_valid)
    max_val = np.nanmax(data_valid)

    if max_val - min_val == 0:
        raise ValueError("La imagen tiene valores constantes, no se puede escalar.")

    # Escalar valores válidos a 0–255 y mantener nodata como está
    scaled = np.full_like(data, 0, dtype=np.uint8)
    mask_valid = (data != nodata_val)
    scaled[mask_valid] = ((data[mask_valid] - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(dataset.GetGeoTransform())
    out_ds.SetProjection(dataset.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(scaled)

    # ✅ Registrar nodata explícitamente
    out_band.SetNoDataValue(nodata_val)

    out_ds.FlushCache()
    out_ds = None
    return output_path


def exportar_raster_y_shapefile(terrain, raster_path, shapefile_path):
    """
    Exporta un producto SNAP a GeoTIFF (8 bits) y lo convierte en un shapefile utilizando GDAL.
    Solo conserva la versión final en 8 bits.
    """

    print("\n--- Iniciando exportación de raster y generación de shapefile ---")

    os.makedirs(os.path.dirname(raster_path), exist_ok=True)

    # Usar ruta temporal para el GeoTIFF original
    raster_path_tmp = raster_path.replace(".tif", "_tmp.tif")

    # 1. Exportar a GeoTIFF temporal desde SNAP
    try:
        print(f"Escribiendo producto a GeoTIFF temporal: {raster_path_tmp}")
        ProductIO.writeProduct(terrain, raster_path_tmp, "GeoTIFF")
        print("GeoTIFF exportado exitosamente.")
    except Exception as e:
        raise RuntimeError(f"Error al escribir el archivo GeoTIFF: {raster_path_tmp}. Error: {e}")

    # 2. Convertir a 8 bits y guardar en la ruta final
    print(f"Convirtiendo a 8 bits: {raster_path}")
    convert_to_8bit_gdal(raster_path_tmp, raster_path, nodata_val=-9999.0)
    print("Conversión a 8 bits completada.")

    # 3. Eliminar el archivo temporal de 16 bits
    try:
        os.remove(raster_path_tmp)
        print(f"Archivo temporal eliminado: {raster_path_tmp}")
    except Exception as e:
        print(f"Advertencia: no se pudo eliminar el archivo temporal: {e}")

    # 4. Convertir a shapefile
    print("\n--- Iniciando conversión de raster a shapefile ---")

    try:
        src_ds = gdal.Open(raster_path)
        print(f"Raster cargado correctamente: {raster_path}")
    except RuntimeError as e:
        raise RuntimeError(f"No se pudo abrir el archivo raster: {raster_path}. Error: {e}")

    srcband = src_ds.GetRasterBand(1)
    if srcband is None:
        raise ValueError(f"El raster no contiene una banda válida: {raster_path}")

    srs = osr.SpatialReference()
    proj = src_ds.GetProjectionRef()
    if not proj:
        print("Advertencia: el raster no tiene proyección. Se usará WGS84 (EPSG:4326) por defecto.")
        srs.ImportFromEPSG(4326)
    else:
        srs.ImportFromWkt(proj)

        # Crear carpeta 'shapes' dentro del path del shapefile
    shapefile_dir = os.path.join(os.path.dirname(shapefile_path), "shapes")
    os.makedirs(shapefile_dir, exist_ok=True)

    # Redefinir la ruta completa al shapefile dentro de la nueva carpeta
    shapefile_name = os.path.basename(shapefile_path)
    shapefile_path = os.path.join(shapefile_dir, shapefile_name)

    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(shapefile_path)
    dst_layer = dst_ds.CreateLayer(os.path.splitext(os.path.basename(shapefile_path))[0], srs=srs)

    field_defn = ogr.FieldDefn("DN", ogr.OFTInteger)
    dst_layer.CreateField(field_defn)

    try:
        print("Convirtiendo raster a polígonos...")
        gdal.Polygonize(srcband, srcband.GetMaskBand(), dst_layer, 0, [], callback=None)
        print(f"Shapefile creado exitosamente: {shapefile_path}")
    except Exception as e:
        raise RuntimeError(f"Error durante la conversión raster a vector: {e}")
    finally:
        src_ds = None
        dst_ds = None
        print("Conversión finalizada.")

    return raster_path, shapefile_path