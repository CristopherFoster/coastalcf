# exportar_vh_a_tiff.py

import numpy as np
from esa_snappy import Product, PixelPos
from osgeo import gdal, osr

def exportar_vh_a_tiff(producto_gc, ruta_salida_sin_extension):
    """
    Exporta la banda 'VH' del producto SNAP como GeoTIFF y devuelve el array de NumPy corregido.
    Reemplaza valores menores a 0 por -2.
    """
    width = producto_gc.getSceneRasterWidth()
    height = producto_gc.getSceneRasterHeight()

    # Verificar si la banda 'VH' existe
    nombres_bandas = list(producto_gc.getBandNames())
    print(f"Bandas disponibles: {nombres_bandas}")
    banda = producto_gc.getBand('Gamma0_VH_GLCMMean')

    datos = np.zeros((height, width), dtype=np.float32)
    banda.readPixels(0, 0, width, height, datos)

    # Reemplazar valores < 0 por -2
    datos[datos < -1] = -1

    # Geo información
    geo_coding = producto_gc.getSceneGeoCoding()
    if geo_coding is None or not geo_coding.canGetGeoPos():
        raise RuntimeError("No se puede obtener geocodificación del producto.")

    latlon_tl = geo_coding.getGeoPos(PixelPos(0, 0), None)
    latlon_br = geo_coding.getGeoPos(PixelPos(width, height), None)
    lon_res = (latlon_br.lon - latlon_tl.lon) / width
    lat_res = (latlon_tl.lat - latlon_br.lat) / height
    geotransform = (latlon_tl.lon, lon_res, 0, latlon_tl.lat, 0, -lat_res)

    # Guardar como .tiff
    ruta_tiff = ruta_salida_sin_extension + ".tif"
    print(f"Guardando GeoTIFF en: {ruta_tiff}")
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(ruta_tiff, width, height, 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(datos)
    dataset.GetRasterBand(1).SetNoDataValue(-9999)
    dataset.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    dataset.SetProjection(srs.ExportToWkt())
    dataset.FlushCache()

    print(f"✅ GeoTIFF guardado correctamente en: {ruta_tiff}")
    return datos
