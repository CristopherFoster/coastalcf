# geometricCorrection.py

from esa_snappy import GPF, HashMap, jpy


def geometricCorrection(speckle, toPrint=True):
    """
    Aplica la corrección geométrica (Terrain Correction) a un producto Sentinel-1,
    asignando -1 como valor NoData en todas las bandas.

    Args:
        speckle: Producto SNAP como entrada.
        toPrint: Si es True, imprime información del producto corregido.

    Returns:
        Producto corregido con -1 como NoData.
    """
    nodata_val = -1.0

    print("\n--- Iniciando corrección geométrica ---")

    # Asignar NoData antes de la corrección
    for bandName in speckle.getBandNames():
        band = speckle.getBand(bandName)
        band.setNoDataValue(nodata_val)
        band.setNoDataValueUsed(True)
        if toPrint:
            print(f"NoData asignado a banda '{bandName}' con valor: {nodata_val}")

    parameters = HashMap()
    parameters.put('demResamplingMethod', "BILINEAR_INTERPOLATION")
    parameters.put('imgResamplingMethod', "BILINEAR_INTERPOLATION")
    parameters.put('pixelSpacingInMeter', 10.0)
    parameters.put('mapProjection', "WGS84(DD)")
    parameters.put('externalDEMNoDataValue', float(nodata_val))
    parameters.put('noDataValue', float(nodata_val))
    parameters.put('nodataValueAtSea', False)
    parameters.put('maskOutAreaWithoutDEM', False)
    parameters.put('pixelSpacingInDegree', 0.0)

    try:
        print("Aplicando corrección geométrica...")
        corrected = GPF.createProduct('Terrain-Correction', parameters, speckle)
        print("Corrección geométrica completada exitosamente.")

        # Reasignar nodata a las bandas corregidas
        for bandName in corrected.getBandNames():
            band = corrected.getBand(bandName)
            band.setNoDataValue(nodata_val)
            band.setNoDataValueUsed(True)
            if toPrint:
                print(f"NoData asignado a banda corregida '{bandName}' con valor: {nodata_val}")

    except Exception as e:
        raise RuntimeError(f"Error durante la corrección geométrica: {e}")

    if toPrint:
        band_names = list(corrected.getBandNames())
        print(f"Bandas resultantes: {band_names}")

    print("Proceso de corrección geométrica finalizado.")
    return corrected
