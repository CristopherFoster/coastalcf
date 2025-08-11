# coastalcf/TresholdsPlots.py

def k_graficos_binarizacion(img_db_product, productos_binarios_corrected,
                                output_directory, object_size=15, sentinel_id="VH"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    from skimage.morphology import remove_small_objects
    import os
    import rasterio
    from rasterio.transform import from_bounds
    from esa_snappy import GeoPos, PixelPos

    titles = ["Otsu", "Niblack", "Sauvola", "MCET", "MCET Iterative"]

    # Colormaps
    cmap_masks = ListedColormap(["white", "purple", "pink"])
    norm_masks = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap_masks.N)
    rocket = sns.color_palette("gray", as_cmap=True).copy()
    rocket.set_bad("white")

    def add_bar_scale(ax, extent, total_km=5, segment_km=1,
                      pad_x=0.05, pad_y=0.06,
                      height_frac=0.015, font_size=10):
        lon_min, lon_max, lat_min, lat_max = extent
        lat_c = (lat_min + lat_max) / 2
        m_per_deg_lon = 111_320 * np.cos(np.deg2rad(lat_c))

        seg_deg = segment_km * 1000 / m_per_deg_lon
        bar_deg = total_km * 1000 / m_per_deg_lon
        bar_ht = (lat_max - lat_min) * height_frac
        x0 = lon_min + (lon_max - lon_min) * pad_x
        y0 = lat_min + (lat_max - lat_min) * pad_y

        n_seg = int(total_km / segment_km)
        for i in range(n_seg):
            xi = x0 + i * seg_deg
            face = "black" if i % 2 == 0 else "white"
            ax.add_patch(Rectangle((xi, y0), seg_deg, bar_ht,
                                   facecolor=face, edgecolor="black",
                                   linewidth=1.0, zorder=4))

        ax.add_patch(Rectangle((x0, y0), bar_deg, bar_ht,
                               facecolor="none", edgecolor="black",
                               linewidth=1.0, zorder=4))

        for km in range(0, total_km + 1, segment_km):
            xi = x0 + km * seg_deg
            ax.text(xi, y0 - bar_ht * 0.55, f"{km}",
                    ha="center", va="top", fontsize=font_size, zorder=5)

    def style_axes(ax, lon_ticks, lat_ticks):
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.grid(linestyle=":", linewidth=0.8, alpha=0.7)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelrotation=20)

    # Lectura de la imagen VH original
    band = img_db_product.getBand("img_db_valid_VH")
    w, h = band.getRasterWidth(), band.getRasterHeight()
    img_db = np.zeros(w * h, dtype=np.float32)
    band.readPixels(0, 0, w, h, img_db)
    img_db = img_db.reshape((h, w))

    geo = img_db_product.getSceneGeoCoding()
    ul, lr = GeoPos(), GeoPos()
    geo.getGeoPos(PixelPos(0, 0), ul)
    geo.getGeoPos(PixelPos(w - 1, h - 1), lr)
    extent = [ul.lon, lr.lon, lr.lat, ul.lat]
    lon_ticks = np.arange(extent[0], extent[1] + 0.01, 0.05)
    lat_ticks = np.arange(extent[2], extent[3] + 0.01, 0.05)

    # Grilla
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 0.03, 1.1], hspace=0.3, wspace=0.15)
    axes = [fig.add_subplot(gs[i // 2, i % 2 * 2]) for i in range(6)]
    cax0 = fig.add_axes([0.43, 0.68, 0.015, 0.18])

    for ax in axes:
        ax.set_aspect("equal")
    for idx in [0, 2, 4]:
        pos = axes[idx].get_position()
        dx = axes[0].get_position().x0 - pos.x0
        new_pos = [pos.x0 + dx, pos.y0, pos.width, pos.height]
        axes[idx].set_position(new_pos)

    nodata_value = -1
    img_mask = np.ma.masked_where(img_db == nodata_value, img_db)
    im0 = axes[0].imshow(img_mask, cmap=rocket,
                         vmin=np.nanmin(img_db[~img_mask.mask]),
                         vmax=np.nanmax(img_db[~img_mask.mask]),
                         extent=extent, aspect="equal")
    style_axes(axes[0], lon_ticks, lat_ticks)
    add_bar_scale(axes[0], extent)
    axes[0].set_title("Original VH Image (dB)", fontsize=25)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    cbar = fig.colorbar(im0, cax=cax0, orientation="vertical")
    cbar.set_label("Backscatter (dB)")

    for i, (nombre, (producto, umbral)) in enumerate(productos_binarios_corrected.items(), start=1):
        band = producto.getBand(f"{nombre}_VH")
        w, h = band.getRasterWidth(), band.getRasterHeight()
        data = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, data)
        data = np.round(data.reshape((h, w))).astype(np.int32)

        nodata_mask = (data == -1)
        data[nodata_mask] = 0
        agua_mask = (data == 0)
        agua_filtrada = remove_small_objects(agua_mask, object_size, connectivity=1)
        final_mask = data.copy()
        final_mask[agua_mask & ~agua_filtrada] = 1
        final_mask[nodata_mask] = -1

        ax = axes[i]
        ax.imshow(final_mask, cmap=cmap_masks, norm=norm_masks,
                extent=extent, aspect="equal")
        style_axes(ax, lon_ticks, lat_ticks)
        add_bar_scale(ax, extent)

        ax.set_title(f"{titles[i-1]} Threshold", fontsize=25)
        ax.text(0.98, 0.98, f"Threshold = {umbral:.4f}",
                transform=ax.transAxes, fontsize=15,
                va="top", ha="right",
                bbox=dict(facecolor="white", edgecolor="gray",
                          boxstyle="round,pad=0.3", alpha=0.8))

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        """
        export_mask = final_mask.copy()
        export_mask[final_mask == -1] = -9999
        transform = from_bounds(extent[0], extent[2], extent[1], extent[3], w, h)
        out_tiff = os.path.join(output_directory,
                                f"{sentinel_id}_{nombre}_filtered.tif")
        with rasterio.open(out_tiff, "w", driver="GTiff", height=h, width=w,
                           count=1, dtype=rasterio.int16, crs="EPSG:4326",
                           transform=transform, nodata=-9999, compress="lzw") as dst:
            dst.write(export_mask.astype(rasterio.int16), 1)
        print(f"✅ Exportado: {out_tiff}")
        """
        # Exportar solo si es Sauvola (tercera posición)
        if i == 3:
            print(i, nombre)  # para ver qué método está siendo procesado en cada iteración
            print(f"Iteración: {i}, Método: {nombre}")
            print("extent:", extent)
            
            export_mask = final_mask.copy()
            export_mask[final_mask == -1] = -9999
            transform = from_bounds(extent[0], extent[2], extent[1], extent[3], w, h)
            out_tiff = os.path.join(output_directory,
                                    f"{sentinel_id}_{nombre}_sauvola.tif")
            with rasterio.open(out_tiff, "w", driver="GTiff", height=h, width=w,
                            count=1, dtype=rasterio.int16, crs="EPSG:4326",
                            transform=transform, nodata=-9999, compress="lzw") as dst:
                dst.write(export_mask.astype(rasterio.int16), 1)
            print(f"✅ Exportado: {out_tiff}")


    legend_elems = [Patch(facecolor="purple", edgecolor="k", label="Detected Water"),
                    Patch(facecolor="pink", edgecolor="k", label="Land / Background"),
                    Patch(facecolor="white", edgecolor="k", label="No Data")]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3, fontsize=14,
               frameon=True, framealpha=1, fancybox=True, bbox_to_anchor=(0.5, 0.08))

    plt.savefig(os.path.join(output_directory,
                             f'{sentinel_id}_ThresholdsFiltered.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
