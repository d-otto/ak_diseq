# this is only run once, so it's ok if it takes its time

import json
import rasterstats
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from ak_diseq.data import load_rgi
import rioxarray
import matplotlib.pyplot as plt
import logging
import pandas as pd
from shapely import LineString

# todo: is this the wrong reference?
from ak_diseq.data import load_flowlines
from ak_diseq import ROOT

plt.switch_backend('agg')
plt.ioff()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %% import rgi

rgi = load_rgi()
rgi = rgi.loc[rgi.Area > 1]  # filter small glaciers


# %% load flowline shapes

flowlines = load_flowlines(rgi.RGIId)

# %% Options

plot_qc = True  # make qc plots for each glacier

# %% Load thickness tifs

# iterate through flowlines
Hf = {}  # dict to capture results
for rgiid in tqdm(rgi.RGIId, total=len(rgi)):
    p = Path(ROOT, f'data/farinotti/composite_thickness_RGI60/RGI60-01/{rgiid}_thickness.tif')
    
    if rgiid not in flowlines.RGIID.values:
        logger.info(f"Skipping {rgiid}, Area = {rgi.loc[rgi.RGIId == rgiid].Area.values[0]}")
        continue
    
    H = rioxarray.open_rasterio(p).sel(band=1)
    logger.debug(f"Opened raster {p}")

    crs = H.rio.crs
    flowline = flowlines.loc[flowlines.RGIID == rgiid]
    flowline = flowline.to_crs(crs)  # reproject shape to the raster's projection (zone varies based on raster file)

    # View the Coordinate Reference System (CRS) & spatial extent
    logger.debug(f"The CRS for H is: {H.rio.crs}")
    logger.debug(f"The spatial extent for H is: {H.rio.bounds()}")
    logger.debug(f"The CRS of the flowlines is: {flowline.crs.to_epsg()}")
    logger.debug(f"The spatial extent of the flowlines is:{flowline.total_bounds}")

    # iterate over points on the line & take thickness value
    sample_points = flowline['geometry'].segmentize(50)
    values = rasterstats.point_query(
        sample_points,
        H.values,
        affine=H.rio.transform(),
        interpolate='bilinear',
        nodata=np.nan
    )[0]

    # capture results
    Hf[rgiid] = values

    if plot_qc:
        feat_rgi = rgi.loc[rgi.RGIId == rgiid].to_crs(
            crs)  # get RGI outline and reproject 

        # plot qc plot of close-up of raster and flowline along with the actual values along the line
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), layout='constrained', gridspec_kw={"height_ratios": [1, 0.5]})
        H.plot(ax=ax[0])
        gdf = gpd.GeoDataFrame().from_features(flowline)
        gdf.plot(ax=ax[0], color='red', marker='o')
        feat_rgi.boundary.plot(ax=ax[0], edgecolor="white", linewidth=1)
        ax[0].set_aspect('equal')
        ax[0].set_title(f"Farinotti {rgiid}, {feat_rgi.iloc[0].Name}; Area = {feat_rgi.iloc[0].Area:.0f} km2")
        ax[1].plot(values)
        plt.savefig(Path(ROOT, "src/scripts/extract_farinotti_thickness/qc/", f"{rgiid}.png"))
        plt.close(fig)

    
    H.close()
logger.info('Complete!')

# clean up and write to file
p = Path(ROOT, "src/scripts/extract_farinotti_thickness/farinotti_hf.csv")
with open(p, 'w') as f:
    json.dump(Hf, f)
