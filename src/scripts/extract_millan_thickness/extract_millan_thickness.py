import json
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from ak_diseq.data import load_rgi
import rioxarray
import matplotlib.pyplot as plt
import logging
import rasterstats
from shapely import LineString
import pandas as pd

from ak_diseq import ROOT
from ak_diseq.data import load_flowlines 

plt.switch_backend('agg')
plt.ioff()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# %% import rgi

rgi = load_rgi()
rgi = rgi.loc[rgi.Area > 1]  # filter small glaciers


#%% load flowline shapes

flowlines = load_flowlines(rgi.RGIId)


#%% Options

plot_qc = True  # make qc plots for each glacier


#%% Load thickness tifs

Hf = {}  # dict to capture results
ps = list(Path(ROOT, 'data/millan/RGI-1/').glob('**/THICKNESS*'))
for p in ps:
    subregion = p.name.split("_")[1]
    H = rioxarray.open_rasterio(p).sel(band=1)
    logger.info(f"Opened raster {p}")

    # View the Coordinate Reference System (CRS) & spatial extent
    logger.info(f"The CRS for H is: {H.rio.crs}")
    logger.info(f"The spatial extent for H is: {H.rio.bounds()}")

    crs = H.rio.crs
    test_lines = flowlines.copy().to_crs(crs)
    x0, y0, x1, y1 = H.rio.bounds()
    test_lines = test_lines.cx[x0:x1, y0:y1]
    logger.info(f"The CRS of the flowlines is: {test_lines.crs.to_epsg()}")
    logger.info(f"The spatial extent of the flowlines is:{test_lines.total_bounds}")

    ## Make sure everything is working right
    # plot map & flowlines
    logger.info(f'Converted flowlines to CRS with EPSG = {crs}. Flowline EPSG = {test_lines.crs.to_epsg()}')
    
    if plot_qc:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), layout='constrained')
        logger.info('Plotting H...')
        H.plot.imshow(ax=ax)
        logger.info('Plotting flowlines...')
        test_lines.plot(ax=ax, color='red')
        p = Path(ROOT, f'src/scripts/extract_millan_thickness/qc/test_region_{subregion}.png')
        logger.info(f'Saving plot to {p.resolve()}')
        plt.savefig(p)
        plt.close()
    
    # iterate through flowlines
    for i, feature in tqdm(enumerate(test_lines.iterfeatures(show_bbox=True)), total=len(test_lines)):
        rgiid = feature['properties']['RGIID']
        feat_rgi = rgi.loc[rgi.RGIId == rgiid].to_crs(crs)  # reproject shape to the raster's projection (zone varies based on raster file)
        rgi_bbox = feat_rgi.iloc[0]['geometry']

        # clip to RGI geometry as a check that everything is correct
        # exception when the flowline does not overlap with the raster
        # since we use method="nearest" to extract points, this is required
        try:
            h = H.rio.clip_box(*rgi_bbox.bounds)  
        except: 
            logger.debug(f"Flowline and raster do not overlap. Linestring bounds: {feature['bbox']}, RGI geometry bounds: {rgi_bbox.bounds}")
            continue
    
        # iterate over points on the line & take thickness value
        #values = []
        #for coords in feature['geometry']['coordinates']:
        # resamples line to have a max distance of 50 * sqrt(2) m between points
        # this guarantees that each point is on a new pixel
        sample_points = LineString(feature['geometry']['coordinates']).segmentize(50)
        values = rasterstats.point_query(
            sample_points,
            H.values,
            affine=H.rio.transform(),
            interpolate='bilinear',
            nodata=np.nan
        )[0]
        if (np.nansum(values) < 1) :  # skip if all points are 0 
            continue
        
        # Check if value exists from previous raster. Might happen because using one file from region 2 to get full coverage
        if rgiid in Hf.keys():
            logger.warning(f"Multiple features found for {rgiid}."
                           f"Existing values ({len(Hf[rgiid])}): {[f'{num:.0f}' for num in Hf[rgiid]]}",
                           f"New values ({len(values)}): {[f'{num:.0f}' for num in values]}")
            # the existing and new values all are the same lengths, so points aren't being dropped
            # executive decision, taking the maximum from each... will see how it looks in QC
            if np.sum(Hf[rgiid]) > np.sum(values):
                values = Hf[rgiid]
       
        # capture results
        Hf[rgiid] = values
        
        if plot_qc:
            # plot qc plot of close-up of raster and flowline along with the actual values along the line
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), layout='constrained', gridspec_kw={"height_ratios":[1, 0.5]})
            h.plot(ax=ax[0])
            gdf = gpd.GeoDataFrame().from_features([feature])
            gdf.plot(ax=ax[0], color='red', marker='o')
            feat_rgi.boundary.plot(ax=ax[0],edgecolor="white", linewidth=1)
            ax[0].set_aspect('equal')
            ax[0].set_title(f"Millan {rgiid}, {feat_rgi.iloc[0].Name}; Area = {feat_rgi.iloc[0].Area:.0f} km2")
            ax[1].plot(values)
            plt.savefig(Path(ROOT,"src/scripts/extract_millan_thickness/qc/",f"{rgiid}.png"))
            plt.close(fig)
        
    logger.info('Closing file.')
    H.close()
    logger.info('Complete!')
    
# clean up and write to file
p = Path(ROOT, "src/scripts/extract_millan_thickness/millan_hf.json")
with open(p, 'w') as f:
    json.dump(Hf, f)
