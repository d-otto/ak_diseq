# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from data import load_rgi
from thickness import load_flowline_geom

class glacierData:
    def __init__(self, filter=True):
        rgi = load_rgi()
        self.rgi = pd.DataFrame(rgi)
        self.rgi['delz'] = self.rgi['Zmax'] - self.rgi['Zmin']# take out of geodataframe

        f_geom = load_flowline_geom()
        cols_to_keep = [col for col in f_geom.columns if col not in self.rgi.columns]
        f_geom = f_geom.loc[:, cols_to_keep]
        self.rgi = self.rgi.merge(f_geom, left_on='RGIId', right_on='RGIID', how='right')

        # filter glaciers from the analysis
        if filter:
            idx_big = (self.rgi['Area'] > 1) & (self.rgi['delz'] > 250)
            idx_len = self.rgi['Lmax'] > 0  # drop any glaciers with negative (aka missing) length
            idx_term_type = self.rgi['TermType'].isin([0, 2])  # Land and lake terminating
            self.mask = idx_big & idx_len & idx_term_type
            self.rgi = self.rgi[self.mask]

        # constants
        self.rho = 916.8
        self.g = 9.81
        self.f = 0.8
        self.dbdz = 0.0065
        self.sb = np.ones(len(self.rgi)) * 1.5e5


    def calc_response_time(self, use_hh=False, dbdz=None):
        if dbdz:
            self.dbdz = dbdz

        # importing/convenience renaming object attributes
        Zela = self.rgi.Zela
        Zmin = self.rgi.Zmin
        slopef = self.rgi.slope_below
        dbdz = self.dbdz * (1000 / self.rho)  # convert to ice equivalent

        # calc bt and thickness
        if use_hh:
            self.rgi.insert(0, 'hf', self.sb / (self.f * self.rho * self.g * np.sin(slopef)))  # avg flowline thickness
        else:
            ensemble_lower = self.rgi.apply(lambda x: np.array(x['ensemble'][0])[x['Zela_idx']:],
                                                        axis=1)  # ensemble thickness below the ELA
            self.rgi.insert(0, 'hf', ensemble_lower.map(np.mean))  # avg flowline thickness below the ELA
            self.rgi.insert(0, 'hh', self.sb / (self.f * self.rho * self.g * np.sin(slopef)))  # avg flowline thickness

        self.rgi.insert(0, 'bt', -dbdz * (Zela - Zmin))  # for vertical gradient
        self.rgi.insert(0, 'tau', np.array(-self.rgi.hf / self.rgi.bt, ndmin=2, dtype='float').reshape(1,
                                                                                                       -1).flatten())  # response time
        # filter crazy large tau for small glaciers
        mask = self.rgi.tau / self.rgi.Area > 200
        self.rgi = self.rgi.loc[~mask]

    def calc_diseq(self, t=140):
        # disequilibrium 3-stage
        eps = 1 / np.sqrt(3)
        feq = 1 - (3 * eps * self.rgi.tau) / t * (1 - np.exp(-t / (eps * self.rgi.tau))) + np.exp(
            -t / (eps * self.rgi.tau)) * (
                          t / (2 * eps * self.rgi.tau) + 2)
        self.rgi.insert(0, 'feq', feq)

    def get_prism(self):
        fp = r'~/src/USGS/glacier_data/prism_ak/ppt/pptanl'
        self.d['prism_ppt'] = get_prism_ak(fp, self.d, lon='CenLon', lat='CenLat')
        self.d['prism_ppt'] = self.d['prism_ppt'] / 100  # now in mm

        fp = r'~/src/USGS/glacier_data/prism_ak/tmean/tmeananl'
        self.d['prism_temp'] = get_prism_ak(fp, self.d, lon='CenLon', lat='CenLat')
        self.d['prism_temp'] = self.d['prism_temp'] / 100  # now in degrees C

    def calc_dist_to_coast(self):
        # Get centeroid distance to coast
        import geopandas as gpd

        fp = r"~/src/USGS/glacier_data/Alaska_Coastline"
        coast = gpd.read_file(fp, crs='EPSG:4326')
        coast['geometry'] = coast['geometry'].exterior
        coastline = gpd.GeoSeries(coast['geometry'].unary_union).set_crs(epsg=4326)
        coastline = coastline.to_crs(epsg=32606).iloc[0]

        # rgi as geodataframe via wkt
        # rgi_centeroids = {r['RGIId']: f" POINT({r['CenLat']} {r['CenLon']})" for idx, r in rgi.iterrows()}
        rgi_centeroids = [f" POINT({r['CenLon']} {r['CenLat']})" for idx, r in self.rgi.iterrows()]
        rgi_centeroids = gpd.GeoSeries.from_wkt(rgi_centeroids, index=self.rgi['RGIId'], crs='EPSG:4326')
        rgi_centeroids = rgi_centeroids.to_crs(epsg=32606)

        dist = [centeroid.distance(coastline) for centeroid in rgi_centeroids]

        self.rgi['dist_from_coast'] = dist


class get_prism_ak:
    def __call__(self, fp, mb, lon='POINT_LON', lat='POINT_LAT', show_plot=False):
        self.show_plot = show_plot
        self.mb = mb
        self.lon = lon
        self.lat = lat

        if isinstance(fp, str):
            return self.pull_value(fp)
        elif isinstance(fp, list):
            obs = []
            for f in fp:
                obs.append(self.pull_value(self, f))
            return np.mean(obs)

    def pull_value(self, f):
        with rasterio.open(f) as prism:
            src_crs = prism.crs
            # src_crs = rasterio.crs.CRS.from_wkt(src_crs.wkt)
            p_3338 = Proj('EPSG:3338')
            p_4326 = Proj('EPSG:4326')
            x, y = p_3338(self.mb[self.lon].to_list(), self.mb[self.lat].to_list())

            # verify visually
            x, y = p_3338(self.mb[self.lon].to_list(), self.mb[self.lat].to_list())
            rows, cols = rasterio.transform.rowcol(prism.transform, x, y)

            raster = prism.read(1)
            # raster = np.flipud(raster)
            raster[rows, cols] = 1000000
            # x_grid, y_grid = np.mgrid[0:raster.shape[0], 0:raster.shape[1]]

            if self.show_plot:
                fig = go.Figure()
                fig.add_traces([
                    go.Heatmap(z=raster, zmin=0, zmax=1000000, colorscale='viridis'),
                    go.Scattergl(x=cols, y=rows, mode='markers', text=self.mb['NAME_x'],
                                 hovertemplate="%{text}<br />%{customdata}", customdata=mb['YEAR'],
                                 marker_color='white'),
                ])
                fig.update_layout(dict(
                    # width=1000,
                    # height=800
                ))
                fig.show()

            return [x[0] for x in rasterio.sample.sample_gen(prism, zip(x, y), 1)]
