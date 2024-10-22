# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
from .data import load_rgi, load_rgi_hypsometry, load_zeller, load_flowlines, load_millan_thickness, load_farinotti_thickness, load_flowline_elevations
import gm

class glacierData:
    def __init__(self, filter=True, thickness_method="millan", aar=0.6):
        rgi = load_rgi()
        self.rgi = pd.DataFrame(rgi)
        self.rgi['delz'] = self.rgi['Zmax'] - self.rgi['Zmin']# take out of geodataframe

        # filter glaciers from the analysis
        if filter:
            idx_big = (self.rgi['Area'] > 1) & (self.rgi['delz'] > 250)
            idx_len = self.rgi['Lmax'] > 0  # drop any glaciers with negative (aka missing) length
            idx_term_type = self.rgi['TermType'].isin([0, 2])  # Land and lake terminating
            self.mask = idx_big & idx_len & idx_term_type
            self.rgi = self.rgi[self.mask]

        # f_geom = load_flowline_geom()
        # cols_to_keep = [col for col in f_geom.columns if col in ['RGIID', 'ensemble', 'h3', 'h2', 'h1', 'z', 'l_above', 'l_below', 'sum', 'aalr', 'slope_above', 'slope_below']]
        # f_geom = f_geom.loc[:, cols_to_keep]
        # self.rgi = self.rgi.merge(f_geom, left_on='RGIId', right_on='RGIID', how='right')
        
        self.get_ela(aar=aar)
        self.flowlines = load_flowlines(self.rgi.RGIId)
        self.Hf = self.get_thickness(thickness_method)
        self.rgi = self.rgi.merge(self.Hf, left_on='RGIId', right_index=True)
        
        # constants
        self.rho = 916.8
        self.g = 9.81
        self.f = 0.8
        self.dbdz = 0.0065
        self.sb = np.ones(len(self.rgi)) * 1.5e5
        
    def get_thickness(self, method):
        Zs = load_flowline_elevations()
        Zs = {k: v for k, v in Zs.items() if k in self.rgi.RGIId.tolist()}
        zelas = pd.Series(self.rgi.Zela, index=self.rgi.RGIId)
        
        if method == "millan":
            H = load_millan_thickness()
            H = {k: v for k, v in H.items() if k in self.rgi.RGIId.tolist()}

        elif method == "farinotti":
            H = load_farinotti_thickness()
            H = {k: v for k, v in H.items() if k in self.rgi.RGIId.tolist()}
        elif method == "HH":
            #Hf =  self.sb / (self.f * self.rho * self.g * np.sin(slopef)))  # avg flowline thickness
            return None
                       
        # get the index at which to split the data
        split_idxs = [len(zs) - np.searchsorted(zs[::-1], zela) for zs, zela in zip(Zs.values(), zelas)]
        # split the data and take the average
        Hf = {k: np.mean(h[idx:]) for (k, h), idx in zip(H.items(), split_idxs)} # mean thickness below the ELA
        Hf = pd.Series(Hf, name='hf')
        return Hf

    def get_ela(self, aar=0.6):
        if aar != "zeller":
            # uses the aar argument
            rgi_hyp = load_rgi_hypsometry()
            hyp = rgi_hyp.iloc[:, 3:]  # cut off metadata
            hyp = hyp.cumsum(axis=1) / 1000
            # get ela alt
            mask = hyp >= (1 - aar)
            idx_above = mask.idxmax(axis=1)  # first altitude where the area % exceeds the threshold
            rgi_hyp.loc[:, 'Zela'] = (idx_above.astype(int) - 25)  # halfway between buckets
            rgi_hyp['abl_area'] = rgi_hyp['Area'] * (1 - aar)
            rgi_hyp['accu_area'] = rgi_hyp['Area'] * aar

            self.rgi = self.rgi.merge(rgi_hyp.loc[:, ['RGIId', 'Zela', 'abl_area', 'accu_area']], on='RGIId', how='left')
        
        else:
            elas = load_zeller()
            
            # fill the missing values for glaciers which are too small to be included in Zeller
            aar = 0.2  # based on the mean of the smallest glaciers in the dataset
            rgi_hyp = load_rgi_hypsometry()
            hyp = rgi_hyp.iloc[:, 3:]  # cut off metadata
            hyp = hyp.cumsum(axis=1) / 1000
            # get ela alt
            mask = hyp >= (1 - aar)
            idx_above = mask.idxmax(axis=1)  # first altitude where the area % exceeds the threshold
            rgi_hyp.loc[:, 'ela'] = (idx_above.astype(int) - 25)  # halfway between buckets
            # Merge the dataframes
            rgi_hyp = pd.merge(rgi_hyp, elas, on='RGIId', how='left')

            # Overwrite values from 'elas' to 'rgi_hyp' where available
            rgi_hyp['Zela'] = rgi_hyp['ela_y'].fillna(rgi_hyp['ela_x'])
            rgi_hyp['aar'] = rgi_hyp['aar'].fillna(aar)

            # Drop the '_x' and '_y' suffixes
            rgi_hyp.drop(['ela_x', 'ela_y'], axis=1, inplace=True)
            
            self.rgi = self.rgi.merge(rgi_hyp.loc[:, ['RGIId', 'Zela', 'aar', 'hyps_ela', 'hyps_aar']], on='RGIId', how='left')
            

    def calc_response_time(self, use_hh=False, dbdz=None):
        if dbdz:
            self.dbdz = dbdz

        # importing/convenience renaming object attributes
        Zela = self.rgi.Zela
        Zmin = self.rgi.Zmin
        slopef = self.rgi.Slope
        dbdz = self.dbdz * (1000 / self.rho)  # convert to ice equivalent

        # calc bt and thickness
        if use_hh:
            self.rgi.insert(0, 'hf', self.sb / (self.f * self.rho * self.g * np.sin(slopef)))  # avg flowline thickness
        else:
            self.rgi.insert(0, 'hh', self.sb / (self.f * self.rho * self.g * np.sin(slopef)))  # avg flowline thickness

        self.rgi.insert(0, 'bt', -dbdz * (Zela - Zmin))  # for vertical gradient
        self.rgi.insert(0, 'tau', np.array(-self.rgi.hf / self.rgi.bt, ndmin=2, dtype='float').reshape(1,
                                                                                                       -1).flatten())  # response time
        # filter crazy large tau for small glaciers
        mask = (self.rgi.tau / self.rgi.Area < 200) & (self.rgi.hf > 0)
        self.rgi = self.rgi.loc[mask]

    def calc_linear_feq(self, t=140):
        # disequilibrium 3-stage
        eps = 1 / np.sqrt(3)
        feq = 1 - (3 * eps * self.rgi.tau) / t * (1 - np.exp(-t / (eps * self.rgi.tau))) + np.exp(
            -t / (eps * self.rgi.tau)) * (
                          t / (2 * eps * self.rgi.tau) + 2)
        self.rgi.insert(0, 'feq', feq)


    def calc_feq(self):
        # calculate feq for linear, GWI, and ETCW scenarios using the 3-stage model
        idx = np.arange(1880, 2021, 1)

        # linear trend
        T_linear = np.linspace(0, 1.2, 142)[1:]

        # Synthetic GWI
        ts1 = np.linspace(0, 0.15, 86)[:-1]
        ts2 = np.linspace(0.15, 1.2, 56)
        T_gwi = np.concatenate([ts1, ts2])

        # ETCW
        ts1 = np.linspace(0, 0.35, 61)[:-1]
        ts2 = np.linspace(0.35, 0.05, 26)[:-1]
        ts3 = np.linspace(0.05, 1.2, 56)
        T_etcw = np.concatenate([ts1, ts2, ts3])

        # run 3 stage model
        Ts = {"linear": T_linear, "gwi": T_gwi, "etcw": T_etcw}
        output = {"linear": [], "gwi": [], "etcw": []}
        for scenario, T in Ts.items():
            melt_factor = -0.65
            b_p = T * melt_factor
            for i, (_, g) in tqdm(enumerate(self.rgi.iterrows())):
                tau = g['tau']
                params_3s = dict(
                    dt=0.01,
                    Atot=g['Area'],
                    W=g['Area'] / g['Lmax'],
                    L=g['Lmax'],
                    H=g['hf'],
                    bt=0,
                    b_p=b_p,
                    ts=idx,
                )
                res = gm.gm3s(tau=tau, **params_3s).run().to_pandas().reset_index()
                res = res.iloc[[-1]]
                res['tau'] = tau
                res['RGIId'] = g['RGIId']
                res['name'] = g['Name']
                res[f'feq_{scenario}'] = res.Lp / (res.Lp_eq)
                output[scenario].append(res)
            output[scenario] = pd.concat(output[scenario], ignore_index=True)
        print("merging output...")
        merged_output = output["linear"].merge(output["gwi"], how="outer", on="RGIId")
        merged_output = merged_output.merge(output["etcw"], how="outer", on="RGIId")
        merged_output = merged_output.set_index("RGIId")
        merged_output = merged_output.loc[:, [col for col in merged_output.columns if col.startswith("feq_")]]
        self.rgi = self.rgi.merge(merged_output, how="left", left_on="RGIId", right_index=True)
        
        return None

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

    
    # def split_flowlines_by_ela(self, ):
    #     lengths = hf.apply(lambda x: split_flowline_length(x['geometry'], x['z'], x['Zela']), axis=1)
    #     lengths = pd.DataFrame(lengths.to_list(), columns=['l_above', 'l_below'], index=hf.index)
    #     lengths['sum'] = lengths['l_above'] + lengths['l_below']
    #     lengths['aalr'] = lengths['l_above'] / (lengths['l_above'] + lengths['l_below'])
    # 
    #     hf = hf.join(lengths)
    #     hf['slope_above'] = np.arctan((hf.Zela - hf.Zmin) / hf.l_below)
    #     hf['slope_below'] = np.arctan((hf.Zmax - hf.Zela) / hf.l_above)
    #     hf['Zela_idx'] = hf.apply(lambda x: nearest_idx(x['z'], x['Zela']), axis=1)
    #     return None
    




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


