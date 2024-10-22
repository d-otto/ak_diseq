# -*- coding: utf-8 -*-

from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Proj

from ak_diseq import ROOT

RGI_AK_lat_min = 52.06300

#todo: replace with pathlib paths

#%% Load RGI

def load_rgi():
    fp = Path(ROOT, "data/rgi60/01_rgi60_Alaska")
    rgi = gpd.read_file(fp)
    

    
    return rgi

def load_rgi_hypsometry():
    # load AK RGI hypsometry
    fp = Path(ROOT, "data/rgi60/01_rgi60_Alaska/01_rgi60_Alaska_hypso.csv")
    rgi_hyp = pd.read_csv(fp)
    rgi_hyp.columns = [x.strip(' ') for x in rgi_hyp.columns]

    # adapt hyps RGIId to RGI6
    rgi_hyp['RGIId'] = rgi_hyp['RGIId'].str.slice_replace(3, 4,
                                                          '6')  # slice from pos 3 to pos 4 and replace with the char '6'

    return rgi_hyp


#%% Load MB data
def load_mb():
    rgi = load_rgi()
    
    # get WGMS id -> RGIID table
    fp = Path(ROOT, "data/DOI-WGMS-FoG-2020-08/WGMS-FoG-2020-08-A-GLACIER.csv")
    wgms = pd.read_csv(fp, encoding='iso-8859-1')
    wgms = wgms.loc[wgms['GLACIER_REGION_CODE'] == 'ALA', :]
    
    fp = Path(ROOT, "data/DOI-WGMS-FoG-2020-08/WGMS-FoG-2020-08-AA-GLACIER_ID_LUT.csv")
    wgms_to_rgi = pd.read_csv(fp, encoding='iso-8859-1')  # latin-1 encoding
    wgms_to_rgi = wgms_to_rgi[(wgms_to_rgi['POLITICAL_UNIT'] == 'US')].drop(columns=[col for col in wgms_to_rgi.columns if
                                                                                     col not in ['WGMS_ID', 'NAME',
                                                                                                 'RGI_ID', 'GLIMS_ID',
                                                                                                 'POLITICAL_UNIT']])
    wgms_to_rgi = wgms_to_rgi.merge(wgms, on=['WGMS_ID', 'POLITICAL_UNIT', 'NAME'], how='right')
    wgms_to_rgi = wgms_to_rgi.rename(columns={'RGI_ID': 'RGIId', 'GLIMS_ID': 'GLIMSId'})
    wgms_to_rgi['RGIId'] = wgms_to_rgi['RGIId'].str.replace('RGI50', 'RGI60')  # messy adapt RGI5 to RGI6
    
    
    # Attempt to smoosh the WGMS names to RGI names
    def clean_glacier_name(wgms_name):
        if wgms_name is np.nan:
            return np.nan
    
        rgi_name = wgms_name.title()
        # apologies if anyone other than me uses this
        # matches 3 groups: 1) anything in parenthesis and the parenthesis 2) anything that isn't a letter or a space, and 3) any 2 character word with a space before it  
        rgi_name = re.sub(r'(\([^)]*\))|([^a-zA-Z -])|(\b[a-zA-Z0-9]{1,2}\b)', '', rgi_name)
        rgi_name = re.sub(r'(^ *)|([ ]{2,})?|( *$)', '',
                          rgi_name)  # remove any moe than 1 consecutive space or leading/trailing spaces 
        words = rgi_name.split(' ')
        if words[-1] == 'Glacier':
            return rgi_name
        else:
            return rgi_name + ' Glacier'
    
    wgms_to_rgi['Name'] = wgms_to_rgi['NAME'].apply(clean_glacier_name)
    
    
    #%%
    # Merge
    wgms_rgi = rgi.merge(wgms_to_rgi, how='right')
    
    # load WGMS balance sheet
    fp = Path(ROOT, "data/DOI-WGMS-FoG-2020-08/WGMS-FoG-2020-08-EEE-MASS-BALANCE-POINT.csv")
    mb = pd.read_csv(fp, encoding='iso-8859-1')  # latin-1 encoding
    mb['Name'] = mb['NAME'].apply(clean_glacier_name)
    #mb.loc[mb['Name'] == 'Yakutat East Glacier', mb['Name']] = 'East Yakutat Glacier'  # resolve discrepancy between WGMS and RGI
    mb['Name'] = mb['Name'].str.replace('Yakutat East Glacier', 'East Yakutat Glacier')
    mb = mb[mb['POLITICAL_UNIT'] == 'US']
    mb['FROM_DATE'] = pd.to_datetime(mb['FROM_DATE'].astype(str), errors='coerce')
    mb['TO_DATE'] = pd.to_datetime(mb['TO_DATE'].astype(str), errors='coerce')
    mb = mb.merge(rgi, how='left')
    mb['point_swe_m'] = mb['POINT_BALANCE']/1000
    
    
    #%% Load JIF data
    
    fp = Path(ROOT, "data/mb/JIF_point_mb_db.xlsx")
    mb_jif = pd.read_excel(fp)
    mb_jif['start_date'] = pd.to_datetime(mb_jif['start_date'])
    mb_jif['end_date'] = pd.to_datetime(mb_jif['end_date'])
    col_map = {
        'name': 'Name',
        'stake': 'POINT_ID',
        'elevation': 'POINT_ELEVATION',
        'balance': 'point_swe_m',
        'error': 'POINT_BALANCE_UNCERTAINTY',
        'season': 'BALANCE_CODE',
        'start_date': 'FROM_DATE',
        'end_date': 'TO_DATE',
        'notes': 'REMARKS',
        'latitude': 'POINT_LAT',
        'longitude': 'POINT_LON'
    }
    mb_jif = mb_jif.rename(columns=col_map)
    mb_jif['YEAR'] = mb_jif['FROM_DATE'].dt.year
    taku_tributaries = ['Matthes Glacier', 'Demorest Glacier', 'Vaughan Lewis Glacier', 'Norris Glacier']
    mb_jif.loc[mb_jif['Name'].isin((taku_tributaries)), 'Name'] = 'Taku Glacier'  # taku tributaries do not have RGIId's
    mb_jif = mb_jif.loc[mb_jif['Name'] != 'Llewellyn Glacier', :]  # filter Llewellyn data because annual balance dates are not known
    mb_jif = mb_jif.merge(rgi, how='left', on='Name')
    
    
    #%% Load Kahiltna pt. 1
    
    fp = Path(ROOT, "data/mb/KAH_20102011_stakedata_databaseformat_JY.xlsx")
    mb_kahiltna = pd.read_excel(fp)
    mb_kahiltna['start_date'] = [f"{x['start_year']}-{x['start_month']}-{x['start_day']}" for _, x in mb_kahiltna.iterrows()]
    mb_kahiltna['start_date'] = pd.to_datetime(mb_kahiltna['start_date'])
    mb_kahiltna['end_date'] = [f"{x['end_year']}-{x['end_month']}-{x['end_day']}" for _, x in mb_kahiltna.iterrows()]
    mb_kahiltna['end_date'] = pd.to_datetime(mb_kahiltna['end_date'])
    col_map = {
        'glacier': 'Name',
        'stake_name': 'POINT_ID',
        'elev (m)': 'POINT_ELEVATION',
        'balance (m w.e.)': 'point_swe_m',
        'error (m w.e.)': 'POINT_BALANCE_UNCERTAINTY',
        'season': 'BALANCE_CODE',
        'start_date': 'FROM_DATE',
        'end_date': 'TO_DATE',
        'notes': 'REMARKS',
        'lat': 'POINT_LAT',
        'lon': 'POINT_LON',
        'end_year': 'YEAR'
    }
    mb_kahiltna = mb_kahiltna.rename(columns=col_map)
    mb_kahiltna['FROM_DATE'] = pd.to_datetime(mb_kahiltna['FROM_DATE'].astype(str), errors='coerce')
    mb_kahiltna['TO_DATE'] = pd.to_datetime(mb_kahiltna['TO_DATE'].astype(str), errors='coerce')
    mb_kahiltna = mb_kahiltna.merge(rgi, how='left', on='Name')
    
    
    #%% Load Kahiltna pt. 2
    
    fp = Path(ROOT, "data/mb/Index_site_KAH.xlsx")
    mb_kah = pd.read_excel(fp)
    mb_kah = mb_kah.melt(id_vars=[col for col in mb_kah.columns if col not in ['bw', 'bs', 'bn']], value_vars=['bw', 'bs', 'bn'], value_name='point_swe_m', var_name='BALANCE_CODE')
    
    ak_crs = Proj("+proj=utm +zone=9 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    mb_kah['POINT_LON'], mb_kah['POINT_LAT'] = ak_crs(mb_kah['UTM E mean (m)'].values, mb_kah['UTM N mean (m)'].values, inverse=True)
    
    
    col_map = {
        'elev mean (m)': 'POINT_ELEVATION',
    }
    mb_kah = mb_kah.rename(columns=col_map)
    mb_kah['Name'] = 'Kahiltna Glacier'
    mb_kah = mb_kah.merge(rgi, how='left', on='Name')
    
    
    #%% Load Eklutna
    
    p = Path(ROOT, r'data/mb/eklutnaGlacierSurfaceMassBalance_2008_2015/eklutnaGlacierSurfaceMassBalance_2008_2015_sass.csv')
    mb_ek = pd.read_csv(p, header=0)
    mb_ek = mb_ek[1:]
    
    ak_crs = Proj("+proj=utm +zone=6 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    mb_ek['easting'], mb_ek['northing'] = ak_crs(mb_ek['easting'].values, mb_ek['northing'].values, inverse=True)
    
    #mb_ek['day'] = pd.to_numeric(mb_ek['day'])
    mb_ek = mb_ek.apply(pd.to_numeric, errors='ignore')
    mb_ek.loc[mb_ek.day > 200, 'BALANCE_CODE'] = 'BA'
    mb_ek.loc[mb_ek.day <= 200, 'BALANCE_CODE'] = 'BW'
    mb_ek['swe'] = mb_ek['point balance'] * mb_ek['density']/1000
    
    mb_ek = mb_ek.sort_values(by='BALANCE_CODE')
    concat = []
    for k, g in mb_ek.groupby(['year', 'stake']):
        ba = g['swe'].iloc[0] + g['swe'].iloc[1]
        g.iloc[0]['swe'] = ba
        concat.append(g.iloc[0])
    mb_ek = pd.DataFrame(concat)
    
    col_map = {
        'stake': 'POINT_ID',
        'elevation': 'POINT_ELEVATION', 
        'swe': 'point_swe_m',
        'year': 'YEAR',
        'easting': 'POINT_LON',
        'northing': 'POINT_LAT',
        'BALANCE_CODE': 'BALANCE_CODE'
    }
    mb_ek = mb_ek.loc[:, col_map.keys()]
    mb_ek = mb_ek.rename(columns=col_map)
    
    mb_ek['Name'] = 'Eklutna Glacier'
    mb_ek = mb_ek.merge(rgi, how='left', on='Name')
    
    #%% Exit glacier
    
    p = Path(ROOT, 'data/mb/KEFJ_Basic_Annual_MassBalance_Summary.xlsx')
    mb_ex = pd.read_excel(p, header=1, na_values=['na'])
    mb_ex = mb_ex[1:-2]
    
    col_map = {
        'Site': 'POINT_ID',
        'Elevation* (m)': 'POINT_ELEVATION', 
        'Easting': 'POINT_LON',
        'Northing': 'POINT_LAT',
    }
    mb_ex = mb_ex.rename(columns=col_map)
    mb_ex = pd.melt(mb_ex, id_vars=['POINT_ID', 'POINT_ELEVATION', 'POINT_LON', 'POINT_LAT'], var_name='YEAR', value_name='point_swe_m')
    
    mb_ex = mb_ex.apply(pd.to_numeric, errors='ignore')
    mb_ex['Name'] = 'Exit Glacier'
    mb_ex['BALANCE_CODE'] = 'BA'
    mb_ex = mb_ex.merge(rgi, how='left', on='Name')
    
    
    #%% Concat mb data
    
    mb = pd.concat([mb, mb_jif, mb_kahiltna, mb_kah, mb_ek, mb_ex], ignore_index=True)
    mb = mb.dropna(how='all', subset=['point_swe_m'])  # drop rows somehow missing mass balance data
    mb = mb.dropna(how='any', subset=['POINT_LAT', 'POINT_LON', 'BALANCE_CODE'])  # drop rows without location data
    # fill in columns for glacier-wide parameters for merged records 
    # cols_to_fill = ['']
    # mb = mb.groupby('Name').fillna()
    
    # recode the melt-season balances to BA. Not exact but better than leaving them in a different group while balance code is in a regression
    balance_code_map = {
        'IN': 'BA',
        'a': 'BA',
        'w': 'BW',
        'BA': 'BA',
        'BW': 'BW',
        'bn': 'BA',
    }
    mb['BALANCE_CODE'] = mb['BALANCE_CODE'].map(balance_code_map)
    mb = mb.loc[(mb['POINT_LAT'] > RGI_AK_lat_min), :]
    mb['YEAR'] = pd.to_numeric(mb['YEAR'])
    
    # replace mislabeled balance codes with the correct code if it is in the remarks
    # only for black rapids
    
    mask = mb.Name == 'Black Rapids Glacier'
    bs_mask = mb.loc[mask, 'REMARKS'].str.strip().str.startswith('BS')
    bw_mask = mb.loc[mask, 'REMARKS'].str.strip().str.startswith('BW')
    mb.loc[mask & bs_mask, 'BALANCE_CODE'] = 'BS'
    mb.loc[mask & bw_mask, 'BALANCE_CODE'] = 'BW'
    
    
    #%% flag bad data
    
    def omit_pts(df, name, year, bx=None, elev_min=None, elev_max=None):
        name_col = 'Name'
        year_col = 'YEAR'
        name_mask = df[name_col] == name
        if hasattr(year, '__iter__'):
            year_mask = df[year_col].isin(year)
        else:
            year_mask = df[year_col] == year 
        
        if bx in ['BW', 'BA']:
            bx_mask = df['BALANCE_CODE'] == bx
        else:
            bx_mask = len(df)*True
        
        if elev_min or elev_max:
            if not elev_min:
                elev_min = df['POINT_ELEVATION'].min()
            elif not elev_max:
                elev_max = df['POINT_ELEVATION'].max()
            
            elev_mask = (df['POINT_ELEVATION'] > elev_min) & (df['POINT_ELEVATION'] <= elev_max)
            df.loc[name_mask & year_mask & elev_mask & bx_mask, 'omit'] = True
        else:
            df.loc[name_mask & year_mask & bx_mask, 'omit'] = True
            
        return df
    
        
    mb['omit'] = False
    mb = omit_pts(mb, 'Taku Glacier', 2003, elev_max=900)  # 2 crazy points at low elevation w/ high positive balances.
    mb = omit_pts(mb, 'Mendenhall Glacier', 2009)  # only 2 pts ~80 m apart but 2 m SWE difference creating the wrong direction slope.
    mb = omit_pts(mb, 'Lemon Creek Glacier', [1957, 1966, 1979, 1984, 1985, 1990, 1996, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013], bx='BA')  # Don't cross 0, seem to have nonsensical slopes, sometimes with the wrong sign.
    mb = omit_pts(mb, 'Lemon Creek Glacier', 2006, bx='BW')  # Wrong slope, clustered elevations
    mb = omit_pts(mb, 'Black Rapids Glacier', np.arange(1972, 1990))
    mb = omit_pts(mb, 'Gilkey Glacier', np.arange(1900, 2022))  # remove all gilkey
    mb = omit_pts(mb, 'Taku Glacier', np.arange(1900, 2022))  # remove all taku

    return mb

#%%

def load_zeller():
    elas = pd.read_csv(Path(ROOT, 'data/zeller/alaska_elas.csv'))
    return elas

def load_flowlines(rgiids):
    shp_path = Path(ROOT, 'data/RGI01_flowlines')
    flowlines = gpd.read_file(shp_path)
    flowlines = flowlines.loc[flowlines['MAIN'] == 1, :]
    flowlines = flowlines.loc[flowlines.RGIID.isin(rgiids)]
    
    return flowlines

def load_millan_thickness():
    p = Path(ROOT, r'src/scripts/extract_millan_thickness/millan_hf.json')
    with open(p, 'r') as f:
        Hf = json.load(f)
    return Hf

def load_farinotti_thickness():
    p = Path(ROOT, r'src/scripts/extract_farinotti_thickness/farinotti_hf.json')
    with open(p, 'r') as f:
        Hf = json.load(f)
    return Hf

def load_flowline_elevations():
    p = Path(ROOT, r'src/scripts/extract_flowline_elevations/flowline_elevations.json')
    with open(p, 'r') as f:
        Zs = json.load(f)
    return Zs