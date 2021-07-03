# https://ipinfo.io/128.238.88.64
from pretreatments import io_operations as io
import folium
import geoip2.database
import os
import geocoder
import numpy as np
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import somoclu
from collections import Counter
'''
reding users services files
'''
class Clust:
    
    def __init__(self):
        self.users = io.read_user_list()
        self.services = io.read_service_list()
        self.map = folium.Map(location=[10, 0], tiles="Mapbox Bright", zoom_start=2)
        
        self.reader_asn = geoip2.database.Reader(os.path.join(os.path.dirname(__file__), '../dataset/geo/GeoLite2-ASN.mmdb'))
        self.reader_city = geoip2.database.Reader(os.path.join(os.path.dirname(__file__), '../dataset/geo/GeoLite2-City.mmdb'))
        
    def get_geoLite_Info(self, ip):
        results_city = []
        results_asn = []
        try:
            response_city = self.reader_city.city(ip)
            results_city.append(response_city.country.name)
            results_city.append(response_city.city.name)
            results_city.append((response_city.location.latitude))
            results_city.append((response_city.location.longitude))
            results_city.append(response_city.subdivisions.most_specific.name)
            results_city.append(response_city.subdivisions.most_specific.iso_code)
            results_city.append(response_city.postal.code)
        except:
            print('The address %s is not in the city database' % ip)
        try:
            response_asn = self.reader_asn.asn(ip)
            results_asn.append('AS' + str(response_asn.autonomous_system_number))
            results_asn.append(response_asn.autonomous_system_organization)
        except:
            print('The address %s is not in the asn database' % ip)
        
        return results_city, results_asn
    
    '''
    Check users and services values in data set.
    Update asn and gps coordinates for selected IP
    '''
    def check_data_set(self, _type):
        
        if _type == 'users':
            db = self.users
        else:
            db = self.services
       
        ips = db['ip_Address']
        asns = db['asn']
        countries = db['country']
        lats = db['latitude']
        lons = db['longitude']
        
        j_asn = 0
        j_country = 0
        j_lat = 0
        j_lon = 0
        invalid = 0
        
        for i, (ip, asn, country, lat, lon) in enumerate(zip(ips, asns, countries, lats, lons)):
            if str(ip) == 'nan':
                continue
            
            if not str(asn) == 'nan':
                l = asn.find(' ')
                if l == -1:
                    l = len(asn)
                _asn = asn[0:l]
            else:
                _asn = ''
            
            results_city, resuts_asn = self.get_geoLite_Info(ip)
           
            print(i, "\t---- ", ip, " ", _asn, " ", country, " ", lat , " ", lon)
            
            if len(results_city) == 0:
                invalid += 1
            else :    
                if country != results_city[0] :
                    
                    if not results_city[0] is None:
                        j_country += 1
                        print("\t", j_country, "\t\t---->", results_city[0])
                        db.loc[db['ip_Address'] == ip, 'country'] = results_city[0]
                
                if lat != results_city[2]:
                    if not results_city[2] is None:
                        j_lat += 1
                        print("\t", j_lat, "\t\t---->", results_city[2])
                        db.loc[db['ip_Address'] == ip, 'latitude'] = float("{0:.3f}".format(results_city[2]))
                    
                if lon != results_city[3] :
                    if not results_city[3] is None:
                        j_lon += 1
                        print("\t", j_lon, "\t\t---->", results_city[3])
                        db.loc[db['ip_Address'] == ip, 'longitude'] = float("{0:.3f}".format(results_city[3])) 
                
            if len(resuts_asn) == 0:
                invalid += 1
                continue
            if _asn != resuts_asn[0] :
                if i == 3890 :
                    print(i)
                j_asn += 1
                print("\t", j_asn, "\t\t---->", resuts_asn[0])
                db.loc[db['ip_Address'] == ip, 'asn'] = str(resuts_asn[0]) 
                if not resuts_asn[1] is None:
                    db.loc[db['ip_Address'] == ip, 'asn'] += ' ' + resuts_asn[1]
                    
                
        print("%d asn modified" % (j_asn))
        print("%d countries modified" % (j_country))
        print("%d latitudes modified" % (j_lat))
        print("%d longitudes modified" % (j_lon))
        print("%d invalid adresses" % (invalid))
        
        name = "new_" + _type + ".txt"
        db.to_csv(name, sep='\t', encoding="utf-8", header=True, index=False)
     
    '''
    Plot users and services 
    '''            
    def plot_users(self, mapp, _type):
        
        if mapp == None :
            mapp = self.map
        
        if _type == 'users':
            db = self.users
            fill_color = 'blue'
        else:
            db = self.services
            fill_color = 'red'
        
        db = db.fillna(0) 
        lats = db['latitude']
        lons = db['longitude']
        invalid = 0          
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            
            if (lat != 0.)and(lon != 0.):
                
                folium.CircleMarker([lat, lon],
                        fill=True,
                        radius=3,
                        weight=0,
                        fill_color=fill_color).add_to(mapp)
            else :
                invalid += 1
        name = "new_" + _type + ".html"
        mapp.save(os.path.join(os.path.dirname(__file__), '../dataset/graph/' + name))
        print(invalid , " invalid coordinates")
        
    def plot_asn(self):
        df = self.users
        group_asn = df.groupby('country')
        list_country = []
        list_count = []
        for name, group in group_asn:
            list_country.append(name)
            list_count.append(len(group.groupby('asn')))
        print(list_country)
        print(list_count)
        
        
        
        
        df = self.services
        group_asn = df.groupby('country')
        for name, group in group_asn:
            i = list_country.index(name) if name in list_country else -1
            if i == -1:
                list_country.append(name)
                list_count.append(len(group.groupby('asn')))
            else:
                list_count[i] += len(group.groupby('asn'))
                
        print(list_country)
        print(list_count)
        print(len(list_country))
        print(len(list_count))
    
    def mean_matrices(self, qos):
        # rt from 9561229 nan values --> added all matrices 140257 nan values --> add matrice dataset_1 :6936 nan values
        # tp from 14216251 nan values --> added all matrices 161868 nan values --> add matrice dataset_1 :10013 nan values
        data_coef = np.zeros([142, 4500])
        data_vals = np.zeros([142, 4500])
        total_val_null = 0
        pbar = tqdm.tqdm(range(64))
        for slot in pbar:
            data = io.read_max_density_MatrixBySlot(qos, slot)
            data = data.fillna(0)
            indice = np.flatnonzero(data.values)
            temp = np.zeros([142, 4500])
            np.put(temp, indice, 1)
            data_coef += temp
            data_vals += data
            
            data = data.values
            total_val_null += len(data[data == 0])
        
        print("nan values for all data set :\t", total_val_null)
        total_val_null = len(data_vals[data_vals == 0])
        print("nan values before added all matrices   :\t", total_val_null)    

        _data = io.read_small_Matrix(qos)
        
        _data = _data.filter(range(142), axis=0)
        _data = _data.filter(range(4500))
        _data = _data.as_matrix()
        data_vals += _data
        
        total_val_null = len(data_vals[data_vals == 0])
        print("nan values after added matrix dataset1   :\t", total_val_null)
        # 
        indice = np.flatnonzero(_data)
        temp = np.zeros([142, 4500])
        np.put(temp, indice, 1)
        data_coef += temp
        
        data_vals = data_vals / data_coef
        # data_vals= np.nan_to_num(data_vals)
        data_vals[np.isnan(data_vals)] = 0.
        
        if qos == 0:
            name = 'rt_mean_matrix.txt'
        else: 
            name = 'tp_mean_matrix.txt'
        path = os.path.join(os.path.dirname(__file__), '../dataset/dataset1/' + name)
        np.savetxt(path, data_vals, fmt='%.3f')
        
    def som_cluster(self, qos, _type):
        if qos == 0:
            name = 'rt_mean_matrix.txt'
        else: 
            name = 'tp_mean_matrix.txt'
            
        file_data = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + name)
        data = np.loadtxt(file_data)
            
        if _type=='user':
            n_columns = 20
            n_rows = 20
            epochs = n_columns * n_rows * 50
            dim = 142
        else:
            n_columns = 80
            n_rows = 80
            epochs = 10000#n_columns * n_rows * 50
            dim = 4500
            data = data.T
        
        file_bmu = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + str(qos) + '_' + _type + '_bmus')  
        file_codebook = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + str(qos) + '_' + _type + '_codebook') 
        file_clustrs = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + str(qos) + '_' + _type + '_clusters')
        file_classes = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + str(qos) + '_' + _type + '_classes')
        file_png = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + str(qos) + '_' + _type + '_clust.png')
        
        
        
        som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid", compactsupport=False, initialization="pca")
         
        som.train(data, epochs=epochs)
        som.cluster() 
        
        np.save(file_bmu, som.bmus)
        np.save(file_codebook, som.codebook)
        np.save(file_clustrs, som.clusters)
        
        classes = []
        for k in range(dim):
            i = som.bmus[k, 1]
            j = som.bmus[k, 0]
            classes.append(som.clusters[i, j])
        np.save(file_classes, classes)
        
        som.view_umatrix(bestmatches=True, filename=file_png)

    

                    
                
         
clust = Clust()
# clust.check_data_set('users')
# clust.check_data_set('services')
# clust.plot_users(mapp=None, _type='users')
# clust.plot_users(mapp=None, _type='services')
# clust.plot_asn()
# clust.mean_matrices(0)
# clust.mean_matrices(1)
clust.som_cluster(1, 'service')

# qos = 0
# _type = 'user'
# if qos == 0:
#     name = 'rt_mean_matrix.txt'
# else: 
#     name = 'tp_mean_matrix.txt'
#             
# file_classes = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + str(qos) + '_' + _type + '_classes.npy')
# file_codebook = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + str(qos) + '_' + _type + '_codebook.npy')
# file_data = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + name)
# file_bmu = os.path.join(os.path.dirname(__file__), '../dataset/clusters/' + str(qos) + '_' + _type + '_bmus.npy')
#         
# classes = np.load(file_classes)
# initialcodebook = np.load(file_codebook)
# data = np.loadtxt(file_data)
# 
# #som = somoclu.Somoclu(n_columns=20, n_rows=20, initialcodebook=initialcodebook, maptype="toroid", compactsupport=False, data=data)
# #som.load_bmus(file_bmu)
# #som
# #som.view_umatrix()
# 
# a = Counter(classes)
# print(a)
# pos = 0
# # for classe in range(8):
# #     _list, = np.where(classes == classe)
# #     print(classe)
# #     print(_list)
# #     print('---------------------------')
# #     
