# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:02:19 2020

@author: richieBao-caDesign设计(cadesign.cn)
"""

#位于百度地图POI数据的抓取与地理空间点地图/1. 百度地图POI数据的抓取-单个分类实现与地理空间点地图/1.1 单个分类POI爬取
def baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False):
    #所调用的库及辅助文件
    import coordinate_transformation as cc    
    import urllib, json, csv,os,pathlib
    
    '''function-百度地图开放平台POI数据爬取'''
    urlRoot='http://api.map.baidu.com/place/v2/search?' #数据下载网址，查询百度地图服务文档
    #切分检索区域
    xDis=(bound_coordinate['rightTop'][0]-bound_coordinate['leftBottom'][0])/partition
    yDis=(bound_coordinate['rightTop'][1]-bound_coordinate['leftBottom'][1])/partition    
    #判断是否要写入文件
    if poi_fn_list:
        for file_path in poi_fn_list:
            fP=pathlib.Path(file_path)
            if fP.suffix=='.csv':
                poi_csv=open(fP,'w',encoding='utf-8')
                csv_writer=csv.writer(poi_csv)    
            elif fP.suffix=='.json':
                poi_json=open(fP,'w',encoding='utf-8')
    num=0
    jsonDS=[] #存储读取的数据，用于.json格式数据的保存
    #循环切分的检索区域，逐区下载数据
    print("Start downloading data...")
    for i in range(partition):
        for j in range(partition):
            leftBottomCoordi=[bound_coordinate['leftBottom'][0]+i*xDis,bound_coordinate['leftBottom'][1]+j*yDis]
            rightTopCoordi=[bound_coordinate['leftBottom'][0]+(i+1)*xDis,bound_coordinate['leftBottom'][1]+(j+1)*yDis]
            for p in page_num_range:  
                #更新请求参数
                query_dic.update({'page_num':str(p),
                                  'bounds':str(leftBottomCoordi[1]) + ',' + str(leftBottomCoordi[0]) + ','+str(rightTopCoordi[1]) + ',' + str(rightTopCoordi[0]),
                                  'output':'json',
                                 })

                url=urlRoot+urllib.parse.urlencode(query_dic)
                data=urllib.request.urlopen(url)
                responseOfLoad=json.loads(data.read()) 
                if responseOfLoad.get("message")=='ok':
                    results=responseOfLoad.get("results") 
                    for row in range(len(results)):
                        subData=results[row]
                        baidu_coordinateSystem=[subData.get('location').get('lng'),subData.get('location').get('lat')] #获取百度坐标系
                        Mars_coordinateSystem=cc.bd09togcj02(baidu_coordinateSystem[0], baidu_coordinateSystem[1]) #百度坐标系-->火星坐标系
                        WGS84_coordinateSystem=cc.gcj02towgs84(Mars_coordinateSystem[0],Mars_coordinateSystem[1]) #火星坐标系-->WGS84
                        
                        #更新坐标
                        subData['location']['lat']=WGS84_coordinateSystem[1]
                        subData['detail_info']['lat']=WGS84_coordinateSystem[1]
                        subData['location']['lng']=WGS84_coordinateSystem[0]
                        subData['detail_info']['lng']=WGS84_coordinateSystem[0]  
                        
                        if csv_writer:
                            csv_writer.writerow([subData]) #逐行写入.csv文件
                        jsonDS.append(subData)
            num+=1       
            print("No."+str(num)+" was written to the .csv file.")
    if poi_json:       
        json.dump(jsonDS,poi_json)
        poi_json.write('\n')
        poi_json.close()
    if poi_csv:
        poi_csv.close()
    print("The download is complete.")
    return jsonDS


#位于百度地图POI数据的抓取与地理空间点地图/1. 百度地图POI数据的抓取-单个分类实现与地理空间点地图/1.2 将.csv格式的POI数据转换为pandas的DataFrame
def csv2df(poi_fn_csv):
    import pandas as pd
    from benedict import benedict #benedict库是dict的子类，支持键列表（keylist）/键路径（keypath），应用该库的flatten方法展平嵌套的字典，准备用于DataFrame数据结构
    import csv
    '''function-转换.csv格式的POI数据为pandas的DataFrame'''
    n=0
    with open(poi_fn_csv, newline='',encoding='utf-8') as csvfile:
        poi_reader=csv.reader(csvfile)
        poi_dict={}    
        poiExceptions_dict={}
        for row in poi_reader:    
            if row:
                try:
                    row_benedict=benedict(eval(row[0])) #用eval方法，将字符串字典"{}"转换为字典{}
                    flatten_dict=row_benedict.flatten(separator='_') #展平嵌套字典
                    poi_dict[n]=flatten_dict
                except:                    
                    print("incorrect format of data_row number:%s"%n)                    
                    poiExceptions_dict[n]=row
            n+=1
            #if n==5:break #因为循环次数比较多，在调试代码时，可以设置停止的条件，节省时间与方便数据查看
    poi_df=pd.concat([pd.DataFrame(poi_dict[d_k].values(),index=poi_dict[d_k].keys(),columns=[d_k]).T for d_k in poi_dict.keys()], sort=True,axis=0)
    # print("_"*50)
    for col in poi_df.columns:
        try:
            poi_df[col]=pd.to_numeric(poi_df[col])
        except:
            pass
            #print("%s data type is not converted..."%(col))
    print("_"*50)
    print(".csv to DataFrame is completed!")
    #print(poi_df.head()) #查看最终DataFrame格式下POI数据
    #print(poi_df.dtypes) #查看数据类型
    return poi_df


#位于百度地图POI数据的抓取与地理空间点地图/1.1 多个分类POI爬取|1.2 批量转换.csv格式数据为GeoDataFrame/1.2.1 定义提取文件夹下所有文件路径的函数
def filePath_extraction(dirpath,fileType):
    import os
    '''以所在文件夹路径为键，值为包含该文件夹下所有文件名的列表。文件类型可以自行定义 '''
    filePath_Info={}
    i=0
    for dirpath,dirNames,fileNames in os.walk(dirpath): #os.walk()遍历目录，使用help(os.walk)查看返回值解释
       i+=1
       if fileNames: #仅当文件夹中有文件时才提取
           tempList=[f for f in fileNames if f.split('.')[-1] in fileType]
           if tempList: #剔除文件名列表为空的情况,即文件夹下存在不为指定文件类型的文件时，上一步列表会返回空列表[]
               filePath_Info.setdefault(dirpath,tempList)
    return filePath_Info


#频数分布计算
def frequency_bins(df,bins):
    import pandas as pd
    '''function-频数分布计算'''

    #A-组织数据
    column_name=df.columns[0]
    column_bins_name=df.columns[0]+'_bins'
    df[column_bins_name]=pd.cut(x=df[column_name],bins=bins,right=False) #参数right=False指定为包含左边值，不包括右边值。
    df_bins=df.sort_values(by=[column_name]) #按照分割区间排序
    df_bins.set_index([column_bins_name,df_bins.index],drop=False,inplace=True) #以price_bins和原索引值设置多重索引，同时配置drop=False参数保留原列。
    #print(df_bins.head(10))

    #B-频数计算
    dfBins_frequency=df_bins[column_bins_name].value_counts() #dropna=False  
    dfBins_relativeFrequency=df_bins[column_bins_name].value_counts(normalize=True) #参数normalize=True将计算相对频数(次数) dividing all values by the sum of values
    dfBins_freqANDrelFreq=pd.DataFrame({'fre':dfBins_frequency,'relFre':dfBins_relativeFrequency})
    #print(dfBins_freqANDrelFreq)

    #C-组中值计算
    dfBins_median=df_bins.median(level=0)
    dfBins_median.rename(columns={column_name:'median'},inplace=True)
    #print(dfBins_median)

    #D-合并分割区间、频数计算和组中值的DataFrame格式数据。
    df_fre=dfBins_freqANDrelFreq.join(dfBins_median).sort_index().reset_index() #在合并时会自动匹配index
    #print(ranmen_fre)

    #E-计算频数比例
    df_fre['fre_percent%']=df_fre.apply(lambda row:row['fre']/df_fre.fre.sum()*100,axis=1)

    return df_fre


#convert points .shp to raster 将点数据写入为raster数据。使用raster.SetGeoTransform,栅格化数据。参考GDAL官方代码
def pts2raster(pts_shp,raster_path,cellSize,field_name=False):
    from osgeo import gdal, ogr,osr
    '''
    function - 将.shp格式的点数据转换为.tif栅格(raster)
    
    Paras:
    pts_shp - .shp格式点数据文件路径
    raster_path - 保存的栅格文件路径
    cellSize - 栅格单元大小
    field_name - 写入栅格的.shp点数据属性字段
    '''
    #定义空值（没有数据）的栅格数值 Define NoData value of new raster
    NoData_value=-9999
    
    #打开.shp点数据，并返回地理区域范围 Open the data source and read in the extent
    source_ds=ogr.Open(pts_shp)
    source_layer=source_ds.GetLayer()
    x_min, x_max, y_min, y_max=source_layer.GetExtent()
    
    #使用GDAL库建立栅格 Create the destination data source
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)
    target_ds=gdal.GetDriverByName('GTiff').Create(raster_path, x_res, y_res, 1, gdal.GDT_Float64) #gdal的数据类型 gdal.GDT_Float64,gdal.GDT_Int32...
    target_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    outband=target_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)

    #向栅格层中写入数据
    if field_name:
        gdal.RasterizeLayer(target_ds,[1], source_layer,options=["ATTRIBUTE={0}".format(field_name)])
    else:
        gdal.RasterizeLayer(target_ds,[1], source_layer,burn_values=[-1])   
        
    #配置投影坐标系统
    spatialRef=source_layer.GetSpatialRef()
    target_ds.SetProjection(spatialRef.ExportToWkt())       
        
    outband.FlushCache()
    return gdal.Open(raster_path).ReadAsArray()


def pts_geoDF2raster(pts_geoDF,raster_path,cellSize,scale):
    from osgeo import gdal,ogr,osr
    import numpy as np
    from scipy import stats
    '''
    function - 将GeoDaraFrame格式的点数据转换为栅格数据
    
    Paras:
    pts_geoDF - GeoDaraFrame格式的点数据
    raster_path - 保存的栅格文件路径
    cellSize - 栅格单元大小
    scale - 缩放核密度估计值
    '''
    #定义空值（没有数据）的栅格数值 Define NoData value of new raster
    NoData_value=-9999
    x_min, y_min,x_max, y_max=pts_geoDF.geometry.total_bounds

    #使用GDAL库建立栅格 Create the destination data source
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)
    target_ds=gdal.GetDriverByName('GTiff').Create(raster_path, x_res, y_res, 1, gdal.GDT_Float64 )
    target_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    outband=target_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)   
    
    #配置投影坐标系统
    spatialRef = osr.SpatialReference()
    epsg=int(pts_geoDF.crs.srs.split(":")[-1])
    spatialRef.ImportFromEPSG(epsg)  
    target_ds.SetProjection(spatialRef.ExportToWkt())
    
    #向栅格层中写入数据
    #print(x_res,y_res)
    X, Y = np.meshgrid(np.linspace(x_min,x_max,x_res), np.linspace(y_min,y_max,y_res))
    positions=np.vstack([X.ravel(), Y.ravel()])
    values=np.vstack([pts_geoDF.geometry.x, pts_geoDF.geometry.y])    
    print("Start calculating kde...")
    kernel=stats.gaussian_kde(values)
    Z=np.reshape(kernel(positions).T, X.shape)
    print("Finish calculating kde!")
    #print(values)
        
    outband.WriteArray(np.flip(Z,0)*scale)        
    outband.FlushCache()
    print("conversion complete!")
    return gdal.Open(raster_path).ReadAsArray()


def start_time():
    import datetime
    '''
    function-计算当前时间
    '''
    start_time=datetime.datetime.now()
    print("start time:",start_time)
    return start_time

def duration(start_time):
    import datetime
    '''
    function-计算持续时间

    Paras:
    start_time - 开始时间
    '''
    end_time=datetime.datetime.now()
    print("end time:",end_time)
    duration=(end_time-start_time).seconds/60
    print("Total time spend:%.2f minutes"%duration)
    
    
def is_outlier(data,threshold=3.5):
   import numpy as np
   '''
   function-判断异常值

   Params:
   data - 待分析的数据，列表或者一维数组
   threshold - 判断是否为异常值的边界条件    
   '''
   MAD=np.median(abs(data-np.median(data)))
   modified_ZScore=0.6745*(data-np.median(data))/MAD
   #print(modified_ZScore)
   is_outlier_bool=abs(modified_ZScore)>threshold    
   return is_outlier_bool,data[~is_outlier_bool]   


def print_html(df,row_numbers=5):
    from IPython.display import HTML
    '''
    function - 在Jupyter中打印DataFrame格式数据为HTML

    Paras:
    df - 需要打印的DataFrame或GeoDataFrame格式数据
    row_numbers - 打印的行数，如果为正，从开始打印如果为负，从末尾打印
     '''
    if row_numbers>0:
        return HTML(df.head(row_numbers).to_html())
    else:
        return HTML(df.tail(abs(row_numbers)).to_html())


def coefficient_of_determination(observed_vals,predicted_vals):
    import pandas as pd
    import numpy as np
    import math
    '''
    function - 回归方程的判定系数
    
    Paras:
    observed_vals - 观测值（实测值）
    predicted_vals - 预测值
    '''
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    #观测值的离差平方和(总平方和，或总的离差平方和)
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    #预测值的离差平方和
    pre_mean=vals_df.pre.mean()
    SS_reg=vals_df.pre.apply(lambda row:(row-pre_mean)**2).sum()
    #观测值和预测值的离差积和
    SS_obs_pre=vals_df.apply(lambda row:(row.obs-obs_mean)*(row.pre-pre_mean), axis=1).sum()
    
    #残差平方和
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
    
    #判断系数
    R_square_a=(SS_obs_pre/math.sqrt(SS_tot*SS_reg))**2
    R_square_b=1-SS_res/SS_tot
            
    return R_square_a,R_square_b


def vector_plot_3d(ax_3d,C,origin_vector,vector,color='r',label='vector',arrow_length_ratio=0.1):
    '''
    funciton - 转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式

    Paras:
    ax_3d - matplotlib的3d格式子图
    C - /coordinate_system - SymPy下定义的坐标系
    origin_vector - 如果是固定向量，给定向量的起点（使用向量，即表示从坐标原点所指向的位置），如果是自由向量，起点设置为坐标原点
    vector - 所要打印的向量
    color - 向量色彩
    label - 向量标签
    arrow_length_ratio - 向量箭头大小     
    '''
    origin_vector_matrix=origin_vector.to_matrix(C)
    x=origin_vector_matrix.row(0)[0]
    y=origin_vector_matrix.row(1)[0]
    z=origin_vector_matrix.row(2)[0]

    vector_matrix=vector.to_matrix(C)
    u=vector_matrix.row(0)[0]
    v=vector_matrix.row(1)[0]
    w=vector_matrix.row(2)[0]
    ax_3d.quiver(x,y,z,u,v,w,color=color,label=label,arrow_length_ratio=arrow_length_ratio)
    
    
def move_alongVectors(vector_list,coeffi_list,C,ax,):
    import random
    import sympy
    '''
    function - 给定向量，及对应系数，延向量绘制

    Paras:
    vector_list - 向量列表，按移动顺序
    coeffi_list - 向量的系数    
    C - SymPy下定义的坐标系
    ax - 子图
    '''
    colors=[color[0] for color in mcolors.TABLEAU_COLORS.items()]  #mcolors.BASE_COLORS, mcolors.TABLEAU_COLORS,mcolors.CSS4_COLORS
    colors__random_selection=random.sample(colors,len(vector_list)-1)
    v_accumulation=[]
    v_accumulation.append(vector_list[0])
    #每个向量绘制以之前所有向量之和为起点
    for expr in vector_list[1:]:
        v_accumulation.append(expr+v_accumulation[-1])

    v_accumulation=v_accumulation[:-1]   
    for i in range(1,len(vector_list)):
        vector_plot_3d(ax,C,v_accumulation[i-1].subs(coeffi_list),vector_list[i].subs(coeffi_list),color=colors__random_selection[i-1],label='v_%s'%coeffi_list[i-1][0],arrow_length_ratio=0.2)    
    
    
def LandsatMTL_info(fp):
    import os
    import re
    '''
    function - 读取landsat *_MTL.txt文件，提取需要的信息

    Paras:
    fp - Landsat 文件根目录

    return:
    band_fp_dic - 返回各个波段的路径字典
    Landsat_para - 返回Landsat 参数 
    '''
    fps=[os.path.join(root,file) for root, dirs, files in os.walk(fp) for file in files] #提取文件夹下所有文件的路径
    MTLPattern=re.compile(r'_MTL.txt',re.S) #匹配对象模式，提取_MTL.txt遥感影像的元数据文件
    MTLFn=[fn for fn in fps if re.findall(MTLPattern,fn)][0]
    with open(MTLFn,'r') as f: #读取所有元数据文件信息
        MTLText=f.read()
    bandFn_Pattern=re.compile(r'FILE_NAME_BAND_[0-9]\d* = "(.*?)"\n',re.S)  #Landsat 波段文件
    band_fn=re.findall(bandFn_Pattern,MTLText)
    band_fp=[[(re.findall(r'B[0-9]\d*',fn)[0], re.findall(r'.*?%s$'%fn,f)[0]) for f in fps if re.findall(r'.*?%s$'%fn,f)] for fn in band_fn] #(文件名，文件路径)
    band_fp_dic={i[0][0]:i[0][1] for i in band_fp}
    #需要数据的提取标签/根据需要读取元数据信息
    values_fields=["RADIANCE_ADD_BAND_10",
                   "RADIANCE_ADD_BAND_11",
                   "RADIANCE_MULT_BAND_10",
                   "RADIANCE_MULT_BAND_11",
                   "K1_CONSTANT_BAND_10",
                   "K2_CONSTANT_BAND_10",
                   "K1_CONSTANT_BAND_11",
                   "K2_CONSTANT_BAND_11",
                   "DATE_ACQUIRED",
                   "SCENE_CENTER_TIME",
                   "MAP_PROJECTION",
                   "DATUM",
                   "UTM_ZONE"]    
    Landsat_para={field:re.findall(re.compile(r'%s = "*(.*?)"*\n'%field),MTLText)[0] for field in values_fields} #（参数名，参数值）
    return  band_fp_dic,Landsat_para #返回所有波段路径和需要的参数值    

def NDVI(RED_band,NIR_band):
    import numpy as np
    '''
    function - 计算NDVI指数

    Paras:
    RED_band - 红色波段
    NIR_band - 近红外波段
    '''
    RED_band=np.ma.masked_where(NIR_band+RED_band==0,RED_band)
    NDVI=(NIR_band-RED_band)/(NIR_band+RED_band)
    NDVI=NDVI.filled(-9999)
    print("NDVI"+"_min:%f,max:%f"%(NDVI.min(),NDVI.max()))
    return NDVI

def las_2_DSM_Classification_raster_(las_fp,save_path):
    import pdal
    '''
    function - 转换单个.las点云数据为分类栅格数据，和DSM栅格数据
    
    Paras:
    las_fp - .las格式文件路径
    save_path - 保存路径列表，分类DSM存储与不同路径下
    
    '''
    
    #pipeline-用于建立分类栅格
    json_classification="""
        {
        "pipeline": [
                     "%s",
                     {
                     "filename":"%s",
                     "type":"writers.gdal",
                     "dimension":"Classification",
                     "data_type":"uint16_t",
                     "output_type":"mean",  
                     "resolution": 1
                     }               
        ]        
        }"""%(las_fp,save_path[0])
    
    #pipeline-用于建立DSM栅格数据
    json_DSM="""
        {
        "pipeline": [
                     "%s",
                     {
                     "filename":"%s",
                     "gdaldriver":"GTiff",
                     "type":"writers.gdal",
                     "output_type":"mean",  
                     "resolution": 1
                     }               
        ]        
        }"""%(las_fp,save_path[1])    
    
    json_combo=[json_classification,json_DSM] #配置选择性输出
    for json in json_combo:
        pipeline=pdal.Pipeline(json)
        pipeline.loglevel=8 #日志级别配置
        if pipeline.validate(): #检查json选项是否正确
            #print(pipeline.validate())
            try:
                count=pipeline.execute()
            except:
                print("\n An exception occurred,the file name:%s"%las_fp)
                print(pipeline.log) #如果出现错误，打印日志查看，以便修正错误代码
                
        else:
            print("pipeline unvalidate!!!")
            
def  las_info_extraction(las_fp,json_combo):
    import pdal
    '''
    function - 转换单个.las点云数据为分类栅格数据，和DSM栅格数据
    
    Paras:
    las_fp - .las格式文件路径
    save_path - 保存路径列表，分类DSM存储与不同路径下
    
    '''
    pipeline_list=[]
    if 'json_classification' in json_combo.keys():
        #pipeline-用于建立分类栅格
        json_classification="""
            {
            "pipeline": [
                         "%s",
                         {
                         "filename":"%s",
                         "type":"writers.gdal",
                         "dimension":"Classification",
                         "data_type":"uint16_t",
                         "output_type":"mean",  
                         "resolution": 1
                         }               
            ]        
            }"""%(las_fp,json_combo['json_classification'])
        pipeline_list.append(json_classification)
        
    elif 'json_DSM' in json_combo.keys():
        #pipeline-用于建立DSM栅格数据
        json_DSM="""
            {
            "pipeline": [
                         "%s",
                         {
                         "filename":"%s",
                         "gdaldriver":"GTiff",
                         "type":"writers.gdal",
                         "output_type":"mean",  
                         "resolution": 1
                         }               
            ]        
            }"""%(las_fp,json_combo['json_DSM']) 
        pipeline_list.append(json_DSM)
        
    elif 'json_ground' in json_combo.keys():
        #pipelin-用于提取ground地表
        json_ground="""
            {
            "pipeline": [
                         "%s",
                         {
                         "type":"filters.range",
                         "limits":"Classification[2:2]"                     
                         },
                         {
                         "filename":"%s",
                         "gdaldriver":"GTiff",
                         "type":"writers.gdal",
                         "output_type":"mean",  
                         "resolution": 1
                         }               
            ]        
            }"""%(las_fp,json_combo['json_ground'])   
        pipeline_list.append(json_ground)
    
    
    for json in pipeline_list:
        pipeline=pdal.Pipeline(json)
        pipeline.loglevel=8 #日志级别配置
        if pipeline.validate(): #检查json选项是否正确
            #print(pipeline.validate())
            try:
                count=pipeline.execute()
            except:
                print("\n An exception occurred,the file name:%s"%las_fp)
                print(pipeline.log) #如果出现错误，打印日志查看，以便修正错误代码
                
        else:
            print("pipeline unvalidate!!!")
    #print("finished conversion...")     

def las_info_extraction_combo(las_dirPath,json_combo_):
    import util,os,re
    from tqdm import tqdm
    '''
    function - 批量转换.las点云数据为DTM和分类栅格
    
    Paras:
    las_dirPath - LAS文件路径
    save_path - 保存路径
    
    return:
        
    '''
    file_type=['las']
    las_fn=util.filePath_extraction(las_dirPath,file_type)
    '''展平列表函数'''
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
    las_fn_list=flatten_lst([[os.path.join(k,las_fn[k][i]) for i in range(len(las_fn[k]))] for k in las_fn.keys()])
    pattern=re.compile(r'[_](.*?)[.]', re.S) 
    for i in tqdm(las_fn_list):  
        fn_num=re.findall(pattern, i.split("\\")[-1])[0] #提取文件名字符串中的数字
        #注意文件名路径中"\"和"/"，不同库支持的类型可能有所不同，需自行调整
        json_combo={key:os.path.join(json_combo_[key],"%s_%s.tif"%(os.path.split(json_combo_[key])[-1],fn_num)).replace("\\","/") for key in json_combo_.keys()}        
        util.las_info_extraction_combo(i.replace("\\","/"),json_combo)
    
       
def raster_mosaic(dir_path,out_fp,):
    import rasterio,glob,os
    from rasterio.merge import merge
    '''
    function - 合并多个栅格为一个
    
    Paras:
    dir_path - 栅格根目录
    out-fp - 保存路径
    
    return:
    out_trans - 返回变换信息
    '''
    
    #迁移rasterio提供的定义数组最小数据类型的函数
    def get_minimum_int_dtype(values):
        """
        Uses range checking to determine the minimum integer data type required
        to represent values.

        :param values: numpy array
        :return: named data type that can be later used to create a numpy dtype
        """

        min_value = values.min()
        max_value = values.max()

        if min_value >= 0:
            if max_value <= 255:
                return rasterio.uint8
            elif max_value <= 65535:
                return rasterio.uint16
            elif max_value <= 4294967295:
                return rasterio.uint32
        elif min_value >= -32768 and max_value <= 32767:
            return rasterio.int16
        elif min_value >= -2147483648 and max_value <= 2147483647:
            return rasterio.int32
    
    search_criteria = "*.tif" #搜寻所要合并的栅格.tif文件
    fp_pattern=os.path.join(dir_path, search_criteria)
    fps=glob.glob(fp_pattern) #使用glob库搜索指定模式的文件
    src_files_to_mosaic=[]
    for fp in fps:
        src=rasterio.open(fp)
        src_files_to_mosaic.append(src)    
    mosaic,out_trans=merge(src_files_to_mosaic)  #merge函数返回一个栅格数组，以及转换信息   
    
    #获得元数据
    out_meta=src.meta.copy()
    #更新元数据
    data_type=get_minimum_int_dtype(mosaic)
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     #通过压缩和配置存储类型，减小存储文件大小
                     "compress":'lzw',
                     "dtype":get_minimum_int_dtype(mosaic), 
                      }
                    )       
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic.astype(data_type))     
    
    return out_trans            

def las_classification_plotWithLegend(las_fp):  
    import rasterio as rio
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib import colors
    from matplotlib.patches import Rectangle
    '''
    function - 显示由.las文件生成的分类栅格文件，并显示图例
    
    Paras:
    las_fp - 分类文件路径
    '''    
    with rio.open(las_fp) as classi_src:
        classi_array=classi_src.read(1)

    las_classi_colorName={0:'black',1:'white',2:'beige',3:'palegreen',4:'lime',5:'green',6:'tomato',7:'silver',8:'grey',9:'lightskyblue',10:'purple',11:'slategray',12:'grey',13:'cadetblue',14:'lightsteelblue',15:'brown',16:'indianred',17:'darkkhaki',18:'azure',9999:'pink'}
    las_classi_colorRGB=pd.DataFrame({key:colors.hex2color(colors.cnames[las_classi_colorName[key]]) for key in las_classi_colorName.keys()})
    classi_array_color=[pd.DataFrame(classi_array).replace(las_classi_colorRGB.iloc[idx]).to_numpy() for idx in las_classi_colorRGB.index]
    classi_array_color_=np.concatenate([np.expand_dims(i,axis=-1) for i in classi_array_color],axis=-1)
    fig, ax=plt.subplots(figsize=(12, 12))
    im=ax.imshow(classi_array_color_, )
    ax.set_title(
        "LAS_classification",
        fontsize=14,
    )

    #增加图例
    color_legend=pd.DataFrame(las_classi_colorName.items(),columns=["id","color"])
    las_classi_name={0:'never classified',1:'unassigned',2:'ground',3:'low vegetation',4:'medium vegetation',5:'high vegetation',6:'building',7:'low point',8:'reserved',9:'water',10:'rail',11:'road surface',12:'reserved',13:'wire-guard(shield)',14:'wire-conductor(phase)',15:'transimission',16:'wire-structure connector(insulator)',17:'bridge deck',18:'high noise',9999:'null'}
    color_legend['label']=las_classi_name.values()
    classi_lengend=[Rectangle((0, 0), 1, 1, color=c) for c in color_legend['color']]

    ax.legend(classi_lengend,color_legend.label,mode='expand',ncol=3)
    plt.tight_layout()
    plt.show()
    
def las_classification_plotWithLegend_(classi_array):  
    import rasterio as rio
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib import colors
    from matplotlib.patches import Rectangle
    '''
    function - 显示由.las文件生成的分类栅格文件，并显示图例
    
    Paras:
    las_fp - 分类文件路径
    '''    
    las_classi_colorName={0:'black',1:'white',2:'beige',3:'palegreen',4:'lime',5:'green',6:'tomato',7:'silver',8:'grey',9:'lightskyblue',10:'purple',11:'slategray',12:'grey',13:'cadetblue',14:'lightsteelblue',15:'brown',16:'indianred',17:'darkkhaki',18:'azure',9999:'white'}
    las_classi_colorRGB=pd.DataFrame({key:colors.hex2color(colors.cnames[las_classi_colorName[key]]) for key in las_classi_colorName.keys()})
    classi_array_color=[pd.DataFrame(classi_array).replace(las_classi_colorRGB.iloc[idx]).to_numpy() for idx in las_classi_colorRGB.index]
    classi_array_color_=np.concatenate([np.expand_dims(i,axis=-1) for i in classi_array_color],axis=-1)
    fig, ax=plt.subplots(figsize=(12, 12))
    im=ax.imshow(classi_array_color_, )
    ax.set_title(
        "LAS_classification",
        fontsize=14,
    )

    #增加图例
    color_legend=pd.DataFrame(las_classi_colorName.items(),columns=["id","color"])
    las_classi_name={0:'never classified',1:'unassigned',2:'ground',3:'low vegetation',4:'medium vegetation',5:'high vegetation',6:'building',7:'low point',8:'reserved',9:'water',10:'rail',11:'road surface',12:'reserved',13:'wire-guard(shield)',14:'wire-conductor(phase)',15:'transimission',16:'wire-structure connector(insulator)',17:'bridge deck',18:'high noise',9999:'null'}
    color_legend['label']=las_classi_name.values()
    classi_lengend=[Rectangle((0, 0), 1, 1, color=c) for c in color_legend['color']]

    ax.legend(classi_lengend,color_legend.label,mode='expand',ncol=3)
    plt.tight_layout()
    plt.show()
    
def raster_reprojection(raster_fp,dst_crs,save_path):
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import rasterio as rio
    '''
    function - 转换栅格投影
    
    Paras:
    raster_fp - 待转换投影的栅格
    dst_crs - 目标投影
    save_path - 保存路径
    '''
    with rio.open(raster_fp) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rio.open(save_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)      
    print("finished reprojecting...")    
    
def get_crs_raster(raster_fp):
    import rasterio as rio
    '''
    function - 获取给定栅格的投影坐标-crs.
    
    Paras:
    raster)fp - 给定栅格文件的路径
    '''
    with rio.open(raster_fp) as raster_crs:
        raster_profile=raster_crs.profile
        return raster_profile['crs']    
    
def animated_gif_show(gif_fp,figsize=(8,8)):
    from PIL import Image, ImageSequence
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML
    '''
    function - 读入.gif，并动态显示
    
    Paras:
    gif_fp - GIF文件路径
    figsize - 图表大小
    '''
    gif=Image.open(gif_fp,'r')
    frames=[np.array(frame.getdata(),dtype=np.uint8).reshape(gif.size[0],gif.size[1]) for frame in ImageSequence.Iterator(gif)] #dtype=np.uint8

    fig=plt.figure(figsize=figsize)
    imgs=[(plt.imshow(img,animated=True),) for img in frames]

    anim=animation.ArtistAnimation(fig, imgs, interval=300,repeat_delay=3000, blit=True)
    return HTML(anim.to_html5_video())

def demo_con_style(a_coordi,b_coordi,ax,connectionstyle):
    '''
    function - 在matplotlib的子图中绘制连接线
    reference - matplotlib官网Connectionstyle Demo

    Paras:
    a_coordi - a点的x，y坐标
    b_coordi - b点的x，y坐标
    ax - 子图
    connectionstyle - 连接线的形式
    '''
    x1, y1=a_coordi[0],a_coordi[1]
    x2, y2=b_coordi[0],b_coordi[1]

    ax.plot([x1, x2], [y1, y2], ".")
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle=connectionstyle,
                                ),
                )

    ax.text(.05, .95, connectionstyle.replace(",", ",\n"),
            transform=ax.transAxes, ha="left", va="top")

'''展平列表函数'''
flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst] 


#from data_generator import DataGenerator
#from knee_locator import KneeLocator

def kneed_lineGraph(x,y):
    import matplotlib.pyplot as plt
    '''
    function - 绘制折线图，及其拐点。需调用kneed库的KneeLocator，及DataGenerator文件

    Paras:
    x - 横坐标，用于横轴标签
    y - 纵坐标，用于计算拐点    
    '''
    #如果调整图表样式，需调整knee_locator文件中的plot_knee（）函数相关参数
    kneedle=KneeLocator(x, y, curve='convex', direction='decreasing')
    print('曲线拐点（凸）：',round(kneedle.knee, 3))
    print('曲线拐点（凹）：',round(kneedle.elbow, 3))
    kneedle.plot_knee(figsize=(8,8))
    
    
class feature_builder_BOW:
    '''
    class - 根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计
    '''   
    def __init__(self,num_cluster=32):
        self.num_clusters=num_cluster

    def extract_features(self,img):
        import cv2 as cv
        '''
        function - 提取图像特征
        
        Paras:
        img - 读取的图像
        '''
        star_detector=cv.xfeatures2d.StarDetector_create()
        key_points=star_detector.detect(img)
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        kp,des=cv.xfeatures2d.SIFT_create().compute(img_gray, key_points) #SIFT特征提取器提取特征
        return des
    
    def visual_BOW(self,des_all):
        from sklearn.cluster import KMeans
        '''
        function - 聚类所有图像的特征（描述子/SIFT），建立视觉词袋
        
        des_all - 所有图像的关键点描述子
        '''
        print("start KMean...")
        kmeans=KMeans(self.num_clusters)
        kmeans=kmeans.fit(des_all)
        #centroids=kmeans.cluster_centers_
        print("end KMean...")
        return kmeans         
    
    def get_visual_BOW(self,training_data):
        import cv2 as cv
        from tqdm import tqdm
        '''
        function - 提取图像特征，返回所有图像关键点聚类视觉词袋
        
        Paras:
        training_data - 训练数据集
        '''
        des_all=[]
        #i=0        
        for item in tqdm(training_data):

            
            classi_judge=item['object_class']
            img=cv.imread(item['image_path'])
            des=self.extract_features(img)
            des_all.extend(des)           
            #print(des.shape)

            #if i==10:break
            #i+=1        
        kmeans=self.visual_BOW(des_all)      
        return kmeans
    
    def normalize(self,input_data):
        import numpy as np
        '''
        fuction - 归一化数据
        
        input_data - 待归一化的数组
        '''
        sum_input=np.sum(input_data)
        if sum_input>0:
            return input_data/sum_input #单一数值/总体数值之和，最终数值范围[0,1]
        else:
            return input_data               
    
    def construct_feature(self,img,kmeans):
        import numpy as np
        '''
        function - 使用聚类的视觉词袋构建图像特征（构造码本）
        
        Paras:
        img - 读取的单张图像
        kmeans - 已训练的聚类模型
        '''
        des=self.extract_features(img)
        labels=kmeans.predict(des.astype(np.float)) #对特征执行聚类预测类标
        feature_vector=np.zeros(self.num_clusters)
        for i,item in enumerate(feature_vector): #计算特征聚类出现的频数/直方图
            feature_vector[labels[i]]+=1
        feature_vector_=np.reshape(feature_vector,((1,feature_vector.shape[0])))
        return self.normalize(feature_vector_)
    
    def get_feature_map(self,training_data,kmeans):
        import cv2 as cv
        '''
        function - 返回每个图像的特征映射（码本映射）
        Paras:
        training_data - 训练数据集
        kmeans - 已训练的聚类模型
        '''
        feature_map=[]
        for item in training_data:
            temp_dict={}
            temp_dict['object_class']=item['object_class']
            #print("Extracting feature for",item['image_path'])
            img=cv.imread(item['image_path'])
            temp_dict['feature_vector']=self.construct_feature(img,kmeans)
            if temp_dict['feature_vector'] is not None:
                feature_map.append(temp_dict)
        #print(feature_map[0]['feature_vector'].shape,feature_map[0])
        return feature_map    
    
from pathlib import Path

class DisplayablePath(object):
    '''
    class - 返回指定路径下所有文件夹及其下文件的结构。代码未改动，迁移于'https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python'
    '''

    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))    

"""
# Code adapted from:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/collections.py

Source License
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""

class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]    
    
def imgs_layoutShow(imgs_root,imgsFn_lst,columns,scale,figsize=(15,10)):
    import math,os
    import matplotlib.pyplot as plt
    from PIL import Image
    '''
    function - 显示一个文件夹下所有图片，便于查看。

    Paras:
    imgs_root - 图像所在根目录
    imgsFn_lst - 图像名列表
    columns - 列数
    '''
    rows=math.ceil(len(imgsFn_lst)/columns)
    fig,axes=plt.subplots(rows,columns,sharex=True,sharey=True,figsize=figsize)   #布局多个子图，每个子图显示一幅图像
    ax=axes.flatten()  #降至1维，便于循环操作子图
    for i in range(len(imgsFn_lst)):
        img_path=os.path.join(imgs_root,imgsFn_lst[i]) #获取图像的路径
        img_array=Image.open(img_path) #读取图像为数组，值为RGB格式0-255        
        img_resize=img_array.resize([int(scale * s) for s in img_array.size] ) #传入图像的数组，调整图片大小
        ax[i].imshow(img_resize)  #显示图像
        ax[i].set_title(i+1)
    fig.tight_layout() #自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("images show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()

    
#将每批次样本X的形状转换为(batch_size,-1)
#from torch import nn
# class flattenLayer(nn.Module):
#     def __init__(self):
#         super(flattenLayer,self).__init__()
#     def forward(self,x):
#         return x.view(x.shape[0],-1)

def evaluate_accuracy(data_iter, net):
    '''
    funtion - 平均模型net在数据集data_iter上的准确率
    '''
    accu_sum,n=0.0,0
    for X,y in data_iter:
        accu_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
    return accu_sum/n

def sgd_v1(params,lr):
    '''
    funtion - 梯度下降，v1版
    '''
    for param in params:
        param.data-=lr*param.grad
        
def train_v1(net,train_iter, test_iter, loss, num_epochs,params=None, lr=None, optimizer=None,interval_print=100):
    from tqdm.auto import tqdm
    '''
    function - 训练模型，v1版
    
    Paras:
    net - 构建的模型结构
    train_iter - 可迭代训练数据集
    test_iter - 可迭代测试数据集
    loss - 损失函数
    num_epochs - 训练迭代次数
    params=None - 初始化模型参数，以列表形式表述，例如[W,b]
    lr=None, - 学习率
    optimizer=None - 优化函数
    intervation - 训练模型，v1版
    '''
    for epoch in tqdm(range(num_epochs)):
        train_l_sum, train_acc_sum, n=0.0, 0.0, 0
        for X, y in train_iter:
            y_pred=net(X)
            l=loss(y_pred,y).sum()
            
            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd_v1(params,lr) #应用自定义的梯度下降法
            else:
                optimizer.step() #应用torch.optim.SGD库的梯度下降法
                
            train_l_sum += l.item()
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        if test_iter is not None:        
            test_acc = evaluate_accuracy(test_iter, net)
        if epoch%interval_print==0:
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'%(epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))       
            
            
def KITTI_info(KITTI_info_fp,timestamps_fp):
    import util
    '''
    function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件，2-包含时间戳的.txt文件
    '''

    import pandas as pd
    import util
    import os

    drive_fp=util.filePath_extraction(KITTI_info_fp,['txt'])
    '''展平列表函数'''
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
    drive_fp_list=flatten_lst([[os.path.join(k,f) for f in drive_fp[k]] for k,v in drive_fp.items()])

    columns=["lat","lon","alt","roll","pitch","yaw","vn","ve","vf","vl","vu","ax","ay","ay","af","al","au","wx","wy","wz","wf","wl","wu","pos_accuracy","vel_accuracy","navstat","numsats","posmode","velmode","orimode"]
    drive_info=pd.concat([pd.read_csv(item,delimiter=' ',header=None) for item in drive_fp_list],axis=0)
    drive_info.columns=columns
    drive_info=drive_info.reset_index()

    timestamps=pd.read_csv(timestamps_fp,header=None)
    timestamps.columns=['timestamps_']
    drive_info=pd.concat([drive_info,timestamps],axis=1,sort=False)
    #drive_29_0071_info.index=pd.to_datetime(drive_29_0071_info["timestamps_"]) #用时间戳作为行(row)索引
    return drive_info            

class movingAverage_inflection:
    import pandas as pd

    '''
    class - 曲线（数据）平滑，与寻找曲线水平和纵向的斜率变化点

    Paras:
    series - pandas 的Series格式数据
    window - 滑动窗口大小，值越大，平滑程度越大
    plot_intervals - 是否打印置信区间，某人为False 
    scale - 偏差比例，默认为1.96, 
    plot_anomalies - 是否打印异常值，默认为False,
    figsize - 打印窗口大小，默认为(15,5),
    threshold - 拐点阈值，默认为0
    '''
    def __init__(self,series, window, plot_intervals=False, scale=1.96, plot_anomalies=False,figsize=(15,5),threshold=0):
        self.series=series
        self.window=window
        self.plot_intervals=plot_intervals
        self.scale=scale
        self.plot_anomalies=plot_anomalies
        self.figsize=figsize

        self.threshold=threshold
        self.rolling_mean=self.movingAverage()

    def masks(self,vec):
        import numpy as np
        '''
        function - 寻找曲线水平和纵向的斜率变化，参考 https://stackoverflow.com/questions/47342447/find-locations-on-a-curve-where-the-slope-changes
        '''
        d=np.diff(vec)
        dd=np.diff(d)

        # Mask of locations where graph goes to vertical or horizontal, depending on vec
        to_mask=((d[:-1] != self.threshold) & (d[:-1] == -dd-self.threshold))
        # Mask of locations where graph comes from vertical or horizontal, depending on vec
        from_mask=((d[1:] != self.threshold) & (d[1:] == dd-self.threshold))
        return to_mask, from_mask

    def apply_mask(self,mask, x, y):
        return x[1:-1][mask], y[1:-1][mask]   

    def knee_elbow(self):
        import numpy as np
        '''
        function - 返回拐点的起末位置
        '''        
        x_r=np.array(self.rolling_mean.index)
        y_r=np.array(self.rolling_mean)
        to_vert_mask, from_vert_mask=self.masks(x_r)
        to_horiz_mask, from_horiz_mask=self.masks(y_r)     

        to_vert_t, to_vert_v=self.apply_mask(to_vert_mask, x_r, y_r)
        from_vert_t, from_vert_v=self.apply_mask(from_vert_mask, x_r, y_r)
        to_horiz_t, to_horiz_v=self.apply_mask(to_horiz_mask, x_r, y_r)
        from_horiz_t, from_horiz_v=self.apply_mask(from_horiz_mask, x_r, y_r)    
        return x_r,y_r,to_vert_t, to_vert_v,from_vert_t, from_vert_v,to_horiz_t, to_horiz_v,from_horiz_t, from_horiz_v

    def movingAverage(self):
        rolling_mean=self.series.rolling(window=self.window).mean()        
        return rolling_mean        

    def plot_movingAverage(self,inflection=False):
        import numpy as np
        from sklearn.metrics import median_absolute_error, mean_absolute_error
        import matplotlib.pyplot as plt
        """
        function - 打印移动平衡/滑动窗口，及拐点
        """

        plt.figure(figsize=self.figsize)
        plt.title("Moving average\n window size = {}".format(self.window))
        plt.plot(self.rolling_mean, "g", label="Rolling mean trend")

        #打印置信区间，Plot confidence intervals for smoothed values
        if self.plot_intervals:
            mae=mean_absolute_error(self.series[self.window:], self.rolling_mean[self.window:])
            deviation=np.std(self.series[self.window:] - self.rolling_mean[self.window:])
            lower_bond=self.rolling_mean - (mae + self.scale * deviation)
            upper_bond=self.rolling_mean + (mae + self.scale * deviation)
            plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
            plt.plot(lower_bond, "r--")

            # 显示异常值，Having the intervals, find abnormal values
            if self.plot_anomalies:
                anomalies=pd.DataFrame(index=self.series.index, columns=self.series.to_frame().columns)
                anomalies[self.series<lower_bond]=self.series[self.series<lower_bond].to_frame()
                anomalies[self.series>upper_bond]=self.series[self.series>upper_bond].to_frame()
                plt.plot(anomalies, "ro", markersize=10)

        if inflection:
            x_r,y_r,to_vert_t, to_vert_v,from_vert_t, from_vert_v,to_horiz_t, to_horiz_v,from_horiz_t, from_horiz_v=self.knee_elbow()
            plt.plot(x_r, y_r, 'b-')
            plt.plot(to_vert_t, to_vert_v, 'r^', label='Plot goes vertical')
            plt.plot(from_vert_t, from_vert_v, 'kv', label='Plot stops being vertical')
            plt.plot(to_horiz_t, to_horiz_v, 'r>', label='Plot goes horizontal')
            plt.plot(from_horiz_t, from_horiz_v, 'k<', label='Plot stops being horizontal')     


        plt.plot(self.series[self.window:], label="Actual values")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.xticks(rotation='vertical')
        plt.show()

def vanishing_position_length(matches_num,coordi_df,epsg,**kwargs):
    from shapely.geometry import Point, LineString, shape
    import geopandas as gpd
    import pyproj
    import numpy as np
    import pandas as pd
    '''
    function - 计算图像匹配特征点几乎无关联的距离，即对特定位置视觉随距离远去而感知消失的距离

    Paras:
    matches_num - 由类dynamicStreetView_visualPerception计算的特征关键点匹配数量
    coordi_df - 包含经纬度的DataFrame，其列名为：lon,lat
    **kwargs - 同类movingAverage_inflection配置参数
    '''
    MAI_paras={'window':15,'plot_intervals':True,'scale':1.96, 'plot_anomalies':True,'figsize':(15*2,5*2),'threshold':0}
    MAI_paras.update(kwargs)   
    #print( MAI_paras)

    vanishing_position={}
    for idx in range(len(matches_num)): 
        x=np.array(range(idx,idx+len(matches_num[idx]))) 
        y=np.array(matches_num[idx])
        y_=pd.Series(y,index=x)   
        MAI=movingAverage_inflection(y_, window=MAI_paras['window'],plot_intervals=MAI_paras['plot_intervals'],scale=MAI_paras['scale'], plot_anomalies=MAI_paras['plot_anomalies'],figsize=MAI_paras['figsize'],threshold=MAI_paras['threshold'])   
        _,_,_,_,from_vert_t, _,_, _,from_horiz_t,_=MAI.knee_elbow()
        if np.any(from_horiz_t!= None) :
            vanishing_position[idx]=(idx,from_horiz_t[0])
        else:
            vanishing_position[idx]=(idx,idx)
    vanishing_position_df=pd.DataFrame.from_dict(vanishing_position,orient='index',columns=['start_idx','end_idx'])

    vanishing_position_df['geometry']=vanishing_position_df.apply(lambda idx:LineString(coordi_df[idx.start_idx:idx.end_idx][['lon','lat']].apply(lambda row:Point(row.lon,row.lat), axis=1).values.tolist()), axis=1)
    crs_4326={'init': 'epsg:4326'}
    vanishing_position_gdf=gpd.GeoDataFrame(vanishing_position_df,geometry='geometry',crs=crs_4326)

    crs_=pyproj.CRS(epsg) 
    vanishing_position_gdf_reproj=vanishing_position_gdf.to_crs(crs_)
    vanishing_position_gdf_reproj['length']=vanishing_position_gdf_reproj.geometry.length
    return vanishing_position_gdf_reproj

class dynamicStreetView_visualPerception:
    '''
    class - 应用Star提取图像关键点，结合SIFT获得描述子，根据特征匹配分析特征变化（视觉感知变化），即动态街景视觉感知

    Paras:
    imgs_fp - 图像路径列表
    knnMatch_ratio - 图像匹配比例，默认为0.75
    '''
    def __init__(self,imgs_fp,knnMatch_ratio=0.75):
        self.knnMatch_ratio=knnMatch_ratio
        self.imgs_fp=imgs_fp

    def kp_descriptor(self,img_fp):
        import cv2 as cv
        '''
        function - 提取关键点和获取描述子
        '''
        img=cv.imread(img_fp)
        star_detector=cv.xfeatures2d.StarDetector_create()        
        key_points=star_detector.detect(img) #应用处理Star特征检测相关函数，返回检测出的特征关键点
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #将图像转为灰度
        kp,des=cv.xfeatures2d.SIFT_create().compute(img_gray, key_points) #SIFT特征提取器提取特征
        return kp,des


    def feature_matching(self,des_1,des_2,kp_1=None,kp_2=None):
        import cv2 as cv
        '''
        function - 图像匹配
        '''
        bf=cv.BFMatcher()
        matches=bf.knnMatch(des_1,des_2,k=2)

        '''
        可以由匹配matches返回关键点（train,query）的位置索引，train图像的索引，及描述子之间的距离
        DMatch.distance - Distance between descriptors. The lower, the better it is.
        DMatch.trainIdx - Index of the descriptor in train descriptors
        DMatch.queryIdx - Index of the descriptor in query descriptors
        DMatch.imgIdx - Index of the train image.
        '''
        '''
        if kp_1 !=None and kp_2 != None:
            kp1_list=[kp_1[mat[0].queryIdx].pt for mat in matches]
            kp2_list=[kp_2[mat[0].trainIdx].pt for mat in matches]
            des_distance=[(mat[0].distance,mat[1].distance) for mat in matches]
            print(des_distance[:5])
        '''

        good=[]
        for m,n in matches:
            if m.distance < self.knnMatch_ratio*n.distance:
                good.append(m) 
        #good_num=len(good)
        return good #,good_num


    def sequence_statistics(self):
        from tqdm import tqdm
        '''
        function - 序列图像匹配计算，每一位置图像与后续所有位置匹配分析
        '''        
        des_list=[]
        print("计算关键点和描述子...")
        for f in tqdm(self.imgs_fp):        
            _,des=self.kp_descriptor(f)
            des_list.append(des)
        matches_sequence={}
        print("计算序列图像匹配数...")
        for i in tqdm(range(len(des_list)-1)):
            matches_temp=[]
            for j_des in des_list[i:]:
                matches_temp.append(self.feature_matching(des_list[i],j_des))
            matches_sequence[i]=matches_temp
        matches_num={k:[len(v) for v in val] for k,val in matches_sequence.items()}
        return matches_num  
    
def raster_clip(raster_fp,clip_boundary_fp,save_path):
   import earthpy.spatial as es
   import geopandas as gpd
   from pyproj import CRS
   import rasterio as rio
   '''
   function - 给定裁切边界，批量裁切栅格数据

   Paras:
   raster_fp - 待裁切的栅格数据文件路径（.tif）,具有相同的坐标投影系统
   clip_boundary - 用于裁切的边界（.shp，WGS84，无投影）

   return:
   rasterClipped_pathList - 裁切后的文件路径列表
   '''
   clip_bound=gpd.read_file(clip_boundary_fp)
   with rio.open(raster_fp[0]) as raster_crs:
       raster_profile=raster_crs.profile
       clip_bound_proj=clip_bound.to_crs(raster_profile["crs"])

   rasterClipped_pathList=es.crop_all(raster_fp, save_path, clip_bound_proj, overwrite=True) #对所有波段band执行裁切
   print("finished clipping.")
   return rasterClipped_pathList   
    