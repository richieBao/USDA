> Created on Mon Jan 25 09\22\04 2021 @author: Richie Bao-caDesign设计(cadesign.cn)

## 1. Sentinel-2，(聚类)土地分类，VGG16，SegNet遥感影像语义分割/解译，超像素级分割(superpixels-segmentation)，高空分辨率特征尺度界定，及尺度的深度流动线/特征区域的延申
机器学习和深度学习的发展更新，改变，或颠覆了很多传统的方法，这发生在不计其数的领域中。在地理信息系统中，遥感影像（remote sensing image, RS）,航拍图像(aerial image),点云数据（Point-Cloud），数字高程(digital elevation maps,DEM)，地图（例如，google map，OSM（open street map），百度地图等）等都成为了机器/深度学习的数据集。其中对遥感影像解读的发展尤为深入，这涉及到土地分类(Land classification)，RS语义分割（Semantic segmenticion），变化监测（Change detection），影像配准(Image registration)，对象（目标）检测（Object detection），云检测（Cloud detection），财富和经济活力测量（Wealth and economic activity measurement），超分辨率(Super resolution)，超像素级分割(superpixels-segmentation)等诸多内容研究方法的革新。在[Satellite Image Deep Learning-Resources for deep learning with satellite & aerial imagery](https://awesomeopensource.com/project/robmarkcole/satellite-image-deep-learning)一文中，作者对当前深度学习在遥感影像数据集，相关技术项目，研究方法，开发工具，开源软件和代码上给出了很详细的汇总。这对风景园林和城乡规划专业在应用机器学习和深度学习解读城市，自然，甚至微观世界都产生不小的影响。

本文引入应用机器/深度学习实现影像解译、土地分类、地物探测，超级像素分割等代码，并由此发展高空分辨率特征尺度界定，及尺度的深度流动线/特征区域的延申等概念。


### 1.1 [Sentinel-2 遥感影像](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)
sentinel-2为高分辨率多光谱成像卫星，为2A和2B两颗卫星，分别于2015-06-23和2017-03-07日发射升空。每颗卫星重访周期为10天，两者则为每5天完成一次对地球赤道地区的完整成像。卫星寿命为7.25年，其携带的多光谱器( MultiSpectral Instrument，MSI)，覆盖13个光谱波段，地面分辨率分别为10m、20m和60m。数据可以从[Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home)处下载。其具体的波段解释与Landsat-8比较如下：

| Landset-8  |                |               | Sentinel-2      |                |               |
|------------|----------------|---------------|-----------------|----------------|---------------|
| Band       | Wavelenght(nm) | Resolution(m) | Band            | Wavelength(nm) | Resolution(m) |
| 1(Coastal) | 430-450        | 30            | 1(Coastal)      | 433-453        | 60            |
| 2(Blue)    | 450-515        | 30            | 2(Blue)         | 458-523        | 10            |
| 3(Green)   | 525-600        | 30            | 3(Green)        | 543-578        | 10            |
| 4(Red)     | 630-680        | 30            | 4(Red)          | 650-680        | 10            |
|            |                |               | 5(red Edge)     | 698-713        | 20            |
|            |                |               | 6(red Edge)     | 733-748        | 20            |
|            |                |               | 7(red Edge)     | 773-793        | 20            |
| 5(NIR)     | 845-885        | 30            | 8(NIR)          | 785-900        | 10            |
|            |                |               | 9(Water vapor)  | 935-955        | 60            |
|            |                |               | 10(SWIR-Cirrus) | 1360-1390      | 60            |
| 6(SWIR-1)  | 1560-1660      | 30            | 11(SWIR-1)      | 1565-1655      | 20            |
| 7(SWIR-2)  | 2100-2300      | 30            | 12(SWIR-2)      | 2100-2280      | 20            |
| 8(PAN)     | 503-676        | 15            |                 |                |               |


Sentinel-2与Landset-8最大的区别除了各个波段的分辨率不同外，还有在近红外波段NIR与红色波段之间细分了Red Edge红边波段，这对检测植被健康信息非常有效。

Sentinel-2产品级别可以划分为，Level-0:原始数据；Level-1A:包含元信息的几何粗矫正产品；Level-1B:辐射率产品，嵌入经GCP优化的几何模型，但未进行相应的几何校正；Level-1C:经正射校正和亚像元级几何精校正后的大气表观反射率产品；Level-2A:由Level-1C产品经过大气校正的大气底层反射率数据(Bottom Of Atmosphere (BOA) reflectance images derived from the associated Level-1C products)。在由Level-1C生成Level-2A（即经辐射定标和大气校正），可以使用European Space Agency,ESA 欧空局发布的[Sen2Cor](http://step.esa.int/main/snap-supported-plugins/sen2cor/)工具，在Windows下的Command Prompt,CMD终端下安装（执行`L2A_Process --help`命令），并执行`L2A_Process +数据位置+参数（可选）`。ESA发布的产品中混合有标识为'_MSIL1C_'的Level-1C产品，标识为'_MSIL2A_'的Level-2A产品，需要注意区分，Sen2Cor工具只对Level-1C产品执行大气校正产生Level-2A产品。

* [rio_tiler库](https://cogeotiff.github.io/rio-tiler/) 是[rasterio](https://rasterio.readthedocs.io/en/latest/)的插件(plugin)，用于从栅格数据集读取网页地图瓦片(web map tiles)。目前支持读取Sentinel 2, Sentinel 1, Landsat 8, CBERS等遥感影像数据，以及基本处理。注意，目前最新的版本为2.0.0，但conda安装支持的版本为1.4.0。2.0.0的版本作者提供了更多的影像处理功能。rio_tiler同时支持读取本地的Sentinel-2影像信息，使用该库中的main方法。

#### 1.1.1 以Web Mercator方式显示Sentinel-2的一个波段


```python
import rio_tiler
help(rio_tiler)
```

    Help on package rio_tiler:
    
    NAME
        rio_tiler - rio-tiler.
    
    PACKAGE CONTENTS
        cbers
        cmap (package)
        errors
        landsat8
        main
        mercator
        profiles
        sentinel1
        sentinel2
        utils
    
    DATA
        version = '1.4.0'
    
    FILE
        c:\users\richi\anaconda3\envs\rs\lib\site-packages\rio_tiler\__init__.py
    
    
    


```python
# 代码迁移于 https://github.com/binder-examples/getting-data/blob/master/Sentinel2.ipynb; updated by richie bao on Fri Jan 29 18:52:53 2021
%matplotlib inline
import matplotlib.pyplot as plt
import math,os
import numpy as np
from rio_tiler import main
from skimage import exposure
```

Web墨卡托投影(Web Mercator)是墨卡托投影的一种变体，是Web地图应用的事实标准。自2005年Google地图采用该投影之后，几乎所有的在线地图提供商都使用这一标准，包括[Google map](https://www.google.com/maps/@41.8305755,-87.6609536,14z), [Mapbox](https://www.mapbox.com/)，[Bing map](https://www.bing.com/maps), [OpenStreetMap](https://www.openstreetmap.org/#map=4/38.01/-95.84),[MapQuest](https://www.mapquest.com/),[Esri](https://www.esri.com/en-us/home)等。其正式的EPSG标识符是EPSG:3857。

几个世纪以来，人们一直在使用坐标系统和地图投影将地球的形状转换成可用的平面地图。而世界地图很大，不能直接在电脑上显示，所以引出快速浏览和缩放地图的机制，地图瓦片(map tiles)。将世界划分为很多小方块，每个小方块都有固定的地理面积和规模。这样可以在不加载整个地图的情况下浏览其中的一小部分。这涉及到几种表示方法，大地坐标，投影系统，像素坐标和金子塔瓦片，及它们之间的相互转换。

1. 度——Degrees Geodetic coordinates WGS84 (EPSG:4326)：使用1984年定义的世界大地测量系统(World Geodetic System)，GPS设备用于定义地球位置的经纬度坐标。

2. 米——Meters Projected coordinates Spherical Mercator (EPSG:3857)：全球投影坐标(Global projected coordinates )，用于GIS，A Web Map Tile Service (WM(T)S)服务的栅格瓦爿(raster tile)生成。

3. 像素——Pixels Screen coordinates XY pixels at zoom：影像金子塔每一层(each level of the pyramid)的特定缩放像素坐标。顶级(zoom=0)通常有$256 \times 256$像素，下一级为$512 \times 512$等。带有屏幕的设备（电脑，手机）等在定义的缩放级别计算像素坐标，并确定应该从服务器加载的区域用于可视屏幕。

4. 瓦片——Tiles Tile coordinates Tile Map Service (ZXY)：影像金子塔中指定缩放级别下(zoom level)瓦爿的索引，即x轴和y轴的位置/索引。每一级别下所有瓦片都有相同的尺寸，通常为$256 \times 256$像素。就是由粗到细不同分辨率的影像集合。其底部为图像的高分辨率表示，为原始图像，瓦片数应与原始图像的大小同；顶部为低分辨率的近似影像，最顶层只有1个瓦片，而后为4，16等。

**球面墨卡托投影金字塔的分辨率和比例列表**

| Zoom level | Resolution (meters / pixel) | Map Scale (at 96 dpi) | Width and Height of map (pixels) |
|------------|-----------------------------|-----------------------|----------------------------------|
| 0          | 156,543.03                  | 1 : 591,658,710.90    | 256                              |
| 1          | 78,271.52                   | 1 : 295,829,355.45    | 512                              |
| 2          | 39,135.76                   | 1 : 147,914,677.73    | 1,024                            |
| 3          | 19,567.88                   | 1 : 73,957,338.86     | 2,048                            |
| 4          | 9,783.94                    | 1 : 36,978,669.43     | 4,096                            |
| 5          | 4,891.97                    | 1 : 18,489,334.72     | 8,192                            |
| 6          | 2,445.98                    | 1 : 9,244,667.36      | 16,384                           |
| 7          | 1,222.99                    | 1 : 4,622,333.68      | 32,768                           |
| 8          | 611.4962263                 | 1 : 2,311,166.84      | 65,536                           |
| 9          | 305.7481131                 | 1 : 1,155,583.42      | 131,072                          |
| 10         | 152.8740566                 | 1 : 577,791.71        | 262,144                          |
| 11         | 76.43702829                 | 1 : 288,895.85        | 524,288                          |
| 12         | 38.21851414                 | 1 : 144,447.93        | 1,048,576                        |
| 13         | 19.10925707                 | 1 : 72,223.96         | 2,097,152                        |
| 14         | 9.554728536                 | 1 : 36,111.98         | 4,194,304                        |
| 15         | 4.777314268                 | 1 : 18,055.99         | 8,388,608                        |
| 16         | 2.388657133                 | 1 : 9,028.00          | 16,777,216                       |
| 17         | 1.194328566                 | 1 : 4,514.00          | 33,554,432                       |
| 18         | 0.597164263                 | 1 : 2,257.00          | 67,108,864                       |
| 19         | 0.298582142                 | 1 : 1,128.50          | 134,217,728                      |
| 20         | 0.149291071                 | 10:24.2               | 268,435,456                      |
| 21         | 0.074645535                 | 05:42.1               | 536,870,912                      |
| 22         | 0.037322768                 | 03:21.1               | 1,073,741,824                    |
| 23         | 0.018661384                 | 02:10.5               | 2,147,483,648                    |

对于金子塔瓦片和坐标之间的转换可以查看[Tiles à la Google Maps](https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/)，其中给出了转换的源代码。

> 参考文献：
1. [Tiles à la Google Maps](https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/)


```python
def deg2num(lat_deg, lon_deg, zoom):
    import math
    '''
    code migrated
    function - 将经纬度坐标转换为指定zoom level缩放级别下，金子塔中瓦片的坐标。
    '''
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def centroid(bounds):
    '''
    code migrated
    function - 根据获取的地图边界坐标[左下角精度，左下角维度，右上角精度，右上角维度]计算中心点坐标
    '''
    bounds = bounds['bounds']
    lat = (bounds[1] + bounds[3]) / 2
    lng = (bounds[0] + bounds[2]) / 2
    return lat, lng
```


```python
scene_id = r'D:\RSi\S2B_MSIL2A_20200709T163839_N0214_R126_T16TDM_20200709T211044.SAFE\GRANULE\L2A_T16TDM_A017455_20200709T164859\IMG_DATA\R10m\T16TDM_20200709T163839_B04_10m.jp2'
bounds = main.bounds(scene_id)
print('影像边界坐标：',bounds['bounds'])
z = 9 #需要调整不同的缩放级别，查看显示结果，如果缩放级别过大，影像则会模糊，无法查看细节；如果缩放比例来过大，数据量增加，则会增加计算时长
x, y = deg2num(*centroid(bounds), z) #指定缩放级别，转换影像中心点的经纬度坐标为金子塔瓦片坐标
print("影像中心点瓦片索引：",x,y)
# This call fetches the data
tile, mask = main.tile(scene_id, x, y, z, tilesize=512) #tilesize参数为瓦片大小，默认值为256
# Move the colour dimension to the last axis
tile = np.transpose(tile, (1, 2, 0))
# Rescale the intensity to make a pretty picture for human eyes
low, high = np.percentile(tile, (1, 97))
tile = exposure.rescale_intensity(tile, in_range=(low, high)) / 65535
# The tile only shows a subset of the whole image, which is several GB large
print("瓦片的形状：",tile.shape)
```

    影像边界坐标： [-88.21648331497754, 41.457515205853376, -86.88130582637692, 42.4526911672117]
    影像中心点瓦片索引： 131 190
    瓦片的形状： (512, 512, 1)
    


```python
plt.figure(figsize=(10,10))
plt.imshow(tile)
plt.axis("off")
plt.show()
```


    
<a href=""><img src="./imgs/25_05.png" height="auto" width="auto" title="caDesign"></a>
    


#### 1.1.2  Sentinel-2波段合成显示
波段合成显示的波段组合与Landsat部分阐述基本同，例如'4_Red、3_Green 、2_Blue'波段组合为自然真彩色。


```python
def Sentinel2_bandsComposite_show(RGB_bands,zoom=10,tilesize=512,figsize=(10,10)):
    %matplotlib inline
    import matplotlib.pyplot as plt
    import math,os
    import numpy as np
    from rio_tiler import main
    from skimage import exposure
    from rasterio.plot import show
    '''
    function - Sentinel-2波段合成显示。需要deg2num(lat_deg, lon_deg, zoom)和centroid(bounds)函数
    '''
    B_band=RGB_bands["B"]
    G_band=RGB_bands["G"]
    R_band=RGB_bands["R"]
    
    bounds = main.bounds(B_band)
    print('影像边界坐标：',bounds['bounds'])
    x, y = deg2num(*centroid(bounds), zoom)
    print("影像中心点瓦片索引：",x,y)

    tile_RGB_list=[np.squeeze(main.tile(band, x, y, zoom, tilesize=tilesize)[0]) for band in RGB_bands.values()]
    tile_RGB_array=np.array(tile_RGB_list).transpose(1,2,0)
    p2, p98=np.percentile(tile_RGB_array, (2,98))
    image=exposure.rescale_intensity(tile_RGB_array, in_range=(p2, p98)) / 65535
    
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
import os
sentinel2_root=r"D:\RSi\S2B_MSIL2A_20200709T163839_N0214_R126_T16TDM_20200709T211044.SAFE\GRANULE\L2A_T16TDM_A017455_20200709T164859\IMG_DATA\R10m"    
RGB_bands={          
          "R":os.path.join(sentinel2_root,'T16TDM_20200709T163839_B04_10m.jp2'),
          "G":os.path.join(sentinel2_root,'T16TDM_20200709T163839_B03_10m.jp2'),
          "B":os.path.join(sentinel2_root,'T16TDM_20200709T163839_B02_10m.jp2'),}

Sentinel2_bandsComposite_show(RGB_bands)
```

    影像边界坐标： [-88.21648331497754, 41.45751520585338, -86.88130582637692, 42.4526911672117]
    影像中心点瓦片索引： 262 380
    


    
<a href=""><img src="./imgs/25_06.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.2 无监督土地分类(聚类方法)
遥感影像的各个波段记录了地物的相关信息，那么以波段的数据作为机器学习的训练数据集，喂入相关模型，可以对应解决相关问题。其中之一为使用聚类的方法初步实现无监督土地分类（K-Menas算法）。sentinel-2影像有多个波段，可以尝试使用单个波段，或者多个波段作为特征向量，对比波段的合成显示，估计不同输入数据聚类结果的效果。

sentinel-2影像的信息均记录于下载文件夹下的'MTD_MSIL2A.xml'中，因此可以从该文件获取各个波段的路径。该文件给出的路径为相对于影像文件夹的相对路径。


```python
def Sentinel2_bandFNs(MTD_MSIL2A_fn):
    import xml.etree.ElementTree as ET
    '''
    funciton - 获取sentinel-2波段文件路径，和打印主要信息
    
    Paras:
    MTD_MSIL2A_fn - MTD_MSIL2A 文件路径
    
    Returns:
    band_fns_list - 波段相对路径列表
    band_fns_dict - 波段路径为值，反应波段信息的字段为键的字典
    '''
    Sentinel2_tree=ET.parse(MTD_MSIL2A_fn)
    Sentinel2_root=Sentinel2_tree.getroot()

    print("GENERATION_TIME:{}\nPRODUCT_TYPE:{}\nPROCESSING_LEVEL:{}".format(Sentinel2_root[0][0].find('GENERATION_TIME').text,
                                                           Sentinel2_root[0][0].find('PRODUCT_TYPE').text,                 
                                                           Sentinel2_root[0][0].find('PROCESSING_LEVEL').text
                                                          ))
    
    print("MTD_MSIL2A.xml 文件父结构:")
    for child in Sentinel2_root:
        print(child.tag,"-",child.attrib)
    print("_"*50)    
    band_fns_list=[elem.text for elem in Sentinel2_root.iter('IMAGE_FILE')] #[elem.text for elem in Sentinel2_root[0][0][11][0][0].iter()]
    band_fns_dict={f.split('_')[-2]+'_'+f.split('_')[-1]:f+'.jp2' for f in band_fns_list}
    print('获取sentinel-2波段文件路径:\n',band_fns_dict)
    
    return band_fns_list,band_fns_dict
    
MTD_MSIL2A_fn=r'D:\RSi\S2B_MSIL2A_20200709T163839_N0214_R126_T16TDM_20200709T211044.SAFE\MTD_MSIL2A.xml'
band_fns_list,band_fns_dict=Sentinel2_bandFNs(MTD_MSIL2A_fn)
```

    GENERATION_TIME:2020-07-09T21:10:44.000000Z
    PRODUCT_TYPE:S2MSI2A
    PROCESSING_LEVEL:Level-2A
    MTD_MSIL2A.xml 文件父结构:
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}General_Info - {}
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Geometric_Info - {}
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Auxiliary_Data_Info - {}
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Quality_Indicators_Info - {}
    __________________________________________________
    获取sentinel-2波段文件路径:
     {'B02_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_B02_10m.jp2', 'B03_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_B03_10m.jp2', 'B04_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_B04_10m.jp2', 'B08_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_B08_10m.jp2', 'TCI_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_TCI_10m.jp2', 'AOT_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_AOT_10m.jp2', 'WVP_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_WVP_10m.jp2', 'B02_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B02_20m.jp2', 'B03_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B03_20m.jp2', 'B04_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B04_20m.jp2', 'B05_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B05_20m.jp2', 'B06_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B06_20m.jp2', 'B07_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B07_20m.jp2', 'B8A_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B8A_20m.jp2', 'B11_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B11_20m.jp2', 'B12_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B12_20m.jp2', 'TCI_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_TCI_20m.jp2', 'AOT_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_AOT_20m.jp2', 'WVP_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_WVP_20m.jp2', 'SCL_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_SCL_20m.jp2', 'B01_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B01_60m.jp2', 'B02_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B02_60m.jp2', 'B03_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B03_60m.jp2', 'B04_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B04_60m.jp2', 'B05_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B05_60m.jp2', 'B06_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B06_60m.jp2', 'B07_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B07_60m.jp2', 'B8A_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B8A_60m.jp2', 'B09_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B09_60m.jp2', 'B11_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B11_60m.jp2', 'B12_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B12_60m.jp2', 'TCI_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_TCI_60m.jp2', 'AOT_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_AOT_60m.jp2', 'WVP_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_WVP_60m.jp2', 'SCL_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_SCL_60m.jp2'}
    

根据返回字典的键，可以提取对应的波段路径名。'earthpy'库的`es.stack`方法可以融合多个波段，同时会返回波段额元数据，包括：'driver'驱动，'dtype'数据类型，'nodata'空值，'width'影像宽度，'height'影像高度，'count'波段数量，'crs'坐标系统（投影），'transform'变换，'blockxsize'x向单元数量（每个单元的精度为10m，即一个像素代表10m的实际地理空间，小于10m的地物则无法分辨），'blockysize'y向单元数量。


```python
import os
import matplotlib.pyplot as plt
import earthpy.spatial as es
import earthpy.plot as ep
import geopandas as gpd

imgs_root=r"D:\RSi\S2B_MSIL2A_20200709T163839_N0214_R126_T16TDM_20200709T211044.SAFE"
bands_selection=["B02_10m","B03_10m","B04_10m","B08_10m"]
stack_bands=[os.path.join(imgs_root,band_fns_dict[b]) for b in bands_selection]
array_stack, meta_data=es.stack(stack_bands)
print("meta_data:\n",meta_data)

ep.plot_bands(array_stack,title=bands_selection,cols=array_stack.shape[0],cbar=True,figsize=(10,10))
plt.show()
```

    meta_data:
     {'driver': 'JP2OpenJPEG', 'dtype': 'uint16', 'nodata': None, 'width': 10980, 'height': 10980, 'count': 4, 'crs': CRS.from_epsg(32616), 'transform': Affine(10.0, 0.0, 399960.0,
           0.0, -10.0, 4700040.0), 'blockxsize': 1024, 'blockysize': 1024, 'tiled': True}
    


    
<a href=""><img src="./imgs/25_07.png" height="auto" width="auto" title="caDesign"></a>
    


在QGIS中读取一个波段，或多个波段的组合显示，绘制裁切边界（设置坐标为WGS84，不配置投影，读取后根据影像的投影再进行定义），用于影像的裁切。裁切文件保存用于指定的文件夹下。


```python
crop_output_dir=r'D:\RSi\crop'
imgs_crs=meta_data['crs']

shape_polygon_fp=r'.\data\geoData\sentinel2Chicago_boundary.shp'
crop_bound=gpd.read_file(shape_polygon_fp)
crop_bound_proj=crop_bound.to_crs(imgs_crs)

crop_imgs=es.crop_all([os.path.join(imgs_root,f+'.jp2') for f in band_fns_list], crop_output_dir, crop_bound_proj, overwrite=True) #对所有波段band执行裁切
print("finished cropping...")
```

    finished cropping...
    

显示裁切后的影像。


```python
import glob
croppedImgs_fns=glob.glob(crop_output_dir+"/*.jp2")
croppedBands_fnsDict={f.split('_')[-3]+'_'+f.split('_')[-2]:f for f in croppedImgs_fns}

bands_selection_=["B02_10m","B03_10m","B04_10m","B08_10m"]  #,"AOT_10m","WVP_10m"
cropped_stack_bands=[croppedBands_fnsDict[b] for b in bands_selection_]

cropped_array_stack,_=es.stack(cropped_stack_bands)
ep.plot_bands(cropped_array_stack,title=bands_selection_,cols=cropped_array_stack.shape[0],cbar=True,figsize=(10,10))
plt.show()
```


    
<a href=""><img src="./imgs/25_08.png" height="auto" width="auto" title="caDesign"></a>
    


可以尝试调整不同的聚类数量`n_cluster`参数，分类越多划分的地物类别也就越细。基于聚类无监督分类的结果并没有明确分类的名称，需要结合已经聚类的结果，根据实际地物情况判别。注意喂入模型数据的形状为(样本数，特征数)，如果理解为矩阵，则每一列为一个特征向量，每一行为一个样本的多个特征值。通常所输入的特征数越多，即波段数越多，分类的精度越好。


```python
from sklearn import cluster
import matplotlib.pyplot as plt

X=cropped_array_stack.reshape(cropped_array_stack.shape[0],-1).transpose(1,0)
print(X)

k_means=cluster.KMeans(n_clusters=8)
k_means.fit(X)
X_cluster = k_means.labels_
X_cluster = X_cluster.reshape(cropped_array_stack[0,:,:].shape)

plt.figure(figsize=(8,8))
plt.imshow(X_cluster, cmap="hsv")

plt.show()
```

    [[2972 3248 3344 3550]
     [2574 2548 2602 2768]
     [1406 1406 1824 1920]
     ...
     [ 825  749  519  484]
     [ 811  738  516  498]
     [ 776  760  538  488]]
    


    
<a href=""><img src="./imgs/25_09.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.3 VGG16卷积神经网络
[VGGNet](https://arxiv.org/abs/1409.1556)研究了在大规模图像识别环境下，卷积网络深度对识别精度的影响。其主要贡献是使用非常小的($3 \times 3$)卷积滤波器(卷积核)和($2 \times 2$)的最大池化层反复堆叠，在深度不断增加的网络下的表现评估。当将网络深度推进到16-19个全支持层时（图表的第C、D列为16层，第E列为19层），可以发现识别精度得以显著提升。该项研究最初用于[ImageNet](http://www.image-net.org/)数据集，并同时能够很好的泛化到其它数据集。在解释['torchvision.models'](https://richiebao.github.io/Urban-Spatial-Data-Analysis_python/#/./notebook_code/CNN)时则是以VGG网络为对象，可返回查看。

对VGGNet网络的理解同样可以对应到图像特征提取-尺度不变特征转换下尺度空间(scale space)的概念上。因为不同的地物尺寸不同，因此对于同一地理范围下的影像，分辨率越高，例如0.3-05m，则可以识别出行人轮廓。但是10m的高空分辨率则无法识别，而对于通常大于10m的对象，例如建筑，绿地则可以识别。这个变换的分辨率就是尺度空间的纵向深度，由降采样实现。对应到VGG网络上，就是网络深度的不断增加，是由'maxpool'最大池化层实现。因为不同地物的尺寸多样，但是通常可以形成一个连续的尺寸变化，例如从室外摆放的餐具，过往或静坐的行人，到车辆，建筑，再到农田和成片的林地，因此为了检测到每一地物对应的尺度空间，采用$2 \times 2$的最大池化能够很好的捕捉到不同的地物。即低分辨率的图像可以忽略掉较小的对象，而专注于该尺度及之上的对象，以此类推。在尺度空间中还有一个水平向，使用不同的卷积核检测同一尺度即深度下地物即图像的特征。不同的卷积核会识别出不同的特征内容，例如对象间的边界形状，颜色的差异变化，以及很多一般常识无法判定但却可以区分对象的特征。因此在每一深度进行卷积操作时，通常要使用多个不同的卷积核，并随机初始化卷积核数值，以捕捉到对象的特征。这对应到深度网络结构中的输出通道数。VGGNet在深度增加过程中，所使用的卷积核大小不变，均为$3 \times 3$。因为深度的逐层增加，不同尺度的地物会被捕捉到，同一大小的卷积核可以检测到不同地物的特征。同时，使用的卷积数量在增加，以适应深度增加，尺度增大，即图像越加模糊时的特征提取。图像的特征并不仅表现在一次卷积的结果上，例如如果应用一次卷积提取了对象的轮廓边界，那么仍然可以再应用卷积提取对象轮廓边界的特征，以此类推。这可以用于解释每一层深度/尺度下使用多层卷积的原因。

<a href=""><img src="./imgs/25_01.png" height='auto' width='600' title="caDesign"></a>

将上述表格的第D列，即VGG16，通过方块序列图的形式可以更好的表述，观察层级间的变化。

<a href=""><img src="./imgs/25_02.png" height='auto' width='1000' title="caDesign"></a>

* [ImageNet数据集](http://www.image-net.org/)

ImageNet数据集于2007年开始建设，已有超过1500万张图像，2万多个类别，是一个庞大的数据集。是根据[WordNet](https://wordnet.princeton.edu/)层次就结构(目前只有名词nouns)组织的图像数据库。其中层次结构的每一个节点都由成百上千张图像描述。ImageNet数据集1000个类别文件可以从[imagenet_classes.txt](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)处下载，其分类涉及动植物，各类人造物。

VGGNet预训练模型已经置于[torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)模型库中，通过下载该模型，来尝试识别对象。参考[VGG-NETS](https://pytorch.org/hub/pytorch_vision_vgg/)

> 参考文献：
1. Karen Simonyan∗ & Andrew Zisserman+.[VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/abs/1409.1556)[J].Published as a conference paper at ICLR 2015.arXiv:1409.1556v6[cs.CV] 10 Apr 2015
2. [VGG-NETS](https://pytorch.org/hub/pytorch_vision_vgg/)

* 01-下载预训练的VGG16模型


```python
import torch
model=torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
model.eval()
```

    Using cache found in C:\Users\richi/.cache\torch\hub\pytorch_vision_v0.6.0
    




    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace=True)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace=True)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace=True)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=1000, bias=True)
      )
    )



* 02-读取一幅图像。执行调整图像大小Resize、裁切CenterCrop、转换为张量ToTensor和标准化Normalize等操作，使其满足网络结构的数据输入需求。


```python
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

cat_01=r'./data/stuff_01.jpg' #cat_01.png;stuff_01.jpg
cat_img=Image.open(cat_01).convert('RGB')
plt.imshow(cat_img)
plt.show()
```


    
<a href=""><img src="./imgs/25_10.png" height="auto" width="auto" title="caDesign"></a>
    



```python
input_image=Image.open(cat_01)
preprocess=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor=preprocess(input_image)
input_batch=input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
print("VGG16输入数据的形状（batchsize, nChannels, Height, Width）：",input_batch.shape)
```

    VGG16输入数据的形状（batchsize, nChannels, Height, Width）： torch.Size([1, 3, 224, 224])
    

* 03 - 图像中的内容预测概率


```python
#输入数据和模型传入GPU执行运输 move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch=input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output=model(input_batch)
#全连接最后一层的线性输出通道数为1000，对应ImageNet数据集的1000个分类 Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities= torch.nn.functional.softmax(output[0], dim=0)
print("预测的1000个分类联合概率分布数组的形状：",probabilities.shape)
```

    预测的1000个分类联合概率分布数组的形状： torch.Size([1000])
    

* 04 - 打印预测概率分布中最大的前几个对象，可以观察到预测的对象，desktop computer(及monitor,laptop,screen,computer keyboard),notebook、desk,都出现在该图像中。而printer和modem则没有，但是modem和插座的形状比较近似。


```python
# Read the categories ImageNet数据集1000个类别文件可以从[imagenet_classes.txt](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)处下载
with open("./data/imagenet_classes.txt", "r") as f:
    categories=[s.strip() for s in f.readlines()]
#显示所预测图像，前几个最大概率对应的分类名 Show top categories per image
top5_prob,top5_catid=torch.topk(probabilities, 10)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

    desktop computer 0.16991862654685974
    notebook 0.15042588114738464
    desk 0.14144189655780792
    monitor 0.13136830925941467
    laptop 0.10931627452373505
    mouse 0.10452879965305328
    screen 0.07919996231794357
    computer keyboard 0.03401118516921997
    printer 0.020951425656676292
    modem 0.008885659277439117
    

### 1.4 SegNet遥感影像语义分割/解译
SegNet于2016年提出，其核心的概念是将网络划分为encoder编码器网络，decoder解码器网络，和一个像素级的分类层(SoftMax)。编码器网络结构与VGG16的13个特征提取卷积层结构相同。而解码器网络的结构与编码器网络刚好相逆，可以理解为反卷积的过程，每个编码器层都对应一个解码器层，将编码结果的低分辨率特征重新映射到输入时的分辨率，以便进行像素级分类，为每个像素生成类概率，输出不同分类的最大值，而得到图像分割图。编码过程是池化层（`nn.MaxPool2d(2, return_indices=True)`）下采样的过程，而解码过程是提取的特征值上采样(`nn.MaxUnpool2d(2)`)的过程。*SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation*一文给出的SegNet网络结构图，非常清晰的解释了这一过程。

<a href=""><img src="./imgs/25_03.png" height='auto' width='500' title="caDesign"></a>

下采样(pooling)就是池化层的作用，增加网络的深度。对于最大池化层，是提取区域内最大值作为输出，那么就可以得到最大值所在位置的索引。因此在上采样(upsampling)的过程中，对于$2 \times 2$池化下采样结果执行上采样时，已经丢失3个权重值，在将特征图放大2倍后，原来特征图的数据会根据下采样时获取的位置索引归位放入。对于池化最大值位置索引，PyTorch的`nn.MaxPool2d()`下`return_indices=True`参数配置可以返回索引值。

<a href=""><img src="./imgs/25_04.png" height='auto' width='300' title="caDesign"></a>

> 参考文献
1. Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, Senior Member, IEEE. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation[J].arXiv:1511.00561v3 [cs.CV] 10 Oct 2016
2. [Deep learning for Earth Observation](https://github.com/nshaud/DeepNetsForEO)

* nn.MaxPool2d(2, return_indices=True)

下述代码片段展示了编码器最大池化，及解码器应用索引值上采样的过程。


```python
pool=nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
input = torch.tensor([[[[ 0.,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8, 9, 10, 11],
                        [12, 13, 14, 15]]]])

output, indices=pool(input)
print("最大池化索引：\n",indices)
print("最大池化结果：\n",output)

upsampling=unpool(output, indices)
print("根据池化索引上采样结果：\n",upsampling)
```

    最大池化索引：
     tensor([[[[ 5,  7],
              [13, 15]]]])
    最大池化结果：
     tensor([[[[ 5.,  7.],
              [13., 15.]]]])
    根据池化索引上采样结果：
     tensor([[[[ 0.,  0.,  0.,  0.],
              [ 0.,  5.,  0.,  7.],
              [ 0.,  0.,  0.,  0.],
              [ 0., 13.,  0., 15.]]]])
    

* 01-调入所用的库


```python
#imports and stuff
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
import os

# Matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
```

*  02 - [ISPRS dataset](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/)数据集，及数据查看，预处理，和小批量可迭代数据加载

[ISPRS](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)是遥感图像数据集。对于遥感图像数据集，因为大量影像的开源和图像解译工具的存在（例如e-Cognition），可以用传统的解译工具建立影像的分割标签，从而建立数据集。因此目前有大量的遥感影像数据集可以使用，避免自行从新建立。下载的ISPRS数据集，包括三个地方分别为Potsdam,Toronto和Vaihingen。每个对应区域的所有数据放置于以地名命名的文件夹下，包含.tif格式（GTiff驱动），投影为CRS-->EPSG:32633 - WGS 84 / UTM zone ?N - Projected的原始影像，以及影像标签，标签类别包括"roads", "buildings", "low veg.", "trees", "cars", "clutter"，可以分辨出主要的地物内容。如果该数据集的标签不能满足解译后使用上的需求，可以用其它满足要求的影像数据集替换，或者用传统工具自行解译部分影像用作训练数据集。数据有blue,green,red和NIR四个波段，不过波段已经合成为RGB,IRRG,RGBIR等形式，通常放置于各自单独的文件夹下。下述训练的数据使用的为IRRG合成的波段，即NIR+red+green。Vaihingen区域数据是由德国摄影测量和遥感协会(German Association of Photogrammetry and Remote Sensing,DGPF)用于测试数字航拍数据的子集。图像为8cm地面分辨率。

建立数据存放的字符串格式化模式，在后续调用`class ISPRS_dataset(torch.utils.data.Dataset)`时使用。将训练集的数据对应到DATA_FOLDE文件夹下，训练集的标签对应到LABEL_FOLDER文件夹下，测试集的数据对应到ERODED_FOLDER文件夹下。


```python
# Parameters
WINDOW_SIZE = (256, 256) # Patch size
STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = r"D:/dataset/ISPRS/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch 10

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

DATASET = 'Vaihingen'

if DATASET == 'Potsdam':
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    # Uncomment the next line for IRRG data
    # DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif'
    # For RGB data
    DATA_FOLDER = MAIN_FOLDER + '2_Ortho_RGB/top_potsdam_{}_RGB.tif'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'    
elif DATASET == 'Vaihingen':
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
```

数据查看，包括影像和对应标签。定义的函数` convert_to_color(arr_2d, palette=palette)`和`convert_from_color(arr_3d, palette=invert_palette)`给定数值和对应RGB颜色值映射字典，实现数值和颜色之间的互相转换。


```python
# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """数值标签转换为RGB颜色标签 Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """RGB颜色标签转换为数值标签（灰度图） RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

# We load one tile from the dataset and we display it
img = io.imread(r'D:\dataset\ISPRS\Vaihingen\top/top_mosaic_09cm_area11.tif')
fig = plt.figure()
fig.add_subplot(121)
plt.imshow(img)

# We load the ground truth
gt = io.imread(r'D:\dataset\ISPRS\Vaihingen\gts_for_participants/top_mosaic_09cm_area11.tif')
fig.add_subplot(122)
plt.imshow(gt)
plt.show()

# We also check that we can convert the ground truth into an array format
array_gt = convert_from_color(gt)
print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)
```


    
<a href=""><img src="./imgs/25_11.png" height="auto" width="auto" title="caDesign"></a>
    


    Ground truth in numerical format has shape (2566,1893) : 
     [[3 3 3 ... 3 3 3]
     [3 3 3 ... 3 3 3]
     [3 3 3 ... 3 3 3]
     ...
     [2 2 2 ... 1 1 1]
     [2 2 2 ... 1 1 1]
     [2 2 2 ... 1 1 1]]
    

定义小批量可迭代数据加载类，同时执行图像增广(image augmentation)，由定义的`data_augmentation`函数执行随机的翻转和镜像；并标准化数据到[0,1]。同时标识`data_augmentation`函数有`@classmethod`装饰器，即标记该方法为类方法的装饰器。除了由实例对象调用外，可以直接由该类调用。如果作为父类，其子类也可以直接调用父类的类方法。


```python
class C:
    @classmethod
    def f(cls,arg_str):
        print(cls,arg_str)
class C_child(C):
    pass
print(C.f("类对象调用类方法..."))
c=C()
print(c.f("类实例调用类方法..."))
print(C_child.f("子类调用父类的类方法..."))
```

    <class '__main__.C'> 类对象调用类方法...
    None
    <class '__main__.C'> 类实例调用类方法...
    None
    <class '__main__.C_child'> 子类调用父类的类方法...
    None
    


```python
def get_random_pos(img, window_shape):
    """给定窗口大小，随机提取部分图像 Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

# Dataset class
class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                            cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()
        
        self.augmentation = augmentation
        self.cache = cache
        
        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
            
    
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = 1/255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        
        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))
```

加载数据。并切分数据集为训练和测试数据集。


```python
# Load the datasets
if DATASET == 'Potsdam':
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    all_ids = ["".join(f.split('')[5:7]) for f in all_files]
elif DATASET == 'Vaihingen':
    #all_ids = 
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]
# Random tile numbers for train/test split
train_ids = random.sample(all_ids, 2 * len(all_ids) // 3 + 1)
test_ids = list(set(all_ids) - set(train_ids))
print("Tiles for training : ", train_ids)
print("Tiles for testing : ", test_ids)

train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)
```

    Tiles for training :  ['26', '23', '13', '7', '1', '3', '34', '30', '5', '32', '21']
    Tiles for testing :  ['15', '11', '28', '17', '37']
    

* 03 - 定义网络


```python
class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)
        
        #-------------------------------------------------------------
        
        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)
        
        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)
        
        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)
        
        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        
    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)
        
        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)
        
        #-------------------------------------------------------------
        
        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = F.log_softmax(self.conv1_1_D(x))
        return x
```

从地址 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth' 下载VGG16网络模型预训练参数。因为下载的预训练参数对应的层名与上述模型定义的不同，因此需要一一对位，将权值映射到新的层名上来。然后应用`net.state_dict().update(mapped_weights)`方法更新权值。


```python
# instantiate the network
net = SegNet()
```


```python
import os
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

# Download VGG-16 weights from PyTorch
vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
    weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
mapped_weights = {}
for k_vgg, k_segnet in zip(vgg16_weights.keys(), net.state_dict().keys()):
    if "features" in k_vgg:
        mapped_weights[k_segnet] = vgg16_weights[k_vgg]
        print("Mapping {} to {}".format(k_vgg, k_segnet))
        
try:
    net.state_dict().update(mapped_weights)
    print("_"*50)
    print("Loaded VGG-16 weights in SegNet !")
except:
    print("Ignore missing keys")
    pass
```

    Mapping features.0.weight to conv1_1.weight
    Mapping features.0.bias to conv1_1.bias
    Mapping features.1.weight to conv1_1_bn.weight
    Mapping features.1.bias to conv1_1_bn.bias
    Mapping features.1.running_mean to conv1_1_bn.running_mean
    Mapping features.1.running_var to conv1_1_bn.running_var
    Mapping features.3.weight to conv1_1_bn.num_batches_tracked
    Mapping features.3.bias to conv1_2.weight
    Mapping features.4.weight to conv1_2.bias
    Mapping features.4.bias to conv1_2_bn.weight
    Mapping features.4.running_mean to conv1_2_bn.bias
    Mapping features.4.running_var to conv1_2_bn.running_mean
    Mapping features.7.weight to conv1_2_bn.running_var
    Mapping features.7.bias to conv1_2_bn.num_batches_tracked
    Mapping features.8.weight to conv2_1.weight
    Mapping features.8.bias to conv2_1.bias
    Mapping features.8.running_mean to conv2_1_bn.weight
    Mapping features.8.running_var to conv2_1_bn.bias
    Mapping features.10.weight to conv2_1_bn.running_mean
    Mapping features.10.bias to conv2_1_bn.running_var
    Mapping features.11.weight to conv2_1_bn.num_batches_tracked
    Mapping features.11.bias to conv2_2.weight
    Mapping features.11.running_mean to conv2_2.bias
    Mapping features.11.running_var to conv2_2_bn.weight
    Mapping features.14.weight to conv2_2_bn.bias
    Mapping features.14.bias to conv2_2_bn.running_mean
    Mapping features.15.weight to conv2_2_bn.running_var
    Mapping features.15.bias to conv2_2_bn.num_batches_tracked
    Mapping features.15.running_mean to conv3_1.weight
    Mapping features.15.running_var to conv3_1.bias
    Mapping features.17.weight to conv3_1_bn.weight
    Mapping features.17.bias to conv3_1_bn.bias
    Mapping features.18.weight to conv3_1_bn.running_mean
    Mapping features.18.bias to conv3_1_bn.running_var
    Mapping features.18.running_mean to conv3_1_bn.num_batches_tracked
    Mapping features.18.running_var to conv3_2.weight
    Mapping features.20.weight to conv3_2.bias
    Mapping features.20.bias to conv3_2_bn.weight
    Mapping features.21.weight to conv3_2_bn.bias
    Mapping features.21.bias to conv3_2_bn.running_mean
    Mapping features.21.running_mean to conv3_2_bn.running_var
    Mapping features.21.running_var to conv3_2_bn.num_batches_tracked
    Mapping features.24.weight to conv3_3.weight
    Mapping features.24.bias to conv3_3.bias
    Mapping features.25.weight to conv3_3_bn.weight
    Mapping features.25.bias to conv3_3_bn.bias
    Mapping features.25.running_mean to conv3_3_bn.running_mean
    Mapping features.25.running_var to conv3_3_bn.running_var
    Mapping features.27.weight to conv3_3_bn.num_batches_tracked
    Mapping features.27.bias to conv4_1.weight
    Mapping features.28.weight to conv4_1.bias
    Mapping features.28.bias to conv4_1_bn.weight
    Mapping features.28.running_mean to conv4_1_bn.bias
    Mapping features.28.running_var to conv4_1_bn.running_mean
    Mapping features.30.weight to conv4_1_bn.running_var
    Mapping features.30.bias to conv4_1_bn.num_batches_tracked
    Mapping features.31.weight to conv4_2.weight
    Mapping features.31.bias to conv4_2.bias
    Mapping features.31.running_mean to conv4_2_bn.weight
    Mapping features.31.running_var to conv4_2_bn.bias
    Mapping features.34.weight to conv4_2_bn.running_mean
    Mapping features.34.bias to conv4_2_bn.running_var
    Mapping features.35.weight to conv4_2_bn.num_batches_tracked
    Mapping features.35.bias to conv4_3.weight
    Mapping features.35.running_mean to conv4_3.bias
    Mapping features.35.running_var to conv4_3_bn.weight
    Mapping features.37.weight to conv4_3_bn.bias
    Mapping features.37.bias to conv4_3_bn.running_mean
    Mapping features.38.weight to conv4_3_bn.running_var
    Mapping features.38.bias to conv4_3_bn.num_batches_tracked
    Mapping features.38.running_mean to conv5_1.weight
    Mapping features.38.running_var to conv5_1.bias
    Mapping features.40.weight to conv5_1_bn.weight
    Mapping features.40.bias to conv5_1_bn.bias
    Mapping features.41.weight to conv5_1_bn.running_mean
    Mapping features.41.bias to conv5_1_bn.running_var
    Mapping features.41.running_mean to conv5_1_bn.num_batches_tracked
    Mapping features.41.running_var to conv5_2.weight
    __________________________________________________
    Loaded VGG-16 weights in SegNet !
    


```python
if torch.cuda.is_available():
    net.to('cuda')
```

* 04 - 定义训练模型相关函数，损失函数，预测精度及相关度量值（全局精度，F1分数和kappa系数）等内容。


```python
def CrossEntropy2d(input, target, weight=None, size_average=True):
    """定义损失函数——2D版交叉熵损失 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target,weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def accuracy(input, target):
    '''定义预测精度'''
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(top, step=10, window_size=(20,20)):
    """给定步幅，窗口形状，滑动过整幅图像，迭代计算窗口所在图像x,y位置值，返回每一切分图像(patch)的x,y坐标值和高宽大小，即yield返回值。参数step可以控制切分窗口叠合的程度 Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def count_sliding_window(top, step=10, window_size=(20,20)):
    """计算图像滑动给定窗口大小的数量 Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, label_values=LABELS):
    '''预测值度量'''
    cm = confusion_matrix(
            gts,
            predictions,
            range(len(label_values)))
    
    print("Confusion matrix :")
    print(cm)
    
    print("---")
    
    #全局精度 Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    print("---")
    
    #F1分数 Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")
        
    #计算kappa系数 Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    return accuracy
```

使用标准的随机梯度下降算法优化网络的权值。如果调入了预先训练的VGG16模型参数，则可调整学习率。即encoder编码部分（VGG16卷积，特征提取部分）的训练速度为decoder解码器的一半。


```python
base_lr = 0.01
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
```

* 05 - 定义测试函数，显示RGB影像，及对应的真实值，及预测值图像。计算`metrics`函数定义的相关预测度量值。


```python
def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    
    # Switch the network to inference mode
    net.eval()

    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
            # Display in progress results
            if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                    _pred = np.argmax(pred, axis=-1)
                    fig = plt.figure()
                    fig.add_subplot(1,3,1)
                    plt.imshow(np.asarray(255 * img, dtype='uint8'))
                    fig.add_subplot(1,3,2)
                    plt.imshow(convert_to_color(_pred))
                    fig.add_subplot(1,3,3)
                    plt.imshow(gt)
                    clear_output()
                    plt.show()
                    
            # Build the tensor
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
            
            # Do the inference
            outs = net(image_patches)
            outs = outs.data.cpu().numpy()
            
            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1,2,0))
                pred[x:x+w, y:y+h] += out
            del(outs)

        pred = np.argmax(pred, axis=-1)

        # Display the result
        clear_output()
        fig = plt.figure()
        fig.add_subplot(1,3,1)
        plt.imshow(np.asarray(255 * img, dtype='uint8'))
        fig.add_subplot(1,3,2)
        plt.imshow(convert_to_color(pred))
        fig.add_subplot(1,3,3)
        plt.imshow(gt)
        plt.show()

        all_preds.append(pred)
        all_gts.append(gt_e)

        clear_output()
        # Compute some metrics
        metrics(pred.ravel(), gt_e.ravel())
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy
```

* 06 - 定义训练函数。输出损失曲线，显示RGB影像，及对应的真实值，及预测值图像。打印损失值和精度值，观察模型训练情况。同时指定文件夹，保存模型参数。


```python
from IPython.display import clear_output

def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch = 5):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    criterion = nn.NLLLoss2d(weight=weights)
    iter_ = 0
    
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()
            
            #print("_"*50)
            #print(iter_)
            #print(loss.data.item())
            losses[iter_] = loss.data.item()#losses[iter_] = loss.data[0]
            mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_])
            
            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data.item(), accuracy(pred, gt)))  #100. * batch_idx / len(train_loader), loss.data[0], accuracy(pred, gt)))
                plt.plot(mean_losses[:iter_]) and plt.show()
                fig = plt.figure()
                fig.add_subplot(131)
                plt.imshow(rgb)
                plt.title('RGB')
                fig.add_subplot(132)
                plt.imshow(convert_to_color(gt))
                plt.title('Ground truth')
                fig.add_subplot(133)
                plt.title('Prediction')
                plt.imshow(convert_to_color(pred))
                plt.show()
            iter_ += 1
            
            del(data, target, loss)
            
        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            acc = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
            torch.save(net.state_dict(), './model/segnet256_epoch{}_{}'.format(e, acc))
    torch.save(net.state_dict(), './model/segnet_final')
```

* 07 - 训练模型


```python
train(net, optimizer, 50, scheduler)
```

    Confusion matrix :
    [[1130697   52437   59581   16636    3790    1023]
     [  67449 1120806    9512    2880     402     484]
     [  57799   19593  701057  180705     633       0]
     [   5195     718   42060  926701      60       0]
     [  12159    3085     402     853   39486      60]
     [      0       0       0       0       0       0]]
    ---
    4456263 pixels processed
    Total accuracy : 87.9379650617569%
    ---
    F1Score :
    roads: 0.8912027485720974
    buildings: 0.9347169427380522
    low veg.: 0.7910825948333304
    trees: 0.8815191754232681
    cars: 0.7864483747609943
    clutter: 0.0
    ---
    Kappa: 0.8395543092166328
    

                                                 

    Confusion matrix :
    [[5195686  209493  329821  101786   18452    2370]
     [ 256497 5068891  102891   19287    1698    1671]
     [ 227024   89954 3164475  898865    1414     331]
     [  49351   13528  430501 5635404     242     294]
     [  43694    7782    1854    2179  141552     745]
     [   2316    2641    6283      42     568       0]]
    ---
    22029582 pixels processed
    Total accuracy : 87.18280719080371%
    ---
    F1Score :
    roads: 0.8933300183903682
    buildings: 0.934941674173659
    low veg.: 0.7518453559847791
    trees: 0.8814351394315565
    cars: 0.7826346577023874
    clutter: 0.0
    ---
    Kappa: 0.8289082609899481
    

<a href=""><img src="./imgs/25_19.png" height="auto" width=800 title="caDesign"></a>
    

* 08 - 加载保存的SegNet模型参数，应用测试数据集测试模型。通过配置stride参数，设置图像被切分为多个小块之间的重叠程度。重叠程度由stride参数和WINDOW_SIZE=(256, 256)参数（即patch大小）确定。

注意，在应用所训练的模型解译新的图像时，新图像的形式应该与训练数据的形式保持一致或近似，这样才能够保证正确的预测结果。例如图像的高空分辨率，以及波段合成信息能够基本相同。


```python
net.load_state_dict(torch.load('./model/segnet_final'))
if torch.cuda.is_available():
    net.to('cuda')
```


```python
_, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
```

    Confusion matrix :
    [[ 807365    5540    2921     933      36       0]
     [   1823  854972    2836     540       0       0]
     [   3496    2499 1414713    5676       0       0]
     [   2211    1272   10821  471329       2       0]
     [    530       1       0      14   12146       0]
     [      0       0       0       0       0       0]]
    ---
    3601676 pixels processed
    Total accuracy : 98.85744858782411%
    ---
    F1Score :
    roads: 0.9892845327223045
    buildings: 0.991585167487699
    low veg.: 0.9901146911387754
    trees: 0.9777321867347352
    cars: 0.9765628140703517
    clutter: nan
    ---
    Kappa: 0.9840434423827237
    

    <ipython-input-9-68568e73fae5>:75: RuntimeWarning: invalid value encountered in double_scalars
      F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
    

    Confusion matrix :
    [[4116613  111172  208743   56765    5369      87]
     [ 164358 4398202   70400   12605     888      60]
     [ 111636   50764 4767993  652294     241       0]
     [  22233    7453  258441 4401913      50       1]
     [  26559    2590    2076     941  102238      38]
     [   2337    2772    6250      19     472       0]]
    ---
    19564573 pixels processed
    Total accuracy : 90.91411808476474%
    ---
    F1Score :
    roads: 0.9206865876766916
    buildings: 0.9541120928262006
    low veg.: 0.8751155267068013
    trees: 0.8970106661199997
    cars: 0.839048009848174
    clutter: 0.0
    ---
    Kappa: 0.8791652103047238
    

<a href=""><img src="./imgs/25_20.png" height="auto" width=800 title="caDesign"></a>    

* 09 - 显示与保存预测的图像分割/解译


```python
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.figure(figsize=(20,5))

i=0
for p, id_ in tqdm(zip(all_preds,test_ids),total=len(all_preds),leave=False):
    img = convert_to_color(p)
    plt.subplot(1,len(all_preds),i+1)
    plt.imshow(img)
    plt.axis('off')
    
    io.imsave('./results/segment_pred/inference_tile_{}.png'.format(id_), img)
    i+=1
plt.show()
```

    100%|██████████| 5/5 [00:05<00:00,  1.08s/it]
    


    
<a href=""><img src="./imgs/25_12.png" height="auto" width="auto" title="caDesign"></a>
    


> 可以尝试在DUC(Dense Upsampling Convolution)图像分割部分，用SegNet模型替换DUC实现。

### 1.5 超像素级分割(superpixels-segmentation)，高空分辨率特征尺度界定，及尺度的深度流动线/特征区域的延申
在景观生态学中，斑块－廊道－基质模型是构成景观空间结构，描述景观空间异质性的一个基本模式。其中斑块是景观格局中的基本组成单元，是指不同于周围背景，相对均质的非线性区域。自然界各种等级系统都普遍存在时间和空间的斑块化。反应系统内部和系统间的相似性或相异性。不同斑块的大小、形状、边界性质及斑块的距离等空间分布特征构成了不同的生态带，形成了生态系统的差异，调节生态过程。廊道是不同于景观基质的现状或带状的景观要素，例如河流廊道，生态廊道等。其中生态廊道又称野生动物生态廊道或绿色廊道，是指用于连接因人类活动或构筑物而被隔开的野生动物种群生境的区域。生态廊道有利于野生动物的迁移扩散，提高生境间的连接，促进濒危物种不同群间的基因交流，降低种群灭绝风险。基质则是景观中面积最大，连接性最好的景观要素类型。斑廊基景观空间结构的提出为城市格局规划提供了依据，在宏观尺度上给出了保护自然生物的空间形式。那么对于一个区域，如何自然界定斑块、廊道和基质的区域？又或者即使是一个可以肉眼辨识的斑块，这个斑块自身也是呈现变化的，可以表现在地物的变化，例如可见的不同林地，不同物种的农田等，或者不可见的地表温度变化，物质流动等，那么又如何细分斑块的空间区域，挖掘斑块变化区域的流动方向或子区域？

一方面是需要能够反映地物变化的信息数据，例如遥感影像的各个波段对不同地物的探测，sentinel-2影像中新增加的5、6、7波段(red edge)，可以有效监测植被健康信息；或衍生数据，例如反演的地表温度，以及NDVI等反应植被分布的指数，NDWI反应水体分布的指数，NDBI反应建城区分布的指数等。另一方面在分析这些数据时，可以介入超像素级分割的概念，探索由像素（或空间点数据）局部分组形成的区域，这类似于聚类的方法，将具有同一或近似属性的区域优先聚集即分割，分割区域的变化根据所提供反应不同内容的数据所确定，例如探索植被分布的NDVI则优先聚集植被指数临近的区域。也可以组合波段，例如red,green和blue波段组合更倾向于优先聚集同一地物，例如建筑区域，林地区域等。

超像素级分割是一种语义分割，是计算机视觉的基本方法，可以更加精准的执行地物分割、探测和分类等深度学习任务；这一方法也同样为景观、生态专业探索地物变化和地物之间的关系提供一新的策略。[scikit-image](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#sphx-glr-auto-examples-segmentation-plot-segmentations-py)提供了四种分割的算法。在下述实验中计算了Felzenszwalb和Quickshift两种方法。Felzenszwalb方法在分割图像时，虽然逐级增加scale参数大小，但是分割的图像并不是上级区域覆盖下级区域；而Quickshift方法，则基本是逐级覆盖的，因此选用Quickshift方法，指定逐级增加的kernel_size参数，获取不同深度分割的结果。通过计算逐级覆盖的分割类别频数统计，及方差等指数，试图找到研究区域不同深度分割下区域间的关联，以及区域的差异性程度。

> 参考文献
1. Efficient graph-based image segmentation, Felzenszwalb, P.F. and Huttenlocher, D.P. International Journal of Computer Vision, 2004
2. Robert E.Ricklefs著. 孙儒泳等译.生态学/The economy of nature[M].高度教育出版社.北京.2004.7 第5版. ------非常值得推荐的读物(教材)，图文并茂

* 01 -  读取所下载的sentinel-2影像的元文件，获取影像波段路径


```python
MTD_MSIL2A_fn=r'D:\RSi\S2B_MSIL2A_20200709T163839_N0214_R126_T16TDM_20200709T211044.SAFE\MTD_MSIL2A.xml'
band_fns_list,band_fns_dict=Sentinel2_bandFNs(MTD_MSIL2A_fn)
```

    GENERATION_TIME:2020-07-09T21:10:44.000000Z
    PRODUCT_TYPE:S2MSI2A
    PROCESSING_LEVEL:Level-2A
    MTD_MSIL2A.xml 文件父结构:
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}General_Info - {}
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Geometric_Info - {}
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Auxiliary_Data_Info - {}
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Quality_Indicators_Info - {}
    __________________________________________________
    获取sentinel-2波段文件路径:
     {'B02_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_B02_10m.jp2', 'B03_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_B03_10m.jp2', 'B04_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_B04_10m.jp2', 'B08_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_B08_10m.jp2', 'TCI_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_TCI_10m.jp2', 'AOT_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_AOT_10m.jp2', 'WVP_10m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R10m/T16TDM_20200709T163839_WVP_10m.jp2', 'B02_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B02_20m.jp2', 'B03_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B03_20m.jp2', 'B04_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B04_20m.jp2', 'B05_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B05_20m.jp2', 'B06_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B06_20m.jp2', 'B07_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B07_20m.jp2', 'B8A_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B8A_20m.jp2', 'B11_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B11_20m.jp2', 'B12_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_B12_20m.jp2', 'TCI_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_TCI_20m.jp2', 'AOT_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_AOT_20m.jp2', 'WVP_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_WVP_20m.jp2', 'SCL_20m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R20m/T16TDM_20200709T163839_SCL_20m.jp2', 'B01_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B01_60m.jp2', 'B02_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B02_60m.jp2', 'B03_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B03_60m.jp2', 'B04_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B04_60m.jp2', 'B05_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B05_60m.jp2', 'B06_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B06_60m.jp2', 'B07_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B07_60m.jp2', 'B8A_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B8A_60m.jp2', 'B09_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B09_60m.jp2', 'B11_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B11_60m.jp2', 'B12_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_B12_60m.jp2', 'TCI_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_TCI_60m.jp2', 'AOT_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_AOT_60m.jp2', 'WVP_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_WVP_60m.jp2', 'SCL_60m': 'GRANULE/L2A_T16TDM_A017455_20200709T164859/IMG_DATA/R60m/T16TDM_20200709T163839_SCL_60m.jp2'}
    

* 02 - 裁切到研究区域，裁切边界由ＱGIS绘制。


```python
import util
import os
raster_fp=[os.path.join(r"D:\RSi\S2B_MSIL2A_20200709T163839_N0214_R126_T16TDM_20200709T211044.SAFE",band_fns_dict[k]) for k in band_fns_dict.keys() if k.split("_")[-1]=="20m"]
clip_boundary_fp=r'.\data\geoData\superPixel_boundary.shp'
save_path=r'D:\RSi\crop_20'
util.raster_clip(raster_fp,clip_boundary_fp,save_path)
```

    finished clipping.
    




    ['D:\\RSi\\crop_20\\T16TDM_20200709T163839_B02_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_B03_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_B04_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_B05_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_B06_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_B07_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_B8A_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_B11_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_B12_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_TCI_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_AOT_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_WVP_20m_crop.jp2',
     'D:\\RSi\\crop_20\\T16TDM_20200709T163839_SCL_20m_crop.jp2']



* 03 - 读取裁切后的影像，显示查看。实验中仅分析了red, blue和green波段组合。可以再深入分析red edge波段对植物的分割,以及计算NDVI，NDWI,NDBI，及反演地表温度来进一步研究不同信息数据分割结果。


```python
import glob
import earthpy.spatial as es  # conda install -c conda-forge geopandas ;pip install earthpy;conda install -c conda-forge earthpy 
import earthpy.plot as ep
import matplotlib.pyplot as plt

save_path=r'D:\RSi\crop_20'
croppedImgs_fns=glob.glob(save_path+"/*.jp2")
croppedBands_fnsDict={f.split('_')[-3]+'_'+f.split('_')[-2]:f for f in croppedImgs_fns}

bands_selection_=['B02_20m', 'B03_20m', 'B04_20m','B05_20m', 'B06_20m', 'B07_20m', 'B8A_20m', 'B11_20m', 'B12_20m']  #, 'TCI_20m', 'AOT_20m', 'WVP_20m', 'SCL_20m'
cropped_stack_bands=[croppedBands_fnsDict[b] for b in bands_selection_]

cropped_array_stack,_=es.stack(cropped_stack_bands)
ep.plot_bands(cropped_array_stack,title=bands_selection_,cols=cropped_array_stack.shape[0],cbar=True,figsize=(20,10))
plt.show()
```


    
<a href=""><img src="./imgs/25_13.png" height="auto" width="auto" title="caDesign"></a>
    



```python
import matplotlib.pyplot as plt
import numpy as np

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import pickle

img=cropped_array_stack[[2,1,0]]
img=img.transpose(1,2,0)
```

* 04 - Felzenszwalb 超像素级分割方法


```python
def superpixel_segmentation_Felzenszwalb(img,scale_list,sigma=0.5, min_size=50):
    import numpy as np
    from skimage.segmentation import felzenszwalb
    from tqdm import tqdm # conda install -c conda-forge tqdm ;conda install -c conda-forge ipywidgets
    '''
    function - 超像素分割，skimage库felzenszwalb方法。给定scale参数列表，批量计算
    '''
    segments=[felzenszwalb(img, scale=s, sigma=sigma, min_size=min_size) for s in tqdm(scale_list)]
    return np.stack(segments)
    
scale_list=[1,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100] 
segs=superpixel_segmentation_Felzenszwalb(img,scale_list)

with open('./results/segs_superpixel.pkl','wb') as f:
    pickle.dump(segs,f)
```

    100%|██████████| 16/16 [01:00<00:00,  3.77s/it]
    

* 05 - 显示分割图像，分割边界。


```python
import math
from skimage import exposure

scale_list=[1,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
with open('./results/segs_superpixel.pkl','rb') as f:
    segs=pickle.load(f)

p2, p98=np.percentile(img, (2,98))
img_=exposure.rescale_intensity(img, in_range=(p2, p98)) / 65535

def markBoundaries_layoutShow(segs_array,img,columns,titles,prefix,figsize=(15,10)):
    import math,os
    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage.segmentation import mark_boundaries
    '''
    function - 给定包含多个图像分割的一个数组，排布显示分割图像边界。

    Paras:
    segs_array - 多个图像分割数组
    img - 底图
    columns - 列数
    titles - 子图标题
    figsize - 图表大小
    '''       
    rows=math.ceil(segs_array.shape[0]/columns)
    fig,axes=plt.subplots(rows,columns,sharex=True,sharey=True,figsize=figsize)   #布局多个子图，每个子图显示一幅图像
    ax=axes.flatten()  #降至1维，便于循环操作子图
    for i in range(segs_array.shape[0]):
        ax[i].imshow(mark_boundaries(img, segs_array[i]))  #显示图像
        ax[i].set_title("{}={}".format(prefix,titles[i]))
    invisible_num=rows*columns-len(segs_array)
    if invisible_num>0:
        for i in range(invisible_num):
            ax.flat[-(i+1)].set_visible(False)
    fig.tight_layout() #自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("segs show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()
    
columns=6   
markBoundaries_layoutShow(segs,img_,columns,scale_list,'scale',figsize=(30,20))
```


    
<a href=""><img src="./imgs/25_14.png" height="auto" width="auto" title="caDesign"></a>
    


* 06 - Quickshift 超像素级分割方法，及显示。


```python
def superpixel_segmentation_quickshift(img,kernel_sizes, ratio=0.5):
    import numpy as np
    from skimage.segmentation import quickshift
    from tqdm import tqdm # conda install -c conda-forge tqdm ;conda install -c conda-forge ipywidgets
    '''
    function - 超像素分割，skimage库quickshift方法。给定kernel_size参数列表，批量计算
    '''
    segments=[quickshift(img, kernel_size=k,ratio=ratio) for k in tqdm(kernel_sizes)]
    return np.stack(segments)
    
kernel_sizes=[3,5,7,9,11,13,15,17,19,21] 
segs_quickshift=superpixel_segmentation_quickshift(img,kernel_sizes)

with open('./results/segs_superpixel_quickshift.pkl','wb') as f:
    pickle.dump(segs_quickshift,f)
```
       
    100%|██████████| 10/10 [29:00<00:00, 174.05s/it][A[A[A
    


```python
with open('./results/segs_superpixel_quickshift.pkl','rb') as f:
    segs_quickshift=pickle.load(f)

kernel_sizes=[3,5,7,9,11,13,15,17,19,21] 
columns=5
markBoundaries_layoutShow(segs_quickshift,img_,columns,kernel_sizes,'kernel_size',figsize=(30,10))
```


    
<a href=""><img src="./imgs/25_15.png" height="auto" width="auto" title="caDesign"></a>
    


* 07 - 显示分割掩码。进一步查看分割区域的变化。


```python
def segMasks_layoutShow(segs_array,columns,titles,prefix,cmap='prism',figsize=(20,10)):
    import math,os
    import matplotlib.pyplot as plt
    from PIL import Image
    '''
    function - 给定包含多个图像分割的一个数组，排布显示分割图像掩码。

    Paras:
    segs_array - 多个图像分割数组
    columns - 列数
    titles - 子图标题
    figsize - 图表大小
    '''       
    rows=math.ceil(segs_array.shape[0]/columns)
    fig,axes=plt.subplots(rows,columns,sharex=True,sharey=True,figsize=figsize)   #布局多个子图，每个子图显示一幅图像
    ax=axes.flatten()  #降至1维，便于循环操作子图
    for i in range(segs_array.shape[0]):
        ax[i].imshow(segs_array[i],cmap=cmap)  #显示图像
        ax[i].set_title("{}={}".format(prefix,titles[i]))
    invisible_num=rows*columns-len(segs_array)
    if invisible_num>0:
        for i in range(invisible_num):
            ax.flat[-(i+1)].set_visible(False)
    fig.tight_layout() #自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("segs show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()
    
columns=5
segMasks_layoutShow(segs_quickshift,columns,kernel_sizes,'kernel_size')
```


    
<a href=""><img src="./imgs/25_16.png" height="auto" width="auto" title="caDesign"></a>
    


* 08 - 多尺度超像素级分割结果叠合频数统计。包括各个层级与其之后所有层级间的计算。


```python
def multiSegs_stackStatistics(segs,save_fp):
    from scipy.ndimage import label
    from tqdm import tqdm
    '''
    function - 多尺度超像素级分割结果叠合频数统计
    '''
    segs=list(reversed(segs))
    stack_statistics={}
    for i in tqdm(range(len(segs)-1)):
        labels=np.unique(segs[i])
        coords=[np.column_stack(np.where(segs[i]==k)) for k in labels]
        i_j={}
        for j in range(i+1,len(segs)):
            j_k={}
            for k in range(len(coords)):
                covered_elements=[segs[j][x,y] for x,y in zip(*coords[k].T)]
                freq=list(zip(np.unique(covered_elements, return_counts=True)))
                j_k[k]=freq
            i_j[j]=j_k
            
        stack_statistics[i]=i_j
    with open(save_fp,'wb') as f:
        pickle.dump(stack_statistics,f)
    
    return stack_statistics
    
stack_statistics=multiSegs_stackStatistics(segs_quickshift,'./results/multiSegs_stackStatistics.pkl')    
```

    100%|██████████| 9/9 [01:05<00:00,  7.24s/it]
    

* 09 - 读取保存的分割层级叠合频数统计文件，提取卷积核即kernel_size最大的层级与之后所有层级的频数统计，转换为DataFrame数据格式。


```python
from tqdm import tqdm
import pickle

with open('./results/multiSegs_stackStatistics.pkl','rb') as f:
    stack_statistics=pickle.load(f)
segsOverlay_0_num={k:[stack_statistics[0][k][i][0][0].shape[0] for i in stack_statistics[0][k].keys()] for k in tqdm(stack_statistics[0].keys())}
```

    100%|██████████| 9/9 [00:00<00:00, 4510.00it/s]
    


```python
import pandas as pd
segsOverlay_0_num_df=pd.DataFrame.from_dict(segsOverlay_0_num)
segsOverlay_0_num_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>8</td>
      <td>10</td>
      <td>12</td>
      <td>12</td>
      <td>16</td>
      <td>17</td>
      <td>20</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>7</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>8</td>
      <td>12</td>
      <td>10</td>
      <td>11</td>
      <td>15</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>434</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>15</td>
      <td>19</td>
    </tr>
    <tr>
      <th>435</th>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>436</th>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>6</td>
      <td>10</td>
    </tr>
    <tr>
      <th>437</th>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>9</td>
      <td>8</td>
      <td>9</td>
      <td>15</td>
    </tr>
    <tr>
      <th>438</th>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>8</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
<p>439 rows × 9 columns</p>
</div>



* 10 - 显示所有对应深度层级的频数变化。


```python
import plotly.express as px
x=list(segsOverlay_0_num_df.index)
y=list(segsOverlay_0_num_df.columns)
fig = px.scatter(segsOverlay_0_num_df, x=x, y=y,
              #hover_data=[],
              title='id_info_df'
             )
fig.show()
```

<a href=""><img src="./imgs/25_17.png" height="auto" width="auto" title="caDesign"></a>

* 11 - 计算对应所有层级，父级分割（kernel_size最大的层级）每一分割类，在各个深度层级上对应覆盖分割类数量的方差统计。可以分析父级每一个分割区域内深度层级下破碎（父级分割区域内子层分割的种类数量）的变化情况，如果值越大，往往父级分割区域内的'斑块'破碎化程度比较高，即区域内具有明显的异质性(差异性)；如果方差值越小，则说明父级分割区域内‘斑块’属性基本近似，区域同质性。

将计算结果叠合到分割图上，以颜色显示方差变化，方便对应地理空间位置，并观察邻域间的情况。


```python
import numpy as np

var=segsOverlay_0_num_df.var(axis=1)
var_dict=var.to_dict()

seg_old=np.copy(segs_quickshift[-1])
seg_new=np.copy(seg_old).astype(float)
for old,new in var_dict.items():
    seg_new[seg_old==old]=new
```


```python
from skimage.measure import regionprops
regions=regionprops(segs_quickshift[-1])
seg_centroids={}
for props in regions:
    seg_centroids[props.label]=props.centroid

x,y=zip(*seg_centroids.values())
labels=seg_centroids.keys()
```


```python
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage import exposure

p2, p98=np.percentile(img, (2,98))
img_=exposure.rescale_intensity(img, in_range=(p2, p98)) / 65535

fig,ax=plt.subplots(1,1,frameon=False,figsize=(15,15)) #plt.rcParams["figure.figsize"] = (10,10)
im1=ax.imshow(mark_boundaries(img_, segs_quickshift[-1]))
im2=ax.imshow(seg_new,cmap='terrain',alpha=.35)

for k,coordi in seg_centroids.items():
    label=ax.text(x=coordi[1] ,y=coordi[0], s=k,ha='center', va='center',color='white')

axins = inset_axes(ax,
                   width="5%",  # width = 5% of parent_bbox width
                   height="50%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )
fig.colorbar(im2, cax=axins)
plt.show()
```


    
<a href=""><img src="./imgs/25_18.png" height="auto" width="auto" title="caDesign"></a>
    


### 1.4 要点
#### 1.4.1 数据处理技术

* Sentinel-2 遥感影像处理

* VGG16卷积神经网络与SegNet的关系

* ImageNet数据集

* SegNet遥感影像语义分割/解译

* ISPRS dataset数据集

*  @classmethod 类方法的装饰器

* 超像素级分割(superpixels-segmentation)

#### 1.4.2 新建立的函数

* function - 将经纬度坐标转换为指定zoom level缩放级别下，金子塔中瓦片的坐标/ code migrated. `deg2num(lat_deg, lon_deg, zoom)`

*  function - 根据获取的地图边界坐标[左下角精度，左下角维度，右上角精度，右上角维度]计算中心点坐标 /code migrated. `centroid(bounds)`

* function - Sentinel-2波段合成显示。需要deg2num(lat_deg, lon_deg, zoom)和centroid(bounds)函数. `Sentinel2_bandsComposite_show(RGB_bands,zoom=10,tilesize=512,figsize=(10,10))`

* funciton - 获取sentinel-2波段文件路径，和打印主要信息. `Sentinel2_bandFNs(MTD_MSIL2A_fn)`

* 数值标签转换为RGB颜色标签 Numeric labels to RGB-color encoding /code migrated. `convert_to_color(arr_2d, palette=palette)`

* RGB颜色标签转换为数值标签（灰度图） RGB-color encoding to grayscale labels /code migrated. ` convert_from_color(arr_3d, palette=invert_palette)`

* 给定窗口大小，随机提取部分图像 Extract of 2D random patch of shape window_shape in the image   /code migrated. `get_random_pos(img, window_shape`

* 定义损失函数——2D版交叉熵损失 2D version of the cross entropy loss  /code migrated. `CrossEntropy2d(input, target, weight=None, size_average=True)`

* 定义预测精度 /code migrated. `accuracy(input, target)`

* 给定步幅，窗口形状，滑动过整幅图像，迭代计算窗口所在图像x,y位置值，返回每一切分图像(patch)的x,y坐标值和高宽大小，即yield返回值。参数step可以控制切分窗口叠合的程度 Slide a window_shape window across the image with a stride of step /code migrated. `sliding_window(top, step=10, window_size=(20,20))`

* 计算图像滑动给定窗口大小的数量 Count the number of windows in an image /code migrated. `count_sliding_window(top, step=10, window_size=(20,20))`

* Browse an iterator by chunk of n elements /code migrated.  `grouper(n, iterable)`

* 预测值度量'  /code migrated `metrics(predictions, gts, label_values=LABELS)`

* SegNet测试函数  /code migrated. `test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)`

* SegNet训练函数 /code migrated. `train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch = 5)`

* function - 超像素分割，skimage库felzenszwalb方法。给定scale参数列表，批量计算. `superpixel_segmentation_Felzenszwalb(img,scale_list,sigma=0.5, min_size=50)`

* function - 给定包含多个图像分割的一个数组，排布显示分割图像边界。`markBoundaries_layoutShow(segs_array,img,columns,titles,prefix,figsize=(15,10))`

* function - 超像素分割，skimage库quickshift方法。给定kernel_size参数列表，批量计算. `superpixel_segmentation_quickshift(img,kernel_sizes, ratio=0.5)`

* function - 给定包含多个图像分割的一个数组，排布显示分割图像掩码。`segMasks_layoutShow(segs_array,columns,titles,prefix,cmap='prism',figsize=(20,10))`

* function - 多尺度超像素级分割结果叠合频数统计. `multiSegs_stackStatistics(segs,save_fp)`

#### 1.4.3 所调用的库


```python
import rio_tiler
from rio_tiler import main
from rasterio.plot import show

import earthpy.spatial as es
import earthpy.plot as ep
import geopandas as gpd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.express as px
import math,os
import numpy as np
import glob
from tqdm import tqdm_notebook as tqdm
import random
import itertools
import pickle
from scipy.ndimage import label

from skimage import exposure
from skimage import io
from skimage.color import rgb2gray
from skimage.measure import regionprops

import xml.etree.ElementTree as ET
from sklearn import cluster
from sklearn.metrics import confusion_matrix
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-90-5a42447cbbc7> in <module>
    ----> 1 import rio_tiler
          2 from rio_tiler import main
          3 from rasterio.plot import show
          4 
          5 import earthpy.spatial as es
    

    ModuleNotFoundError: No module named 'rio_tiler'


#### 1.4.4 参考文献
1. [Tiles à la Google Maps](https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/)
2. Karen Simonyan∗ & Andrew Zisserman+.[VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/abs/1409.1556)[J].Published as a conference paper at ICLR 2015.arXiv:1409.1556v6[cs.CV] 10 Apr 2015
3. [VGG-NETS](https://pytorch.org/hub/pytorch_vision_vgg/)
4. Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, Senior Member, IEEE. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation[J].arXiv:1511.00561v3 [cs.CV] 10 Oct 2016
5. [Deep learning for Earth Observation](https://github.com/nshaud/DeepNetsForEO)
6. Efficient graph-based image segmentation, Felzenszwalb, P.F. and Huttenlocher, D.P. International Journal of Computer Vision, 2004
7. Robert E.Ricklefs著. 孙儒泳等译.生态学/The economy of nature[M].高度教育出版社.北京.2004.7 第5版. ------非常值得推荐的读物(教材)，图文并茂
