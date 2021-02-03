# 代码工具索引
> 本书的目的不仅在于一般意义上的教材，以及专业上相关实验的阐述，同时，使大量实验所编写代码片段形成一个个可以方便迁移使用的工具，形成工具包。

## 1. 单个分类POI数据爬取与地理空间点地图
1. function-百度地图开放平台POI数据爬取，`baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False);`
2. function-转换.csv格式的POI数据为pandas的DataFrame,`csv2df(poi_fn_csv)`

## 2. 多个分类POI数据爬取与描述性统计
3. function-百度地图开放平台POI数据批量爬取，`baiduPOI_batchCrawler(poi_config_para)`。需要调用单个分类POI爬取函数`baiduPOI_dataCrawler(query_dic,4.bound_coordinate,partition,page_num_range,poi_fn_list=False)`
4. funciton-以所在文件夹路径为键，值为包含该文件夹下所有文件名的列表。文件类型可以自行定义 `filePath_extraction(dirpath,fileType)`
5. funciton-.csv格式POI数据批量转换为GeoDataFrame，`poi_csv2GeoDF_batch(poi_paths,fields_extraction,save_path)`。需要调用转换.csv格式的POI数据为pandas的DataFrame函数`csv2df(poi_fn_csv)`
6. funciton-使用plotly以表格形式显示DataFrame格式数据,`ployly_table(df,column_extraction)`
7. function-频数分布计算，`frequency_bins(df,bins)`

## 3. 正态分布与概率密度函数，异常值处理
8. funciton-数据集z-score概率密度函数分布曲线(即观察值/实验值 observed/empirical data)与标准正态分布(即理论值 theoretical set)比较，`comparisonOFdistribution(df,field,bins=100)`
9. function-判断异常值，`is_outlier(data,threshold=3.5)`

## 4. OpenStreetMap（OSM）数据处理
10. function-转换shape的polygon为osmium的polygon数据格式（.txt），用于.osm地图数据的裁切，`shpPolygon2OsmosisTxt(shape_polygon_fp,osmosis_txt_fp)`
11. class-通过继承osmium类 class osmium.SimpleHandler读取.osm数据, `osmHandler(osm.SimpleHandler)`
12. function-根据条件逐个保存读取的osm数据（node, way and area）,`save_osm(osm_handler,osm_type,save_path=r"./data/",fileType="GPKG")`
13. function-计算当前时间，`start_time()`
14. function-计算持续时间, `duration(start_time)`

## 5.核密度估计与地理空间点密度分布
15. function - 将.shp格式的点数据转换为.tif栅格(raster)，`pts2raster(pts_shp,raster_path,cellSize,field_name=False)`
16. function - 将GeoDaraFrame格式的点数据转换为栅格数据， `pts_geoDF2raster(pts_geoDF,raster_path,cellSize,scale)`

## 6.标准误，中心极限定理，t分布，统计显著性，效应量，置信区间；公共健康数据的地理空间分布与相关性分析
17. function - 在Jupyter中打印DataFrame格式数据为HTML, `print_html(df,row_numbers=5`
18. function - 生成颜色列表或者字典， `generate_colors()`

## 7.简单回归，多元回归
19. function - 在matplotlib的子图中绘制连接线，`demo_con_style(a_coordi,b_coordi,ax,connectionstyle)`
20. function - 回归方程的判定系数， `coefficient_of_determination(observed_vals,predicted_vals)`
21. function - 简单线性回归方程-回归显著性检验（回归系数检验），` ANOVA(observed_vals,predicted_vals,df_reg,df_res)`
22. function - 简单线性回归置信区间估计，以及预测区间， `confidenceInterval_estimator_LR(x,sample_num,X,y,model,confidence=0.05)`
23. function - DataFrame数据格式，成组计算pearsonr相关系数，`correlationAnalysis_multivarialbe(df)`
24. function - 回归方程的修正自由度的判定系数，` coefficient_of_determination_correction(observed_vals,predicted_vals,independent_variable_n)`
25. function - 多元线性回归方程-回归显著性检验（回归系数检验），全部回归系数的总体检验，以及单个回归系数的检验， `ANOVA_multivarialbe(observed_vals,predicted_vals,independent_variable_n,a_i,X)`
26. function - 多元线性回归置信区间估计，以及预测区间，` confidenceInterval_estimator_LR_multivariable(X,y,model,confidence=0.05)`

## 8. 回归公共健康数据，与梯度下降法
27. function - 返回指定邻近数目的最近点坐标，`k_neighbors_entire(xy,k=3)`
28. function - 多项式回归degree次数选择，及正则化,` PolynomialFeatures_regularization(X,y,regularization='linear')`
29. 梯度下降法 - 定义模型，定义损失函数，定义梯度下降函数，定义训练模型函数

## 9. 线性代数基础的代码表述
30. funciton - 转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式，`vector_plot_3d(ax_3d,C,origin_vector,vector,color='r',label='vector',arrow_length_ratio=0.1)`
31. function - 给定向量，及对应系数，延向量绘制，`move_alongVectors(vector_list,coeffi_list,C,ax,)`
32. function - 将向量集合转换为向量矩阵，并计算简化的行阶梯形矩阵，`vector2matrix_rref(v_list,C)`

## 10. Landsat遥感影像处理，数字高程，主成成分分析
33. function - 按照文件名中的数字排序文件列表， `fp_sort(fp_list,str_pattern,prefix="")`
34. function - 读取landsat *_MTL.txt文件，提取需要的信息，`LandsatMTL_info(fp)`
35. funciton - 转换SymPy的vector及Matrix数据格式为matplotlib可以打印的数据格式，` vector_plot_2d(ax_2d,C,origin_vector,vector,color='r',label='vector',width=0.022)`
36. function - 给定圆心，半径，划分份数，计算所有直径的首尾坐标， `circle_lines(center,radius,division)`
37. function - 计算二维点到直线上的投影，`point_Proj2Line(line_endpts,point)`
38. function - 计算NDVI指数，` NDVI(RED_band,NIR_band)`

## 11.遥感影像解译（基于NDVI），建立采样工具（GUI_tkinter），混淆矩阵
39. function - 给定裁切边界，批量裁切栅格数据，`raster_clip(raster_fp,clip_boundary_fp,save_path)`
40. function - 指定波段，同时显示多个遥感影像，`w_180310_array(img_stack_list,band_num)`
41. function - 将变量名转换为字符串， `variable_name(var)`
42. function - 拉伸图像 `contract stretching，image_exposure(img_bands,percentile=(2,98))`
43. function - 将数据按照给定的百分数划分，并给定固定的值,整数值或RGB色彩值，`data_division(data,division,right=True)`
44. function - 多个栅格数据，给定百分比，变化观察，`percentile_slider(season_dic)`
45. 基于tkinter，开发交互式GUI采样工具 ` class AutoScrollbar(ttk.Scrollbar`   `class CanvasImage(ttk.Frame)`

## 12. 点云数据（激光雷达）处理——分类数据，DSM，建筑高度提取，插值
46. function - 转换单个.las点云数据为分类栅格数据，和DSM栅格数据等，`las_info_extraction(las_fp,json_combo)`
47. function - 显示由.las文件生成的分类栅格文件，并显示图例，`las_classification_plotWithLegend(las_fp)`
48. function - 批量转换.las点云数据为DSM和分类栅格，`las_info_extraction_combo(las_dirPath,json_combo_)`
49. function - 合并多个栅格为一个，`raster_mosaic(dir_path,out_fp,)`
50. function - 迁移rasterio提供的定义数组最小数据类型的函数，`get_minimum_int_dtype(values)`
51. function - 获取给定栅格的投影坐标-crs，`get_crs_raster(raster_fp)`
52. function - 转换栅格投影，`raster_reprojection(raster_fp,dst_crs,save_path)`
53. function - 根据给定栅格的范围，裁切.shp格式数据，并定义投影同给定栅格，`clip_shp_withRasterExtent(vector_shp_fp,reference_raster_fp,save_path)`
54. function - 使用rasterio.fill的插值方法，补全缺失的数据，`rasters_interpolation(raster_path,save_path,max_search_distance=400,smoothing_iteration=0)`
55. function - 使用earthpy库显示遥感影像（一个波段），`raster_show(raster_fp,title='raster',vmin_vmax=[0.25,0.95],cmap="turbo")`

## 13. 卷积，SIR传播模型，成本栅格与物种散布，SIR空间传播模型
56. class - 一维卷积动画解析，可以自定义系统函数和信号函数，`class dim1_convolution_SubplotAnimation(animation.TimedAnimation)`
57. function - 定义系统响应函数.类型-1， `G_T_type_1()`
58. function - 定义输入信号函数，类型-1， `F_T_type_1(timing)`
59. function - 定义系统响应函数.类型-2，`G_T_type_2()`
60. function - 定义输入信号函数，类型-2， `F_T_type_2(timing)`
61. function - 读取MatLab的图表数据，类型-A， `read_MatLabFig_type_A(matLabFig_fp,plot=True)`
62. function - 应用一维卷积，根据跳变点分割数据， `curve_segmentation_1DConvolution(data,threshold=1)`
63. function - 根据索引，分割列表，`lst_index_split(lst, args)`
64. function - 展平列表函数， `flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]`
65. function - 嵌套列表，子列表前后插值，`nestedlst_insert(nestedlst)`
66. function - 使用matplotlib提供的方法随机返回浮点型RGB， `uniqueish_color()`
67. function - 定义SIR传播模型微分方程， `SIR_deriv(y,t,N,beta,gamma,plot=False)`
68. function - 显示图像以及颜色R值，或G,B值，`img_struc_show(img_fp,val='R',figsize=(7,7))`
69. class - 定义基于SIR模型的二维卷积扩散，`class convolution_diffusion_img`
70. function - 读入.gif，并动态显示，`animated_gif_show(gif_fp,figsize=(8,8))`
71. fuction - 降采样二维数组，根据每一block内值得频数最大值，即最多出现得值为每一block的采样值，`downsampling_blockFreqency(array_2d,blocksize=[10,10])`
72. class - SIR的空间传播模型， `SIR_spatialPropagating`


## 14. 微积分基础的代码表述
73. function - 在matplotlib的子图中绘制多个连接线，`demo_con_style_multiple(a_coordi,b_coordi,ax,connectionstyle)`

## 15.聚类与城市色彩
74. function - 提取.kml文件中的坐标信息, `kml_coordiExtraction(kml_pathDict)`.
75. function - 显示一个文件夹下所有图片，便于查看, `imgs_layoutShow(imgs_root,imgsFn_lst,columns,scale,figsize=(15,10))`
76. function - 提取数码照片的属性信息和拍摄数据，即可交换图像文件格式（Exchangeable image file format，Exif）,`img_exif_info(img_fp,printing=True)`
77. function - 将像素RGB颜色投射到色彩空间中，直观感受图像颜色的分布, `imgs_colorSpace(imgs_root,imgsFn_lst,columns,scale,figsize=(15,10))`.
78. class - 定义K-Means算法, `class K_Means`.
79. function - 读取与压缩图像，返回2，3维度数组, `img_rescale(img_path,scale)`.
80. function - 聚类的方法提取图像主题色，并打印图像、聚类预测类的二维显示和主题色带, `img_theme_color(imgs_root,imgsFn_lst,columns,scale,)`.
81. function - 显示所有图像主题色，获取总体印象, `themeColor_impression(theme_color)`.
82. function - 保存文件,将文件存储为json数据格式, `save_as_json(array,save_root,fn)`.

## 16.城市生活圈，DBSCAN连续距离聚类，卡方分布及独立性检验，协方差估计，信息熵与均衡度
83. function - 提取分析所需数据，并转换为skleran的bunch存储方式，统一格式，方便读取. `poi_json2sklearn_bunch(fps,save_path)`
84. class - 使用DBSCAN算法实现连续距离聚类和空间结构分析. `poi_spatial_distribution_structure` 

包括：

85. function - 打印数组频数. `frequency_array(slef,array)`
86. function - 单次聚类. `clustering_DBSCAN(self,eps_single)`
87. function - 根据聚类距离列表批量处理（聚类）. `clustering_batch_computing(self)`
88. function - 保存聚类结果于.shp文件中,及打印一组预览. `poi2shp(self)`
89. function - 卡方独立性检验，分析POI一级行业分类类标与聚类簇的相关性. `poi_chi_2Test(self)`
90. unction - POI一级行业分类的业态结构. `POI_structure(self)`

---

91. function - 绘制折线图，及其拐点. `kneed_lineGraph(x,y)`
92. class - 图片合并. `class combine_pics`

包括：

93. function - 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表，按字母-数字顺序。 ·file_sorting(self·
94. function - 读取与压缩图片. `read_compress_imgs(self,imgs_fp)`
95. function - 建立文件夹，用于存储拼合的图片. `make_dir(self)`
96. function - 拼合图片. `imgs_combination(self,imgs)`
97. function - 计算POI的均衡都. `equilibriumDegree_hierarchy(poi_clustering,poi_columns,poi_label)`

## 17. openCV-计算机视觉，特征提取，尺度空间（scale space），动态街景视觉感知
98. function - 应用OpenCV计算高斯模糊，并给定滑动条调节参数, `Gaussion_blur(img_fp)`
99. function - SIFT(scale invariant feature transform) 尺度不变特征变换)特征点检测, `SIFT_detection(img_fp,save=False)`
100. function - 使用Star特征检测器提取图像特征, `STAR_detection(img_fp,save=False)`
101. function - OpenCV 根据图像特征匹配图像。迁移官网的三种方法，1-ORB描述子蛮力匹配　Brute-Force Matching with ORB Descriptors；2－使用SIFT描述子和比率检测蛮力匹配 Brute-Force Matching with SIFT Descriptors and Ratio Test; 3-基于FLANN的匹配器 FLANN based Matcher, `feature_matching(img_1_fp,img_2_fp,index_params=None,method='FLANN')`
102. function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件，2-包含时间戳的.txt文件，`KITTI_info(KITTI_info_fp,timestamps_fp`
103.  function - 使用plotly的go.Scattermapbox方法，在地图上显示点及其连线，坐标为经纬度, `plotly_scatterMapbox(df,**kwargs)`
104. class - 应用Star提取图像关键点，结合SIFT获得描述子，根据特征匹配分析特征变化（视觉感知变化），即动态街景视觉感知, `dynamicStreetView_visualPerception`
105. class - 曲线（数据）平滑，与寻找曲线水平和纵向的斜率变化点, `movingAverage_inflection`
106. function - 计算图像匹配特征点几乎无关联的距离，即对特定位置视觉随距离远去而感知消失的距离, `vanishing_position_length(matches_num,coordi_df,epsg,**kwargs)`

## 18. [SQLite]数据库，[Flask] 构建实验用网络应用平台，逆向工程，视觉感知-基于图像的空间分类
107. function - pandas方法把DataFrame格式数据写入数据库（同时创建表）, `df2SQLite(engine,df,table,method='fail')`
108. function - 按字典的键，成对匹配，返回用于写入SQLite数据库的列表, `zip_dic_tableSQLite(dic,table_model)`
109. class - 返回指定路径下所有文件夹及其下文件的结构。代码未改动，迁移于'https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python', `DisplayablePath(object)`
110. function - 使用OpenCV的方法压缩保存图像  , `imgs_compression_cv(imgs_root,imwrite_root,imgsPath_fp,gap=1,png_compression=9,jpg_quality=100)`
111. function - 读取KITTI文件信息，1-包括经纬度，惯性导航系统信息等的.txt文件。只返回经纬度、海拔信息, `KITTI_info_gap(KITTI_info_fp,save_fp,gap=1)`
112. function - 将KITTI图像路径与经纬度信息对应起来，并存入SQLite数据库, `KITTI_info2sqlite(imgsPath_fp,info_fp,replace_path,db_fp,field,method='fail')`


## 19. 视觉词袋（BOW），决策树（Decision trees）->随机森林（Random forests），交叉验证 cross_val_score，视觉感知-图像分类_识别器，网络实验平台服务器端部署

113. function - 按照分类提取图像路径与规律, `load_training_data(imgs_df,classi_field,classi_list,path_field)`
114. class - 根据所有图像关键点描述子聚类建立图像视觉词袋，获取每一图像的特征（码本）映射的频数统计, `feature_builder_BOW`

包括：

115. function - 提取图像特征, `extract_features(self,img)`
116. function - 聚类所有图像的特征（描述子/SIFT），建立视觉词袋, `visual_BOW(self,des_all)`
117. function - 提取图像特征，返回所有图像关键点聚类视觉词袋, `get_visual_BOW(self,training_data)`
118. uction - 归一化数据, ` normalize(self,input_data)`
119. function - 使用聚类的视觉词袋构建图像特征（构造码本）, `construct_feature(self,img,kmeans)`
120. function - 返回每个图像的特征映射（码本映射）, `get_feature_map(self,training_data,kmeans)`

---

121. function - 根据指定的（多个）列，将分类转换为整数表示，区间为[0,分类数-1], ``df_multiColumns_LabelEncoder(df,columns=None)
122. function - 计算信息熵分量, `entropy_compomnent(numerator,denominator)`
123. function - 计算信息增量(IG), `IG(df_dummies)`
124. function - 使用决策树分类，并打印决策树流程图表。迁移于Sklearn的'Understanding the decision tree structure', https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py, `decisionTree_structure(X,y,criterion='entropy',cv=None,figsize=(6, 6))`
125. class - 用极端随机森林训练图像分类器, `ERF_trainer`
126. class - 图像识别器，基于图像分类模型，视觉词袋以及图像特征, `ImageTag_extractor`

## 20. 从解析解(analytical solution)-->到数值解(numerical solution) | 从机器学习[scikit-learn](https://scikit-learn.org/stable/)-->到深度学习[pytorch](https://PyTorch.org/)

127. function - sigmoid函数, `sigmoid(x)`
128. function - 定义总误差偏导, `partialD_E_total_prediction(true_value_,predicted_value_)`
129. function -定义激活函数偏导, `partialD_activation(x)`
130. fucntion - 定义加权和偏导, `partialD_weightedSUM(w_)`
131. function - 三层神经网络，各层2个神经元的示例，前向传播函数, `forward_func(input_list,weight_list,bias_list,output_list)`
132. class - 每一神经元的输入加权和、激活函数，以及输出计算, `NEURON`
133. class - 神经网络每一层的定义, `NEURAL_LAYER`
134. class - 构建多层感知机（神经网络）, ` NEURAL_NETWORK`
135. PyTorch自定义激活函数、模型、损失函数及梯度下降法, `sigmoid(X), net(X), loss(y_hat,y)`


## 21. 伯努利分布(Bernouli distribution)，似然函数，最大/极大似然估计(Maximum Likelihood Estimation, MLE)，逻辑回归(Logistic Regression, LR)二分类，SoftMax回归多分类

136. function - 按指定区间缩放/映射数组/Rescale an arrary linearly. `rescale_linear(array, new_min, new_max)`
137. class - 自定义逻辑回归(二元分类). `LogisticRegression:`

包括：

138. function - 根据一元一次函数构建二维离散型分类数据集，类标为1和0. `generate_dataset_linear(self,slop,intercept,num=100,multiple=10,magnitude=50)
139. function - 使用Sklearn提供的make_classification方法，直接构建离散型分类数据集. `make_classification_dataset(self,n_features=2,n_classes=2,n_samples=100,n_informative=2,n_redundant=0,n_clusters_per_class=1)`
140. function - 绘制逻辑回归结果. `plot(self,X,y,weights,figsize=(10,10))`
141. function - 逻辑回归函数_sigmoid函数. `sigmoid(self,x)`
142. function - 随机梯度下降（优化）. `stocGradAscent1(self,dataMatrix, classLabels, numIter=150)`
143. function - 代入梯度下降法训练的权重值于逻辑回归函数（sigmoid），预测样本特征对应的类标. `classifyVector(self,inX, weights)`
144. function - 测试训练模型. `test_accuracy(self,X_,y_,weights)`

---

145. function - 使用Sklearn提供的make_classification方法，直接构建离散型分类数据集;并打印查看. `make_classification_dataset_(n_features=2,n_classes=2,n_samples=100,n_informative=2,n_redundant=0,n_clusters_per_class=1,figsize=(10,10),plot=True)`
146. class - 自定义softmax回归多分类

包括：

147. function - 定义softmax函数. `softmax(self,Z)`
148. function - 定义线性函数. `linear_weights(self,X)`
149. function - 定义损失函数. `closs(self,y_one_hot,y_probs)`
150. function - numpy实现one-hot-code. `one_hot(self,y)`
151. function - 定义梯度下降法，权值更新. `gradient(self,X,y_one_hot,y_probs,lr,lambda_)`
152. function - 预测. `predict_test(self,X_,y_`
153. function - 更新打印损失曲线. `loss_curve(self,figsize=(8,5))`
154. function - 训练模型. `train(self,X,y,epochs,lr=0.1,lambda_=0.01)`

---

155. class - 自定义softmax回归多分类_PyTorch版. `softmax_multiClass_classification_UDF_pytorch`

包括：

156. function - 定义SoftMax函数. `SoftMax(self,Z)`
157. function - 定义模型，含线性模型输入，SoftMax回归输出. `net(self,X)`
158. function - 定义交叉熵损失函数. `cross_entropy(self,y_pred,y)`
159. function - 定义分类准确率，即正确预测数量与总预测数量之比. `accuracy(self,y_pred,y)`
160. funtion - 平均模型net在数据集data_iter上的准确率. `evaluate_accuracy(self,data_iter)`
161. funtion - 梯度下降. `sgd(self,lr)`
162. function - 训练模型. `train(self,train_iter,epochs,lr,test_iter=None)`

---

163. class - 将每批次样本X的形状转换为(batch_size,-1). `flattenLayer(nn.Module)`
164. funtion - 平均模型net在数据集data_iter上的准确率. `evaluate_accuracy(data_iter, net)`
165. funtion - 梯度下降，v1版. `sgd_v1(params,lr)`
166. function - 训练模型，v1版. `train_v1(net,train_iter, test_iter, loss, num_epochs,params=None, lr=None, optimizer=None,interval_print=100)`
167. function - 将Fashion-MNIST数据集，整型类标转换为名称. `fashionMNIST_label_num2text(labels_int)`
168. function - 打印显示Fashion-MNIST数据集图像. `fashionMNIST_show(imgs,labels,figsize=(12, 12))`


## 22. 卷积神经网络，可视化卷积层/卷积核,tensorboard,torchvision.models与VGG网络

169. class - PyTorch 卷积层，池化层输出尺寸(shape)计算，及根据输入，输出尺寸(shape)反推pad填充大小. `conv2d_output_size_calculation`

包括：

170. function - 如果num=2，则返回(2,2)；如果num=(2,2)，则返回(2,2). `num2tuple(self,num)`
171. funciton - 计算PyTorch的nn.Conv2d卷积方法的输出尺寸。以卷积核左上角对位图像左上角第一个像素值开始由左-->右，由上-->下卷积计算. `conv2d_output_shape(self,h_w, kernel_size=1, stride=1, pad=0, dilation=1)`
172. function - 以卷积核右下角对位图像左上角第一个像素值开始由左-->右，由上-->下卷积计算. `convtransp2d_output_shape(self,h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0)`
173. function - conv2d_output_shape 方法的逆，求填充pad. `conv2d_get_padding(self,h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1)`
174. function - convtransp2d_output_shape 方法的逆，求填充pad. `convtransp2d_get_padding(self,h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, out_pad=0)`
175. function - pooling池化层输出尺寸，同conv2d_output_shape. ` pooling_output_shape(self,h_w, kernel_size=1, stride=1, pad=0, dilation=1)`

---

176. fucntion - 一次性计算卷积输出尺寸. `conv2d_outputSize_A_oneTime(convs_params)`
177. function - 下载读取fashionMNIST数据集，并建立训练、测试可迭代数据集. `load_fashionMNIST(root,batchsize=4,num_workers=2,resize=None,n_mean=0.5,n_std=0.5)`
178. 自定义net_fashionMNIST(nn.Module)网络. `net_fashionMNIST(nn.Module)`
179. 定义显示图像函数（一个batch批次）. `matplotlib_imshow(img, one_channel=False)`
180. function - 用训练的网络预测给定的一组图像，并计算相应的概率. `images_to_probs(net, images)`
181. function - 用训练的网络预测给定的一组图像，并计算概率后，显示图像、预测值以及概率，和实际的标签. `plot_classes_preds(net, images, labels)`
182. class - 取回卷积神经网络中所有卷积层及其权重. `class conv_retriever`
183. function - 可视化卷积核 visualize the conv layer filters. `visualize_convFilter(conv_layer,model_weight,output_name,figsize=(10,10))`
184. function - 可视化所有卷积层（卷积结果）. `visualize_convLayer(imgs_batch,conv_layers,model_weights,num_show=6,figsize=(10,10))`
185. function - 自定义VGG. `vgg_block(num_convs,in_channels,out_channels)`;`vgg(conv_arch,fc_features,fc_hidden_units=4096)`
186. function - 查看可迭代数据形状. `dataiter_view(dataiter)`
187. function - 模型精度计算. `evaluate_accuracy_V2(data_iter, net, device=None)`
188. function - 训练模型，v2版. `train_v2(net, train_iter, test_iter,optimizer, device, num_epochs)`

## 23. 对象检测(Mask R-CNN)与人流量估算；图像分割（torchvision.models）与城市空间内容统计，及关联网络结构

189. class - PennFudanPed数据集，建立可迭代对象(迁移). `PennFudanDataset(torch.utils.data.Dataset)`
190. function - PennFudanPed数据集，行人对象检测模型精调（迁移）. `get_instance_segmentation_model(num_classes)`
191. function - 转换图像为tensor，可以包含多种变化（迁移）. `get_transform(train)`
192. function - fcn_resnet101模型，图像分割的类别给与颜色标识（迁移）. `decode_segmap_FCN_RESNET101(image, nc=21)`
193. function - 应用 torchvision.models.segmentation.fcn_resnet101预测图像，并打印显示分割预测结果（迁移，更新）. `segmentation_FCN_RESNET101_plot(net, path, show_orig=True, dev='cuda',img_resize=640,figsize=(20, 20))`
194. function - 应用 torchvision.models.segmentation.fcn_resnet101预测图像，并打印显示分割预测结果的动画. `segmentation_FCN_RESNET101_animation(net, paths,save_path='./animation.mp4' ,dev='cuda',img_resize=640,interval=150,figsize=(20, 20))`
195. function - 应用 torchvision.models.segmentation.fcn_resnet101预测图像，返回预测结果. `egmentation_FCN_RESNET101(net, path, show_orig=True, dev='cuda',img_resize=640)`
196. function - 计算列表的频数. `count_list_frequency(lst)`
197. function - 根据连续的图像分割数据，计算各个一对象（真实世界存在的物）与其它对象对应的数量，构建网络结构，分析相互关系. `objects_network_PascalVOC(seg_idxs,figsize=(15,15),layout='spring_layout',w_ratio=0.5)`

## 24. 开放神经网络交换-ONNX(Open Neural Network,Exchage)，netron，Cityscapes数据集，DUC(Dense Upsampling Convolution)图像分割与城市空间要素组成，时空量度，绿视率

198. class - 使用类的属性存储参数，并配置参数immutable与mutable互相转换的方法，迁移'https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/collections.py'.  `AttrDict(dict)`
199. function - DUC图像分割，及预测图像保存. `sementicSeg_DUC_pred(DUC_output_root,img_fps,preprocess,predict)`
200. function - 读取DUC语义分割结果保存的'KITTI_DUC_confi.pkl'和'KITTI_DUC_raw.pkl'文件，用于位置图像对象/语义分割类别频数统计. `DUC_pred_frequency_moment(KITTI_DUC_raw_fp,KITTI_DUC_confi_fp)`
201. function - 计算DataFrame每行的信息熵，用于df.apply(lambda). `entroy_df_row(row)`

## 25. Sentinel-2，(聚类)土地分类，VGG16，SegNet遥感影像语义分割/解译，超像素级分割(superpixels-segmentation)，高空分辨率特征尺度界定，及尺度的深度流动线/特征区域的延申


202. function - 将经纬度坐标转换为指定zoom level缩放级别下，金子塔中瓦片的坐标/ code migrated. `deg2num(lat_deg, lon_deg, zoom)`
203.  function - 根据获取的地图边界坐标[左下角精度，左下角维度，右上角精度，右上角维度]计算中心点坐标 /code migrated. `centroid(bounds)`
204. function - Sentinel-2波段合成显示。需要deg2num(lat_deg, lon_deg, zoom)和centroid(bounds)函数. `Sentinel2_bandsComposite_show(RGB_bands,zoom=10,tilesize=512,figsize=(10,10))`
205. funciton - 获取sentinel-2波段文件路径，和打印主要信息. `Sentinel2_bandFNs(MTD_MSIL2A_fn)`
206. 数值标签转换为RGB颜色标签 Numeric labels to RGB-color encoding /code migrated. `convert_to_color(arr_2d, palette=palette)`
207. RGB颜色标签转换为数值标签（灰度图） RGB-color encoding to grayscale labels /code migrated. ` convert_from_color(arr_3d, palette=invert_palette)`
208. 给定窗口大小，随机提取部分图像 Extract of 2D random patch of shape window_shape in the image   /code migrated. `get_random_pos(img, window_shape`
209. 定义损失函数——2D版交叉熵损失 2D version of the cross entropy loss  /code migrated. `CrossEntropy2d(input, target, weight=None, size_average=True)`
210. 定义预测精度 /code migrated. `accuracy(input, target)`
211. 给定步幅，窗口形状，滑动过整幅图像，迭代计算窗口所在图像x,y位置值，返回每一切分图像(patch)的x,y坐标值和高宽大小，即yield返回值。参数step可以控制切分窗口叠合的程度 Slide a window_shape window across the image with a stride of step /code migrated. `sliding_window(top, step=10, window_size=(20,20))`
212. 计算图像滑动给定窗口大小的数量 Count the number of windows in an image /code migrated. `count_sliding_window(top, step=10, window_size=(20,20))`
213. Browse an iterator by chunk of n elements /code migrated.  `grouper(n, iterable)`
214. 预测值度量'  /code migrated `metrics(predictions, gts, label_values=LABELS)`
215. SegNet测试函数  /code migrated. `test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)`
216. SegNet训练函数 /code migrated. `train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch = 5)`
217. function - 超像素分割，skimage库felzenszwalb方法。给定scale参数列表，批量计算. `superpixel_segmentation_Felzenszwalb(img,scale_list,sigma=0.5, min_size=50)`
218. function - 给定包含多个图像分割的一个数组，排布显示分割图像边界。`markBoundaries_layoutShow(segs_array,img,columns,titles,prefix,figsize=(15,10))`
219. function - 超像素分割，skimage库quickshift方法。给定kernel_size参数列表，批量计算. `superpixel_segmentation_quickshift(img,kernel_sizes, ratio=0.5)`
220. function - 给定包含多个图像分割的一个数组，排布显示分割图像掩码。`segMasks_layoutShow(segs_array,columns,titles,prefix,cmap='prism',figsize=(20,10))`
221. function - 多尺度超像素级分割结果叠合频数统计. `multiSegs_stackStatistics(segs,save_fp)`