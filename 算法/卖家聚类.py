import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as shc

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 导入数据集
products = pd.read_csv('products.csv')
o_items = pd.read_csv("order_items1.csv")
orders = pd.read_csv("orders.csv")
sellers = pd.read_csv("sellers.csv")
customers = pd.read_csv("customers.csv")


# 合并各个表并删除不用的信息
olist = orders.merge(o_items, on='order_id', how='left')
olist = olist.merge(products, on='product_id', how='outer')
olist = olist.merge(customers, on='customer_id', how='outer')
olist = olist.merge(sellers, on='seller_id', how='outer')

olist.drop(['customer_id',
            'payment_type', 'payment_installments',
            'customer_unique_id', 'customer_city',
            'seller_city', 'shipping_limit_date'], axis=1, inplace=True)

# dropna() 函数，可以将 olist 数据框中包含 NaN 值的行删除
olist = olist.dropna()
# 通过 info() 函数，可以打印出 olist 数据集的基本信息，包括数据集的类型、列名、非空值的数量等等。
olist.info()


olist['order_delivered_customer_date'] = pd.to_datetime(olist.order_delivered_customer_date)
olist['order_estimated_delivery_date'] = pd.to_datetime(olist.order_estimated_delivery_date)
# 计算「order_delivered_customer_date」和「order_estimated_delivery_date」之间的时间差,并将结果赋值给「delivery_delay」列,这个时间差以天为单位
olist['delivery_delay'] = olist.order_delivered_customer_date - olist.order_estimated_delivery_date

# astype()函数将「delivery_delay」列的数据类型转换为「timedelta64[D]」，以确保时间差以天为单位
olist['delivery_delay'] = pd.to_timedelta(olist.delivery_delay).dt.ceil('D')

# 删除用不上的列
olist_encoded = olist.drop(['customer_state', 'seller_state', 'product_category_name'], axis=1)

olist_encoded.tail(10)  # 显示olist_encoded中最后10行的数据

data = olist_encoded.groupby('seller_id').agg({'order_id': 'nunique',
                                               'product_id': 'nunique',
                                               'payment_value': 'mean',
                                               'review_score': 'mean',
                                               'product_name_lenght': 'mean',
                                               'product_description_lenght': 'mean',
                                               'product_photos_qty': 'mean',
                                               'delivery_delay': 'mean',

                                               })

scaler = MinMaxScaler()  # MinMaxScaler是一种常用的缩放器，它通过将数据缩放到指定的最小值和最大值之间，使得数据符合均匀分布

def Kmeansplots(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    X = data.copy()  # 将data的数据复制给变量X
    # 将数据集中指定数据类型的特征列名称存储在numerical_features变量中
    numerical_features = list(data.select_dtypes(include=['int64', 'float64',
                                                          'uint8']).columns)
    # 定义一个预处理器preprocessor,其中包含一个名为scaler的转换器，用于对数据集numerical_features进行特征缩放
    preprocessor = ColumnTransformer([
        ('scaler', scaler, numerical_features)])

    plt.figure(figsize=(24, 14))
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 2, 3)
    ax5 = plt.subplot(2, 2, 4)

    # 定义一个名为visual_grid1的列表，其中包含了三个元组,每个元组都表示一个可视化网格，用于展示聚类算法中不同指标（'distortion'(失真度)、'silhouette'(轮廓系数)、'calinski_harabasz'(聚类评估指标)）随聚类数变化的情况
    # KElbowVisualizer是一个用于展示聚类结果的可视化工具，可以显示不同指标（如失真度、轮廓系数和Calinski-Harabasz指数）随聚类数变化的情况
    visual_grid1 = [
        (Pipeline([("preprocessor", preprocessor), ("kelbowvisualizer"
                                                    , KElbowVisualizer(KMeans(), K=(4, 12), metric='distortion',
                                                                       ax=ax1))]), 'kelbowvisualizer'),
        # KMeans()表示使用KMeans算法作为聚类算法，K=(4, 12)表示聚类数的范围为4到12
        (Pipeline([("preprocessor", preprocessor), ("kelbowvisualizer"
                                                    , KElbowVisualizer(KMeans(), K=(4, 12), metric='silhouette',
                                                                       ax=ax2))]), 'kelbowvisualizer'),
        (Pipeline([("preprocessor", preprocessor), ("kelbowvisualizer"
                                                    , KElbowVisualizer(KMeans(), K=(4, 12), metric='calinski_harabasz',
                                                                       ax=ax3))]), 'kelbowvisualizer')
    ]
    # 对visual_grid1列表中的每个可视化网格进行拟合和最终化操作，并通过第一个可视化网格确定最佳的聚类数（K）值
    i = 0
    for viz in visual_grid1:
        viz[0].fit(X)
        if i == 0:
            # Defining the best K by distortion method
            K = viz[0].named_steps['kelbowvisualizer'].elbow_value_
            i = i + 1
        viz[0].named_steps[viz[1]].finalize()
    # 展示聚类结果的轮廓系数（'silhouette'）和聚类之间的距离（'distance'）
    visual_grid2 = [(Pipeline([("preprocessor", preprocessor), ("silhouettevisualizer",
                                                                SilhouetteVisualizer(KMeans(K, random_state=0),
                                                                                     ax=ax4))]),
                     'silhouettevisualizer'),
                    (Pipeline([("preprocessor", preprocessor), ("distancevisualizer",
                                                                InterclusterDistance(KMeans(K, random_state=0),
                                                                                     ax=ax5))]),
                     'distancevisualizer')]
    # 对visual_grid2列表中的每个可视化网格进行拟合和最终化操作
    for viz in visual_grid2:
        viz[0].fit(X)
        viz[0].named_steps[viz[1]].finalize()

    kmeans_model = Pipeline([("preprocessor", preprocessor),  # 对数据进行预处理
                             ("kmeans", KMeans(K, random_state=0))])  # 使用K个聚类中心进行聚类
    kmeans_model.fit(X)  # 对数据集X进行聚类，最后将聚类结果存储在X的"kmeans_label"列中

    kmeans_labels = kmeans_model.named_steps['kmeans'].labels_  # 获取聚类标签
    X["kmeans_label"] = kmeans_labels  # 将聚类标签（kmeans_labels）存储到数据集X的"kmeans_label"列中

    return X  # 返回带有聚类标签的数据集X


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
X_0 = Kmeansplots(data)  # 绘制 KMeans 聚类算法的可视化结果
plt.show()


# 对data_1进行降维处理
data_1_reduced = data[['order_id', 'payment_value', 'review_score',
                         'product_description_lenght', 'product_photos_qty',
                         'delivery_delay']]
# 对降维之后的数据进行聚类
X_1_reduced = Kmeansplots(data_1_reduced)
plt.show()

# 对X_1_reduced按kmeans_label进行分类
print(X_1_reduced.groupby('kmeans_label').agg({'order_id': ['count', 'median'],
                                         'payment_value': 'median',
                                         'review_score': 'median',
                                         'product_description_lenght': 'median',
                                         'product_photos_qty': 'median',
                                         'delivery_delay': 'median'
                                         }).to_string()

)

# 数据集 X_1_reduced 中不同列在不同聚类标签下的分布情况
n_plot = 231
fig = plt.figure(figsize=(24, 14))
for i in X_1_reduced.columns:
    if i != 'kmeans_label':
        axes = fig.add_subplot(n_plot)
        sns.violinplot(x=X_1_reduced['kmeans_label'], y=X_1_reduced[i], ax=axes)
        axes.set_title(i)
        axes.yaxis.grid(True)
        axes.set_xlabel('kmeans_label')
        n_plot = n_plot + 1
plt.show()

plt.figure(figsize=(24, 8))
plt.title("卖家的树状图")
# linkage函数对数据进行层次聚类分析，生成一个链接矩阵,使用 ward 方法进行聚类,用欧几里得距离作为距离度量
data_1_reduced = data_1_reduced.astype('timedelta64[s]')
link = shc.linkage(data_1_reduced, method='ward', metric='euclidean')
dend = shc.dendrogram(link, color_threshold=11000)
plt.show()

X_hier = data_1_reduced.copy()
# 聚类
cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
# 训练聚类模型并做出预测
agrup = cluster.fit_predict(X_hier)
X_hier['labels'] = agrup
X_hier.groupby('labels').agg({'order_id': ['count', 'median'],
                              'payment_value': 'median',
                              'review_score': 'median',
                              'product_description_lenght': 'median',
                              'product_photos_qty': 'median',
                              'delivery_delay': 'median'
                              })
# 数据集 X_hier 中不同列在不同聚类标签下的分布情况
n_plot = 231
fig = plt.figure(figsize=(24, 14))
for i in X_hier:
    if i != 'labels':
        axes = fig.add_subplot(n_plot)
        sns.violinplot(x=X_hier['labels'], y=X_hier[i], ax=axes)
        axes.set_title(i)
        axes.yaxis.grid(True)
        axes.set_xlabel('hierarchical_label')
        n_plot = n_plot + 1
plt.show()
