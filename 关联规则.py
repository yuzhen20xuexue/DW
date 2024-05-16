import pandas as pd
import numpy as np

orders_data = pd.read_csv('olist_order_items_dataset.csv')
products_data = pd.read_csv('olist_products_dataset.csv')
translations_data = pd.read_csv('product_category_name_translation.csv')

orders_data.head()
orders_data.info()
products_data.head()
products_data.info()
translations_data.head()
translations_data.info()

products_data = products_data.merge(translations_data, on='product_category_name', how="left")
products_data['product_category_name_english'].head()

orders_data = orders_data.merge(products_data[['product_id', 'product_category_name_english']], on='product_id',
                                how='left')

orders_data.head()

# 删除没有产品名称的数据
orders_data.dropna(inplace=True, subset=['product_category_name_english'])

# 查看产品列表中的唯一值数量和类别名称数量
print("Number of available product ID    : ", orders_data['product_id'].nunique())
print("Number of available category name : ", orders_data['product_category_name_english'].nunique())

Total_transactions = orders_data.groupby("order_id").product_category_name_english.unique()
Total_transactions.head()

Total_transactions.value_counts()[:50].plot(kind='bar', figsize=(15, 5))

Total_transactions.value_counts()[-50:].plot(kind='bar', figsize=(15, 5))

Total_transactions_list = Total_transactions.tolist()

# 总交易次数
len(Total_transactions_list)

# Lets count unique item of categories per transaction.
counts_category = [len(transaction) for transaction in Total_transactions_list]

# Median item list
print('一个订单大部分商品类别数：', np.median(counts_category))

print('一个订单的最多商品类别数：', np.max(counts_category))

from mlxtend.preprocessing import TransactionEncoder

# 对数据进行编码处理
encoder = TransactionEncoder()
Category_column_data = encoder.fit_transform(Total_transactions_list)

# Convert array to pandas DataFrame.
Category_column_data = pd.DataFrame(Category_column_data, columns=encoder.columns_)
Category_column_data.head()

# 每个订单商品类别分布
Category_column_data.mean(axis=0) * 100
print(Category_column_data)

# 一个订单中购买类别数的分布
Category_column_data.sum(axis=1).value_counts(normalize=True) * 100
print(Category_column_data)

print(Category_column_data.columns)

# 分组一些类似的类别，并看到合并的结果
Category_column_data['books'] = Category_column_data['books_imported'] | Category_column_data['books_technical']
Category_column_data[['books', 'books_imported', 'books_technical']].mean(axis=0)
Category_column_data['sports_leisure_health_beauty'] = Category_column_data['sports_leisure'] & Category_column_data[
    'health_beauty']

Category_column_data['sports_leisure_health_beauty'].mean(axis=0)

# 计算运动休闲和健康美容的联合支持
joint_support = (Category_column_data['sports_leisure'] & Category_column_data['health_beauty']).mean()
joint_support / Category_column_data['sports_leisure'].mean()
joint_support / Category_column_data['health_beauty'].mean()

from mlxtend.frequent_patterns import apriori

# 对数据应用先验算法
# 最小支持阈值为0.01。
frequent_item_sets = apriori(Category_column_data, use_colnames=True, min_support=0.01)
print(frequent_item_sets)

# 增加阈值以获得具有 1 个以上项目的大量项集
frequent_item_sets = apriori(Category_column_data, use_colnames=True, min_support=0.00005, max_len=2)
print(frequent_item_sets)

# 计算关联规则
from mlxtend.frequent_patterns import association_rules

associationRule = association_rules(frequent_item_sets, metric='support', min_threshold=0.0001)
print(associationRule.head().to_string())

# 修剪关联规则
associationRule2 = association_rules(frequent_item_sets, metric='confidence', min_threshold=0.01)
print(associationRule2.to_string())

# 选择0.095以上的相应支持。
associationRule2 = associationRule2[associationRule2['consequent support'] > 0.095]
print("选择0.095以上的相应支持:\n", associationRule2.to_string())

associationRule2 = associationRule2[associationRule2['leverage'] > 0.0]
print("\n关联规则：\n", associationRule2.to_string())
