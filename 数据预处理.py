import numpy as np
import pandas as pd

# 导入相关数据集，并借机重命名
orders = pd.read_csv('olist_orders_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')
order_items = pd.read_csv('算法/order_items1.csv')
products = pd.read_csv('olist_products_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')

result = order_items.groupby('order_id').last().reset_index()
result.to_csv(r'C:\Users\xy\Desktop\数据仓库\order_items.csv', index=False)
print(result.to_string(index=False))


# 统计缺失值数量和比例

def missing_value(data, num):
    null_data = data.isnull().sum().sort_values(ascending=False)  # 统计空值数量，并按照降序排列
    percent_1 = data.isnull().sum() / data.isnull().count()  # 计算空值占比
    missing_data = pd.concat([null_data, percent_1.apply(lambda x: format(x, '.2%'))],
                             axis=1, keys=['total missing', 'missing percentage'])
    print('\n该表中缺失值数量和占比如下：\n', missing_data.head(num))


missing_value(orders, 8)
missing_value(payments, 5)
missing_value(customers, 5)
missing_value(order_items, 9)
missing_value(products, 9)
missing_value(order_reviews, 7)
missing_value(sellers, 4)

print(order_items.describe())

# 缺失值剔除
orders = orders.dropna()
products = products.dropna(subset=['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'])

print(payments.describe())
print(products.describe().to_string(index=False))

payments = payments.drop(index=payments[payments['payment_installments'] == 0].index)
# print(payments.describe())

products = products.drop(index=products[products['product_weight_g'] == 0].index)
# print(products.describe().to_string(index=False))

# print(payments[payments['payment_value'] == 0].to_string(index=False))

# 查看每一行数据是否存在重复值
print("orders中的重复值个数", orders.duplicated().sum())
print("payments中的重复值个数", payments.duplicated().sum())
print("customers中的重复值个数", customers.duplicated().sum())
print("order_items中的重复值个数", order_items.duplicated().sum())
print("products中的重复值个数", products.duplicated().sum())
print("order_reviews中的重复值个数", order_reviews.duplicated().sum())
print("seller中的重复值个数", sellers.duplicated().sum())

# payment_type异常值剔除
payments = payments.drop(index=payments[payments['payment_type'] == "not_defined"].index)
print(payments[payments['payment_value'] == 0].to_string(index=False))

# 把order中字符串转化为日期时间数据类型
def transform_datetime(data, colum_list):
    for i in colum_list:
        data[i] = pd.to_datetime(data[i])
    print('数据类型转化成功！\n')


colum_list = ['order_purchase_timestamp', 'order_approved_at',
              'order_delivered_carrier_date', 'order_delivered_customer_date',
              'order_estimated_delivery_date']
transform_datetime(orders, colum_list)
orders.info()

missing_value(orders, 5)
missing_value(payments, 5)
missing_value(customers, 5)
missing_value(order_items, 5)

# 观察payment表中的付款异常值
print(payments.describe())

# 查看payment_installments异常值的详情数据


print(payments[payments['payment_installments'] == 0].to_string(index=False))

# 异常值剔除
payments = payments.drop(index=payments[payments['payment_installments'] == 0].index)
print(payments.describe())

print(payments[payments['payment_installments'] == 0].to_string(index=False))

# 查询payment_value异常值的详情数据
print(payments[payments['payment_value'] == 0].to_string(index=False))

# payment_type异常值剔除
payments = payments.drop(index=payments[payments['payment_type'] == "not_defined"].index)
print(payments[payments['payment_value'] == 0].to_string(index=False))

# 查看每一行数据是否存在重复值
print("orders中的重复值个数", orders.duplicated().sum())
print("payments中的重复值个数", payments.duplicated().sum())
print("customers中的重复值个数", customers.duplicated().sum())
print("order_items中的重复值个数", order_items.duplicated().sum())


# 把order中字符串转化为日期时间数据类型
def transform_datetime(data, colum_list):
    for i in colum_list:
        data[i] = pd.to_datetime(data[i])
    print('数据类型转化成功！\n')


colum_list = ['order_purchase_timestamp', 'order_approved_at',
              'order_delivered_carrier_date', 'order_delivered_customer_date',
              'order_estimated_delivery_date']
transform_datetime(orders, colum_list)
orders.info()

# 把清洗好的数据导出
# orders.to_csv(r'C:\Users\xy\Desktop\数据仓库\sx\orders.csv', index=False)
# payments.to_csv(r'C:\Users\xy\Desktop\数据仓库\sx\payments.csv', index=False)
# customers.to_csv(r'C:\Users\xy\Desktop\数据仓库\sx\customers.csv', index=False)
# order_items.to_csv(r'C:\Users\xy\Desktop\数据仓库\sx\order_items.csv', index=False)
# products.to_csv(r'C:\Users\xy\Desktop\数据仓库\sx\products.csv', index=False)
# sellers.to_csv(r'C:\Users\xy\Desktop\数据仓库\sx\sellers.csv', index=False)
# order_reviews.to_csv(r'C:\Users\xy\Desktop\数据仓库\sx\order_reviews.csv', index=False)
