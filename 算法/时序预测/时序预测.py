import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
import itertools
import holidays
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from IPython.display import clear_output
from tqdm import tqdm
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

orders = pd.read_csv('olist_orders_dataset.csv')
orders.head()
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders = orders[(orders['order_purchase_timestamp'] >= pd.to_datetime('2017-01-01')) & (
        orders['order_purchase_timestamp'] <= pd.to_datetime('2018-08-20'))]
orders = orders[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp']]
orders = orders[orders['order_status'].isin(['shipped', 'delivered', 'invoiced'])]
orders['date'] = orders['order_purchase_timestamp'].dt.date
orders = orders.groupby('date').agg(total_orders=('order_id', 'nunique')).reset_index()
orders['date'] = pd.to_datetime(orders['date'])
orders['dayofweek'] = orders['date'].dt.dayofweek
orders['quarter'] = orders['date'].dt.quarter
orders['month'] = orders['date'].dt.month
orders['year'] = orders['date'].dt.year
orders['dayofyear'] = orders['date'].dt.dayofyear
orders['dayofmonth'] = orders['date'].dt.day
orders['weekofyear'] = orders['date'].dt.isocalendar().week

br_holidays = holidays.BR()
orders['events'] = orders['date'].apply(lambda x: br_holidays.get(x))
orders.loc[orders['date'] == pd.to_datetime('2017-11-24'), 'events'] = 'black friday'
ann_color = '#c449cc'  # annotation color
arrowprops = dict(arrowstyle='-|>', color=ann_color, linewidth=2)
plt.figure(figsize=(20, 5))
sns.lineplot(data=orders, x='date', y='total_orders')

events = orders[orders['events'].notna()].to_dict('records')
for event in events:
    plt.annotate(event['events'],
                 xy=(event['date'], event['total_orders']),  # arrow position (x, y)
                 xytext=(event['date'], min(event['total_orders'] + 200, 1200)),  # text position (x, y)
                 fontsize=10,
                 arrowprops=arrowprops,
                 color=ann_color)

# 每周销售有一个规律，人们倾向于在工作日而不是周末购买产品，在年中和年底购买更多，在月底购买更少
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for idx, col in enumerate(['dayofweek', 'quarter', 'month', 'dayofmonth']):
    sns.boxplot(data=orders, x=col, y='total_orders', ax=ax[idx])

dates = orders['date'].sort_values().unique()
num_dates = len(dates)
dates_train = dates[:int(0.8 * num_dates)]
dates_test = dates[int(0.8 * num_dates):]
train = orders[orders['date'].isin(dates_train)]
test = orders[orders['date'].isin(dates_test)]
scaler = MinMaxScaler()
scaler.fit(train[['total_orders']])
train['scaling'] = scaler.transform(train[['total_orders']])
test['scaling'] = scaler.transform(test[['total_orders']])
train['rmv_outliers'] = winsorize(train['scaling'], (0.01, 0.009))
winsorized_min, winsorized_max = train[train['rmv_outliers'] != train['scaling']]['rmv_outliers'].min(), \
    train[train['rmv_outliers'] != train['scaling']]['rmv_outliers'].max()
winsorized_min, winsorized_max
test.loc[test['scaling'] > winsorized_max, 'rmv_outliers'] = winsorized_max
test.loc[test['scaling'] < winsorized_min, 'rmv_outliers'] = winsorized_min
test.loc[(test['scaling'] >= winsorized_min) & (test['scaling'] <= winsorized_max), 'rmv_outliers'] = \
    test[(test['scaling'] >= winsorized_min) & (test['scaling'] <= winsorized_max)]['scaling']


def plot_evaluation(dataset):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for idx, col in enumerate(['dayofweek', 'quarter', 'month', 'dayofmonth']):
        plot_data = dataset.groupby(col)['squared_error'].mean().apply(np.sqrt).reset_index()
        sns.barplot(data=plot_data, x=col, y='squared_error', ax=ax[idx], color='orange')
        ax[idx].set_ylabel('RMSE')

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    plot_data = dataset.groupby(['y', 'yhat'])['squared_error'].mean().apply(np.sqrt).reset_index()
    sns.scatterplot(data=plot_data, x='y', y='yhat', ax=ax[0], color='black')
    sns.scatterplot(data=plot_data, x='y', y='squared_error', ax=ax[1], color='black')
    ax[0].set_title("实际销售额和预测销售额之间的相关性")
    ax[0].set_xlabel('实际价值')
    ax[0].set_ylabel('预测价值')
    ax[1].set_xlabel('实际价值')
    ax[1].set_ylabel('RMSE')
    ax[1].set_title("实际值和RMSE之间的相关性")
    plt.show()


def plot_time_series(dataset, forecast2_half_year):
    plt.figure(figsize=(20, 5))
    ax = sns.lineplot(x='ds', y='yhat', data=dataset, label='预测值', color='orange')
    ax.fill_between(dataset['ds'], dataset['yhat_lower'], dataset['yhat_upper'], alpha=0.2)
    sns.lineplot(x='ds', y='y', data=dataset, label='真实值', color='black')
    sns.lineplot(x='ds', y='yhat', data=forecast2_half_year, label='未来60天预测值', color='red')
    plt.title("实际与预测的时间序列图")
    plt.show()


param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative']
}

holidays = orders[orders['events'].notna()][['date', 'events']]
holidays[['lower_window', 'upper_window']] = [-7, 7]
holidays = holidays.rename(columns={'date': 'ds', 'events': 'holiday'})


def predict_eval(train, test, variant, tuning=False, holiday_context=False):
    dataset = train.copy()
    dataset_test = test.copy()
    col = {'date': 'ds', variant: 'y'}
    dataset = dataset.rename(columns=col)
    dataset_test = dataset_test.rename(columns=col)
    m = Prophet()

    if holiday_context:
        m = Prophet(holidays=holidays)
        m.add_country_holidays(country_name='BR')

    if tuning:
        # 生成所有参数组合
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # 在此处存储每个参数的 RMSE

        # 使用交叉验证评估所有参数
        print("交互验证")
        for params in tqdm(all_params):
            if holiday_context:
                m = Prophet(**params).fit(dataset)  # 具有给定参数的拟合模型
            else:
                m = Prophet(holidays=holidays, **params).fit(dataset)
            df_cv = cross_validation(model=m, initial='200 days', period='30 days', horizon='30 days')

            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        clear_output()
        best_params = all_params[np.argmin(rmses)]
        print(best_params)
        if holiday_context:
            m = Prophet(holidays=holidays, changepoint_prior_scale=best_params['changepoint_prior_scale'],
                        seasonality_prior_scale=best_params['seasonality_prior_scale'],
                        seasonality_mode=best_params['seasonality_mode'])
        else:
            m = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                        seasonality_prior_scale=best_params['seasonality_prior_scale'],
                        seasonality_mode=best_params['seasonality_mode'])

    m.fit(dataset)
    future1 = pd.DataFrame(dates_test, columns=['ds'])
    forecast = m.predict(future1)
    forecast_today = m.predict(dataset)

    future2 = m.make_future_dataframe(periods=60)
    forecast2 = m.predict(future2)
    forecast2_half_year = forecast2[-60:]
    print(forecast2_half_year[['ds', 'yhat']])


    forecast_today.index = dataset.index
    dataset[['yhat', 'yhat_lower', 'yhat_upper']] = forecast_today[['yhat', 'yhat_lower', 'yhat_upper']]
    dataset['squared_error'] = np.square(dataset['y'] - dataset['yhat'])

    print("RMSE均方根误差:")
    print(f"RMSE: {np.sqrt(dataset['squared_error'].mean())}")
    plot_time_series(dataset, forecast2_half_year)
    # plot_evaluation(dataset)

    forecast.index = dataset_test.index
    dataset_test[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]
    dataset_test['squared_error'] = np.square(dataset_test['y'] - dataset_test['yhat'])


"""    print("测试数据集评估")
    print(f"RMSE: {np.sqrt(dataset_test['squared_error'].mean())}")
    plot_time_series(dataset_test, forecast2_half_year)
    # plot_evaluation(dataset_test)
"""

predict_eval(train, test, 'total_orders', tuning=True, holiday_context=True)
