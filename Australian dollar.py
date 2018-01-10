
# coding: utf-8

# In[1]:


import warnings 
import itertools
import statsmodels.api as sm

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# In[2]:


data = pd.read_csv('test.csv')
print (data.head())
print ('\n Data Types:')
print (data.dtypes)


# In[3]:


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('test.csv', parse_dates=['date'], index_col='date',date_parser=dateparse)
print (data.head())


# In[4]:


data.index


# In[5]:


ts = data['Australian dollar / dollar']
ts.head(10)


# In[6]:


ts_log = np.log(ts)
plt.plot(ts_log)


# In[7]:


moving_avg = pd.rolling_mean(ts_log,150)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')


# In[8]:


ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(150)


# In[9]:


p = d = q = range(0, 2)


# In[10]:


pdq = list(itertools.product(p, d, q))


# In[11]:


seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


# In[12]:


print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[13]:


warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[15]:


mod = sm.tsa.statespace.SARIMAX(data,
                                order=(1, 0, 1),
                                seasonal_order=(0, 0, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])


# In[16]:


pred = results.get_prediction(start=pd.to_datetime('2015-01-04'), dynamic=False)
pred_ci = pred.conf_int()


# In[25]:


ax = data['2010':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Australian dollar / dollar')
plt.legend()

plt.show()


# In[59]:


date_forecasted = pred.predicted_mean
date_truth = data['2015-01-05':]


# In[60]:


pred_dynamic = results.get_prediction(start=pd.to_datetime('2015-01-04'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()


# In[61]:


ax = data['2010':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2015-01-04'), data.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('Australian dollar / dollar')
plt.legend()
plt.show()


# In[62]:


data_forecasted = pred_dynamic.predicted_mean
data_truth = data['2015-01-04':]
print(data_forecasted)
type(data_truth)


# In[63]:


pred_uc = results.get_forecast(steps=30)


# In[64]:


pred_ci = pred_uc.conf_int()


# In[65]:


ax = data.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Australian dollar / dollar')

plt.legend()
plt.show()


# In[195]:


high = [pred_ci.iloc[:, 1]]
print (type(lower))
lower = pred_ci.iloc[:, 0].to_dict()
high = pred_ci.iloc[:, 1].to_dict()
from collections import Counter,OrderedDict

sum_dict = dict(Counter(high)+Counter(lower))

keys = sum_dict.keys()
values = sum_dict.values()

result = dict(zip(keys, [x / 2 for x in values]))
ordered = OrderedDict(sorted(result.items(), key=lambda t: t[0]))

final = {}
for key, value in ordered.items():
    final[str(key.date())] = value


# In[196]:


final


# In[197]:


import json
json.dumps(final)

