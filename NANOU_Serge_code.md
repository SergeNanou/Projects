
                                          TEST PYTHON CODE

Using packages


```python
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split

```

1-Import data AND back vizualisation


```python
df = pd.read_csv('C:/Users/akase/Desktop/Dossier_test/OnlineNewsPopularity.csv')
```


```python
df.shape
```




    (39644, 61)



The first elements of dataframe


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>timedelta</th>
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>...</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>http://mashable.com/2013/01/07/amazon-instant-...</td>
      <td>731.0</td>
      <td>12.0</td>
      <td>219.0</td>
      <td>0.663594</td>
      <td>1.0</td>
      <td>0.815385</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.7</td>
      <td>-0.350000</td>
      <td>-0.600</td>
      <td>-0.200000</td>
      <td>0.500000</td>
      <td>-0.187500</td>
      <td>0.000000</td>
      <td>0.187500</td>
      <td>593</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http://mashable.com/2013/01/07/ap-samsung-spon...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>255.0</td>
      <td>0.604743</td>
      <td>1.0</td>
      <td>0.791946</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.033333</td>
      <td>0.7</td>
      <td>-0.118750</td>
      <td>-0.125</td>
      <td>-0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>http://mashable.com/2013/01/07/apple-40-billio...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>211.0</td>
      <td>0.575130</td>
      <td>1.0</td>
      <td>0.663866</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.100000</td>
      <td>1.0</td>
      <td>-0.466667</td>
      <td>-0.800</td>
      <td>-0.133333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>http://mashable.com/2013/01/07/astronaut-notre...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>531.0</td>
      <td>0.503788</td>
      <td>1.0</td>
      <td>0.665635</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.136364</td>
      <td>0.8</td>
      <td>-0.369697</td>
      <td>-0.600</td>
      <td>-0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>http://mashable.com/2013/01/07/att-u-verse-apps/</td>
      <td>731.0</td>
      <td>13.0</td>
      <td>1072.0</td>
      <td>0.415646</td>
      <td>1.0</td>
      <td>0.540890</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>0.033333</td>
      <td>1.0</td>
      <td>-0.220192</td>
      <td>-0.500</td>
      <td>-0.050000</td>
      <td>0.454545</td>
      <td>0.136364</td>
      <td>0.045455</td>
      <td>0.136364</td>
      <td>505</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>



The last elements of dataframe


```python
df.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>timedelta</th>
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>...</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39639</th>
      <td>http://mashable.com/2014/12/27/samsung-app-aut...</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>346.0</td>
      <td>0.529052</td>
      <td>1.0</td>
      <td>0.684783</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.75</td>
      <td>-0.260000</td>
      <td>-0.5</td>
      <td>-0.125000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>1800</td>
    </tr>
    <tr>
      <th>39640</th>
      <td>http://mashable.com/2014/12/27/seth-rogen-jame...</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>328.0</td>
      <td>0.696296</td>
      <td>1.0</td>
      <td>0.885057</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.136364</td>
      <td>0.70</td>
      <td>-0.211111</td>
      <td>-0.4</td>
      <td>-0.100000</td>
      <td>0.300000</td>
      <td>1.000000</td>
      <td>0.200000</td>
      <td>1.000000</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>39641</th>
      <td>http://mashable.com/2014/12/27/son-pays-off-mo...</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>442.0</td>
      <td>0.516355</td>
      <td>1.0</td>
      <td>0.644128</td>
      <td>24.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>...</td>
      <td>0.136364</td>
      <td>0.50</td>
      <td>-0.356439</td>
      <td>-0.8</td>
      <td>-0.166667</td>
      <td>0.454545</td>
      <td>0.136364</td>
      <td>0.045455</td>
      <td>0.136364</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>39642</th>
      <td>http://mashable.com/2014/12/27/ukraine-blasts/</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>682.0</td>
      <td>0.539493</td>
      <td>1.0</td>
      <td>0.692661</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.062500</td>
      <td>0.50</td>
      <td>-0.205246</td>
      <td>-0.5</td>
      <td>-0.012500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1100</td>
    </tr>
    <tr>
      <th>39643</th>
      <td>http://mashable.com/2014/12/27/youtube-channel...</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>157.0</td>
      <td>0.701987</td>
      <td>1.0</td>
      <td>0.846154</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.50</td>
      <td>-0.200000</td>
      <td>-0.2</td>
      <td>-0.200000</td>
      <td>0.333333</td>
      <td>0.250000</td>
      <td>0.166667</td>
      <td>0.250000</td>
      <td>1300</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>




```python
df.columns
```




    Index(['url', ' timedelta', ' n_tokens_title', ' n_tokens_content',
           ' n_unique_tokens', ' n_non_stop_words', ' n_non_stop_unique_tokens',
           ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos',
           ' average_token_length', ' num_keywords', ' data_channel_is_lifestyle',
           ' data_channel_is_entertainment', ' data_channel_is_bus',
           ' data_channel_is_socmed', ' data_channel_is_tech',
           ' data_channel_is_world', ' kw_min_min', ' kw_max_min', ' kw_avg_min',
           ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg',
           ' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares',
           ' self_reference_max_shares', ' self_reference_avg_sharess',
           ' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday',
           ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday',
           ' weekday_is_sunday', ' is_weekend', ' LDA_00', ' LDA_01', ' LDA_02',
           ' LDA_03', ' LDA_04', ' global_subjectivity',
           ' global_sentiment_polarity', ' global_rate_positive_words',
           ' global_rate_negative_words', ' rate_positive_words',
           ' rate_negative_words', ' avg_positive_polarity',
           ' min_positive_polarity', ' max_positive_polarity',
           ' avg_negative_polarity', ' min_negative_polarity',
           ' max_negative_polarity', ' title_subjectivity',
           ' title_sentiment_polarity', ' abs_title_subjectivity',
           ' abs_title_sentiment_polarity', ' shares'],
          dtype='object')




```python
df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timedelta</th>
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>...</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>...</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
      <td>39644.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>354.530471</td>
      <td>10.398749</td>
      <td>546.514731</td>
      <td>0.548216</td>
      <td>0.996469</td>
      <td>0.689175</td>
      <td>10.883690</td>
      <td>3.293638</td>
      <td>4.544143</td>
      <td>1.249874</td>
      <td>...</td>
      <td>0.095446</td>
      <td>0.756728</td>
      <td>-0.259524</td>
      <td>-0.521944</td>
      <td>-0.107500</td>
      <td>0.282353</td>
      <td>0.071425</td>
      <td>0.341843</td>
      <td>0.156064</td>
      <td>3395.380184</td>
    </tr>
    <tr>
      <th>std</th>
      <td>214.163767</td>
      <td>2.114037</td>
      <td>471.107508</td>
      <td>3.520708</td>
      <td>5.231231</td>
      <td>3.264816</td>
      <td>11.332017</td>
      <td>3.855141</td>
      <td>8.309434</td>
      <td>4.107855</td>
      <td>...</td>
      <td>0.071315</td>
      <td>0.247786</td>
      <td>0.127726</td>
      <td>0.290290</td>
      <td>0.095373</td>
      <td>0.324247</td>
      <td>0.265450</td>
      <td>0.188791</td>
      <td>0.226294</td>
      <td>11626.950749</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>164.000000</td>
      <td>9.000000</td>
      <td>246.000000</td>
      <td>0.470870</td>
      <td>1.000000</td>
      <td>0.625739</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.050000</td>
      <td>0.600000</td>
      <td>-0.328383</td>
      <td>-0.700000</td>
      <td>-0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>946.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>339.000000</td>
      <td>10.000000</td>
      <td>409.000000</td>
      <td>0.539226</td>
      <td>1.000000</td>
      <td>0.690476</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.800000</td>
      <td>-0.253333</td>
      <td>-0.500000</td>
      <td>-0.100000</td>
      <td>0.150000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1400.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>542.000000</td>
      <td>12.000000</td>
      <td>716.000000</td>
      <td>0.608696</td>
      <td>1.000000</td>
      <td>0.754630</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.100000</td>
      <td>1.000000</td>
      <td>-0.186905</td>
      <td>-0.300000</td>
      <td>-0.050000</td>
      <td>0.500000</td>
      <td>0.150000</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>2800.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>731.000000</td>
      <td>23.000000</td>
      <td>8474.000000</td>
      <td>701.000000</td>
      <td>1042.000000</td>
      <td>650.000000</td>
      <td>304.000000</td>
      <td>116.000000</td>
      <td>128.000000</td>
      <td>91.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>843300.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 60 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39644 entries, 0 to 39643
    Data columns (total 61 columns):
    url                               39644 non-null object
     timedelta                        39644 non-null float64
     n_tokens_title                   39644 non-null float64
     n_tokens_content                 39644 non-null float64
     n_unique_tokens                  39644 non-null float64
     n_non_stop_words                 39644 non-null float64
     n_non_stop_unique_tokens         39644 non-null float64
     num_hrefs                        39644 non-null float64
     num_self_hrefs                   39644 non-null float64
     num_imgs                         39644 non-null float64
     num_videos                       39644 non-null float64
     average_token_length             39644 non-null float64
     num_keywords                     39644 non-null float64
     data_channel_is_lifestyle        39644 non-null float64
     data_channel_is_entertainment    39644 non-null float64
     data_channel_is_bus              39644 non-null float64
     data_channel_is_socmed           39644 non-null float64
     data_channel_is_tech             39644 non-null float64
     data_channel_is_world            39644 non-null float64
     kw_min_min                       39644 non-null float64
     kw_max_min                       39644 non-null float64
     kw_avg_min                       39644 non-null float64
     kw_min_max                       39644 non-null float64
     kw_max_max                       39644 non-null float64
     kw_avg_max                       39644 non-null float64
     kw_min_avg                       39644 non-null float64
     kw_max_avg                       39644 non-null float64
     kw_avg_avg                       39644 non-null float64
     self_reference_min_shares        39644 non-null float64
     self_reference_max_shares        39644 non-null float64
     self_reference_avg_sharess       39644 non-null float64
     weekday_is_monday                39644 non-null float64
     weekday_is_tuesday               39644 non-null float64
     weekday_is_wednesday             39644 non-null float64
     weekday_is_thursday              39644 non-null float64
     weekday_is_friday                39644 non-null float64
     weekday_is_saturday              39644 non-null float64
     weekday_is_sunday                39644 non-null float64
     is_weekend                       39644 non-null float64
     LDA_00                           39644 non-null float64
     LDA_01                           39644 non-null float64
     LDA_02                           39644 non-null float64
     LDA_03                           39644 non-null float64
     LDA_04                           39644 non-null float64
     global_subjectivity              39644 non-null float64
     global_sentiment_polarity        39644 non-null float64
     global_rate_positive_words       39644 non-null float64
     global_rate_negative_words       39644 non-null float64
     rate_positive_words              39644 non-null float64
     rate_negative_words              39644 non-null float64
     avg_positive_polarity            39644 non-null float64
     min_positive_polarity            39644 non-null float64
     max_positive_polarity            39644 non-null float64
     avg_negative_polarity            39644 non-null float64
     min_negative_polarity            39644 non-null float64
     max_negative_polarity            39644 non-null float64
     title_subjectivity               39644 non-null float64
     title_sentiment_polarity         39644 non-null float64
     abs_title_subjectivity           39644 non-null float64
     abs_title_sentiment_polarity     39644 non-null float64
     shares                           39644 non-null int64
    dtypes: float64(59), int64(1), object(1)
    memory usage: 18.5+ MB
    

2-Target shares exploratory


```python
df[' shares'].describe()
```




    count     39644.000000
    mean       3395.380184
    std       11626.950749
    min           1.000000
    25%         946.000000
    50%        1400.000000
    75%        2800.000000
    max      843300.000000
    Name:  shares, dtype: float64




```python
df[' shares']
```




    0          593
    1          711
    2         1500
    3         1200
    4          505
    5          855
    6          556
    7          891
    8         3600
    9          710
    10        2200
    11        1900
    12         823
    13       10000
    14         761
    15        1600
    16       13600
    17        3100
    18        5700
    19       17100
    20        2800
    21         598
    22         445
    23        1500
    24         852
    25         783
    26        1500
    27        1800
    28         462
    29         425
             ...  
    39614     1400
    39615     5700
    39616     2100
    39617      691
    39618     1400
    39619     1200
    39620     2400
    39621    24300
    39622     2900
    39623      947
    39624     3200
    39625     1400
    39626     1100
    39627     1200
    39628     1000
    39629     2400
    39630     1500
    39631      914
    39632     1700
    39633     1500
    39634     1000
    39635     1300
    39636     1700
    39637     1400
    39638     1200
    39639     1800
    39640     1900
    39641     1900
    39642     1100
    39643     1300
    Name:  shares, Length: 39644, dtype: int64




```python
plt.hist(df[' shares'])
```




    (array([  3.95630000e+04,   6.00000000e+01,   1.10000000e+01,
              4.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              0.00000000e+00,   3.00000000e+00,   1.00000000e+00,
              1.00000000e+00]),
     array([  1.00000000e+00,   8.43309000e+04,   1.68660800e+05,
              2.52990700e+05,   3.37320600e+05,   4.21650500e+05,
              5.05980400e+05,   5.90310300e+05,   6.74640200e+05,
              7.58970100e+05,   8.43300000e+05]),
     <a list of 10 Patch objects>)




![png](output_16_1.png)


3-Outliers dropping


```python
#drop outliers in the dataset
df_new = df[(df[' shares'] < 200000) | (df[' shares'] == 200000)]  
df_new.shape
```




    (39627, 61)




```python
#histogramm of variable share
sns.distplot(df_new[' shares'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2569dfb1eb8>




![png](output_19_1.png)



```python

df_new_1 = df[(df[' shares'] < 50000) | (df[' shares'] == 50000)]  
df_new_1.shape
```




    (39441, 61)




```python
sns.distplot(df_new_1[' shares'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2569f2d1c88>




![png](output_21_1.png)



```python
df_new_2 = df[(df[' shares'] < 10000) | (df[' shares'] == 10000)]  
df_new_2.shape
```




    (37459, 61)




```python
sns.distplot(df_new_2[' shares'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x256a06e5cc0>




![png](output_23_1.png)



```python
df_new_3 = df[(df[' shares'] < 8000) | (df[' shares'] == 8000)]  
df_new_3.shape
df_new_3[' shares'].describe()
```




    count    36761.000000
    mean      1896.326433
    std       1500.398385
    min          1.000000
    25%        919.000000
    50%       1300.000000
    75%       2300.000000
    max       8000.000000
    Name:  shares, dtype: float64




```python
df_new_3.shape
```




    (36761, 61)




```python
#histogramm of variable kw_avg_avg
sns.distplot(df_new_2[' kw_avg_avg'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x256a07da630>




![png](output_26_1.png)



```python
#dropping outliers 
df_new_2_1 = df_new_2[(df[' kw_avg_avg'] < 8000) | (df[' kw_avg_avg'] == 8000)]  
df_new_2_1.shape
```

    C:\Users\akase\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      
    




    (37253, 61)




```python
sns.distplot(df_new_2_1[' kw_avg_avg'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x256a09f60f0>




![png](output_28_1.png)


3-Correlations of variable shares and explication 


```python
#correlations of a target variable
correlations = df_new_2_1.corr()
correlations = correlations[" shares"].sort_values(ascending=False)
correlations
```




     shares                           1.000000
     kw_avg_avg                       0.198626
     kw_max_avg                       0.124048
     is_weekend                       0.104687
     kw_min_avg                       0.096792
     data_channel_is_socmed           0.090247
     num_hrefs                        0.079517
     weekday_is_saturday              0.071584
     weekday_is_sunday                0.071321
     LDA_03                           0.069049
     data_channel_is_tech             0.067463
     num_imgs                         0.067100
     num_keywords                     0.062575
     LDA_04                           0.058610
     self_reference_avg_sharess       0.056963
     self_reference_max_shares        0.053125
     kw_avg_min                       0.052025
     global_subjectivity              0.049803
     global_rate_positive_words       0.046918
     LDA_00                           0.046792
     self_reference_min_shares        0.046010
     global_sentiment_polarity        0.043787
     title_sentiment_polarity         0.042013
     abs_title_sentiment_polarity     0.041849
     kw_max_min                       0.040862
     title_subjectivity               0.038809
     n_tokens_content                 0.034507
     data_channel_is_lifestyle        0.029928
     kw_avg_max                       0.029451
     max_positive_polarity            0.028032
     timedelta                        0.027173
     num_self_hrefs                   0.026649
     kw_min_min                       0.020765
     rate_positive_words              0.017055
     avg_positive_polarity            0.014334
     num_videos                       0.014123
     kw_min_max                       0.010650
     n_non_stop_words                 0.010484
     n_unique_tokens                  0.009677
     n_non_stop_unique_tokens         0.008944
     abs_title_subjectivity           0.003139
     max_negative_polarity           -0.003151
     weekday_is_friday               -0.004652
     kw_max_max                      -0.007759
     weekday_is_monday               -0.008595
     global_rate_negative_words      -0.012973
     avg_negative_polarity           -0.014676
     min_negative_polarity           -0.014751
     data_channel_is_bus             -0.017104
     min_positive_polarity           -0.021368
     weekday_is_thursday             -0.021975
     weekday_is_tuesday              -0.025033
     weekday_is_wednesday            -0.030730
     n_tokens_title                  -0.033803
     average_token_length            -0.039004
     rate_negative_words             -0.050413
     LDA_01                          -0.054326
     data_channel_is_entertainment   -0.084390
     data_channel_is_world           -0.124779
     LDA_02                          -0.131418
    Name:  shares, dtype: float64




```python
#correlation matrix
corr_matrix = df_new_2_1.corr()
f, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(corr_matrix, vmax=.8, square=True);
```


![png](output_31_0.png)



```python
#share correlation matrix
k = 12 # number of variables
col_s = corr_matrix.nlargest(k, ' shares')[' shares'].index
c_m = np.corrcoef(df[col_s].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(c_m, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=col_s.values, xticklabels=col_s.values)
plt.show()

```


![png](output_32_0.png)


4-Regression models


```python
X = df_new_2_1.drop(['url', ' shares'], 1)  
y = df_new_2_1[' shares'] 
```


```python


std_scale = preprocessing.StandardScaler().fit(X)
X_scale = std_scale.transform(X)
```


```python
# regression using statsmodels
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 shares   R-squared:                       0.094
    Model:                            OLS   Adj. R-squared:                  0.093
    Method:                 Least Squares   F-statistic:                     68.01
    Date:                Mon, 10 Sep 2018   Prob (F-statistic):               0.00
    Time:                        01:55:54   Log-Likelihood:            -3.2961e+05
    No. Observations:               37253   AIC:                         6.593e+05
    Df Residuals:                   37195   BIC:                         6.598e+05
    Df Model:                          57                                         
    Covariance Type:            nonrobust                                         
    ==================================================================================================
                                         coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------------------
     timedelta                         0.0974      0.060      1.620      0.105      -0.020       0.215
     n_tokens_title                    2.6473      4.407      0.601      0.548      -5.991      11.286
     n_tokens_content                  0.0984      0.035      2.852      0.004       0.031       0.166
     n_unique_tokens                   6.5684    293.296      0.022      0.982    -568.301     581.437
     n_non_stop_words                 31.2661    867.914      0.036      0.971   -1669.869    1732.401
     n_non_stop_unique_tokens       -215.2929    248.250     -0.867      0.386    -701.870     271.285
     num_hrefs                         5.4959      1.040      5.285      0.000       3.458       7.534
     num_self_hrefs                  -15.4561      2.698     -5.729      0.000     -20.744     -10.169
     num_imgs                          3.4914      1.371      2.547      0.011       0.805       6.178
     num_videos                        2.8586      2.451      1.166      0.243      -1.945       7.662
     average_token_length           -150.7209     36.978     -4.076      0.000    -223.198     -78.244
     num_keywords                     24.3576      5.710      4.266      0.000      13.166      35.549
     data_channel_is_lifestyle      -159.8123     60.830     -2.627      0.009    -279.041     -40.584
     data_channel_is_entertainment  -241.6804     39.693     -6.089      0.000    -319.479    -163.881
     data_channel_is_bus            -243.0578     58.943     -4.124      0.000    -358.587    -127.529
     data_channel_is_socmed          396.8108     56.980      6.964      0.000     285.128     508.493
     data_channel_is_tech            276.8101     57.052      4.852      0.000     164.986     388.634
     data_channel_is_world           -11.5592     58.199     -0.199      0.843    -125.630     102.512
     kw_min_min                        0.6492      0.248      2.617      0.009       0.163       1.135
     kw_max_min                       -0.0066      0.015     -0.441      0.659      -0.036       0.023
     kw_avg_min                        0.0076      0.091      0.083      0.934      -0.170       0.185
     kw_min_max                       -0.0003      0.000     -1.554      0.120      -0.001    7.27e-05
     kw_max_max                       -0.0001   8.91e-05     -1.526      0.127      -0.000    3.86e-05
     kw_avg_max                       -0.0007      0.000     -5.130      0.000      -0.001      -0.000
     kw_min_avg                       -0.0718      0.012     -5.911      0.000      -0.096      -0.048
     kw_max_avg                       -0.0650      0.006    -10.768      0.000      -0.077      -0.053
     kw_avg_avg                        0.6060      0.025     23.891      0.000       0.556       0.656
     self_reference_min_shares         0.0006      0.001      0.493      0.622      -0.002       0.003
     self_reference_max_shares         0.0004      0.001      0.609      0.542      -0.001       0.002
     self_reference_avg_sharess        0.0015      0.002      0.882      0.378      -0.002       0.005
     weekday_is_monday              1.082e+05      9e+05      0.120      0.904   -1.66e+06    1.87e+06
     weekday_is_tuesday             1.081e+05      9e+05      0.120      0.904   -1.66e+06    1.87e+06
     weekday_is_wednesday           1.081e+05      9e+05      0.120      0.904   -1.66e+06    1.87e+06
     weekday_is_thursday            1.081e+05      9e+05      0.120      0.904   -1.66e+06    1.87e+06
     weekday_is_friday              1.082e+05      9e+05      0.120      0.904   -1.66e+06    1.87e+06
     weekday_is_saturday             3.62e+04      3e+05      0.121      0.904   -5.52e+05    6.24e+05
     weekday_is_sunday              3.622e+04      3e+05      0.121      0.904   -5.52e+05    6.24e+05
     is_weekend                     7.241e+04      6e+05      0.121      0.904    -1.1e+06    1.25e+06
     LDA_00                        -1.071e+05      9e+05     -0.119      0.905   -1.87e+06    1.66e+06
     LDA_01                        -1.078e+05      9e+05     -0.120      0.905   -1.87e+06    1.66e+06
     LDA_02                        -1.079e+05      9e+05     -0.120      0.905   -1.87e+06    1.66e+06
     LDA_03                        -1.079e+05      9e+05     -0.120      0.905   -1.87e+06    1.66e+06
     LDA_04                        -1.075e+05      9e+05     -0.120      0.905   -1.87e+06    1.66e+06
     global_subjectivity             407.5208    129.044      3.158      0.002     154.591     660.451
     global_sentiment_polarity      -312.3706    255.113     -1.224      0.221    -812.398     187.657
     global_rate_positive_words       83.5835   1091.321      0.077      0.939   -2055.436    2222.603
     global_rate_negative_words     -659.7663   2102.052     -0.314      0.754   -4779.847    3460.314
     rate_positive_words             634.9722    846.718      0.750      0.453   -1024.619    2294.563
     rate_negative_words             471.6378    853.931      0.552      0.581   -1202.091    2145.367
     avg_positive_polarity          -188.7258    207.996     -0.907      0.364    -596.404     218.953
     min_positive_polarity          -282.6354    174.226     -1.622      0.105    -624.123      58.852
     max_positive_polarity           -31.8782     65.102     -0.490      0.624    -159.481      95.724
     avg_negative_polarity           -74.0433    191.110     -0.387      0.698    -448.625     300.538
     min_negative_polarity           -43.3816     69.559     -0.624      0.533    -179.719      92.956
     max_negative_polarity            51.2269    159.422      0.321      0.748    -261.246     363.699
     title_subjectivity              146.9565     41.805      3.515      0.000      65.018     228.895
     title_sentiment_polarity        121.3334     38.335      3.165      0.002      46.196     196.471
     abs_title_subjectivity          260.9587     55.323      4.717      0.000     152.524     369.394
     abs_title_sentiment_polarity     -5.2520     60.554     -0.087      0.931    -123.940     113.436
    ==============================================================================
    Omnibus:                    15495.986   Durbin-Watson:                   1.969
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            63524.875
    Skew:                           2.084   Prob(JB):                         0.00
    Kurtosis:                       7.852   Cond. No.                     1.38e+16
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.34e-16. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    

5-Features selection using random forest


```python
df_new_2_1 = df_new_2_1.drop(['url'], 1)
```


```python
# Labels are the values we want to predict
labels = np.array(df_new_2_1[' shares'])
```


```python
# Convert to numpy array
df_new_2_1 = np.array(df_new_2_1)
```


```python
# feature importance using random forest
rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(X, y)
ranking = np.argsort(-rf.feature_importances_)
f, ax = plt.subplots(figsize=(15, 12))
sns.barplot(x=rf.feature_importances_[ranking], y=X.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()
plt.show()
```


![png](output_41_0.png)



```python
# use the top 10 features only
X= X.iloc[:,ranking[:10]]
print(X)

```

            kw_avg_avg    kw_max_avg   timedelta    LDA_00     kw_avg_max  \
    0         0.000000      0.000000       731.0  0.500331       0.000000   
    1         0.000000      0.000000       731.0  0.799756       0.000000   
    2         0.000000      0.000000       731.0  0.217792       0.000000   
    3         0.000000      0.000000       731.0  0.028573       0.000000   
    4         0.000000      0.000000       731.0  0.028633       0.000000   
    5         0.000000      0.000000       731.0  0.022245       0.000000   
    6         0.000000      0.000000       731.0  0.020082       0.000000   
    7         0.000000      0.000000       731.0  0.022224       0.000000   
    8         0.000000      0.000000       731.0  0.458250       0.000000   
    9         0.000000      0.000000       731.0  0.040000       0.000000   
    10        0.000000      0.000000       731.0  0.025004       0.000000   
    11        0.000000      0.000000       731.0  0.028628       0.000000   
    12        0.000000      0.000000       731.0  0.150493       0.000000   
    13        0.000000      0.000000       731.0  0.033386       0.000000   
    14        0.000000      0.000000       731.0  0.028780       0.000000   
    15        0.000000      0.000000       731.0  0.033334       0.000000   
    17        0.000000      0.000000       731.0  0.866666       0.000000   
    18        0.000000      0.000000       731.0  0.437374       0.000000   
    20        0.000000      0.000000       731.0  0.020069       0.000000   
    21        0.000000      0.000000       731.0  0.028774       0.000000   
    22        0.000000      0.000000       731.0  0.311931       0.000000   
    23        0.000000      0.000000       731.0  0.033334       0.000000   
    24        0.000000      0.000000       731.0  0.300294       0.000000   
    25        0.000000      0.000000       731.0  0.020006       0.000000   
    26        0.000000      0.000000       731.0  0.022243       0.000000   
    27        0.000000      0.000000       731.0  0.022275       0.000000   
    28        0.000000      0.000000       731.0  0.020041       0.000000   
    29        0.000000      0.000000       731.0  0.866657       0.000000   
    30        0.000000      0.000000       731.0  0.744049       0.000000   
    31        0.000000      0.000000       731.0  0.028645       0.000000   
    ...            ...           ...         ...       ...            ...   
    39613  2575.552255   4032.469314         9.0  0.121078  258400.000000   
    39614  2176.103343   3385.393320         9.0  0.452531  245654.714286   
    39615  3605.376162   5471.574662         9.0  0.020322  194840.000000   
    39616  3835.453639   5829.174629         9.0  0.033414  433366.666667   
    39617  2125.697028   3411.416667         9.0  0.028638  153457.142857   
    39618  2833.317260   3785.425532         9.0  0.022223  315200.000000   
    39619  2704.113469   4366.727273         9.0  0.020001  221420.000000   
    39620  1674.398279   3785.425532         9.0  0.022227  168711.666667   
    39622  1908.414298   3385.393320         9.0  0.136609  195322.222222   
    39623  2007.708083   3385.393320         9.0  0.025004  275137.500000   
    39624  3240.673969   6433.333333         9.0  0.028572  224885.714286   
    39625  2824.660363   3900.000000         9.0  0.020009  310130.000000   
    39626  2883.640499   3385.393320         9.0  0.040020  571200.000000   
    39627  1996.065900   3385.393320         9.0  0.033336  333833.333333   
    39628  2263.370238   3385.393320         9.0  0.040000  445640.000000   
    39629  5288.236851  13546.045454         9.0  0.040000  304580.000000   
    39630  1575.740886   3385.393320         9.0  0.109512  173830.000000   
    39631  2314.316345   3385.393320         9.0  0.040001  281640.000000   
    39632  2730.943979   4966.668990         9.0  0.028579  266628.571429   
    39633  2122.427735   3385.393320         9.0  0.033334  211750.000000   
    39634  5286.919100   7519.376771         9.0  0.022259  309255.555556   
    39635  3929.218477   7028.659675         9.0  0.865892  501016.666667   
    39636  4349.053221   6880.687034         8.0  0.165551  275140.000000   
    39637  2746.804338   4288.893701         8.0  0.025000  200800.000000   
    39638  2665.713159   4301.332394         8.0  0.551338  484083.333333   
    39639  3031.115764   4004.342857         8.0  0.025038  374962.500000   
    39640  3411.660830   5470.168651         8.0  0.029349  192985.714286   
    39641  4206.439195   6880.687034         8.0  0.159004  295850.000000   
    39642  1777.895883   3384.316871         8.0  0.040004  254600.000000   
    39643  3296.909481   3613.512953         8.0  0.050001  366200.000000   
    
            average_token_length    LDA_01   self_reference_min_shares  \
    0                   4.680365  0.378279                       496.0   
    1                   4.913725  0.050047                         0.0   
    2                   4.393365  0.033334                       918.0   
    3                   4.404896  0.419300                         0.0   
    4                   4.682836  0.028794                       545.0   
    5                   4.359459  0.306718                      8500.0   
    6                   4.654167  0.114705                       545.0   
    7                   4.617796  0.150733                       545.0   
    8                   4.855670  0.028979                         0.0   
    9                   5.090909  0.040000                         0.0   
    10                  4.617788  0.287301                         0.0   
    11                  4.657754  0.028573                         0.0   
    12                  4.233577  0.025934                     10700.0   
    13                  4.343860  0.033427                       770.0   
    14                  5.023166  0.028814                      4800.0   
    15                  4.620235  0.033334                         0.0   
    17                  5.445844  0.033333                         0.0   
    18                  4.844660  0.200363                      5000.0   
    20                  4.686699  0.020005                       545.0   
    21                  5.296675  0.028577                       704.0   
    22                  4.629983  0.232678                       545.0   
    23                  4.824000  0.033335                     16100.0   
    24                  4.422131  0.050001                      2800.0   
    25                  4.259398  0.367328                       924.0   
    26                  4.782477  0.022587                      2500.0   
    27                  4.635918  0.362350                       545.0   
    28                  4.382716  0.020031                         0.0   
    29                  5.228216  0.033337                         0.0   
    30                  4.620056  0.169324                      6100.0   
    31                  4.985782  0.200300                         0.0   
    ...                      ...       ...                         ...   
    39613               0.000000  0.120580                         0.0   
    39614               4.884956  0.028572                      1200.0   
    39615               0.000000  0.020021                         0.0   
    39616               0.000000  0.033347                         0.0   
    39617               4.314834  0.029660                       721.0   
    39618               5.153689  0.312897                       812.0   
    39619               4.478972  0.519475                      1400.0   
    39620               4.992736  0.364140                      1500.0   
    39622               4.866310  0.022223                      2000.0   
    39623               4.146789  0.025005                       878.0   
    39624               4.263403  0.172060                      1300.0   
    39625               4.589286  0.219217                      1100.0   
    39626               4.313253  0.040004                      3200.0   
    39627               4.633867  0.033336                       749.0   
    39628               5.078275  0.040000                       921.0   
    39629               4.856459  0.040903                      1100.0   
    39630               4.054990  0.020049                       691.0   
    39631               4.382038  0.040000                      1700.0   
    39632               5.005172  0.028573                      1400.0   
    39633               5.094463  0.033335                         0.0   
    39634               4.685259  0.022242                       941.0   
    39635               4.891213  0.034032                      7200.0   
    39636               4.569550  0.020013                      4900.0   
    39637               4.552486  0.768945                       857.0   
    39638               4.923767  0.033337                      2000.0   
    39639               4.523121  0.025001                     11400.0   
    39640               4.405488  0.028575                      2100.0   
    39641               5.076923  0.025025                      1400.0   
    39642               4.975073  0.040003                       452.0   
    39643               4.471338  0.799339                      2100.0   
    
            global_subjectivity   n_unique_tokens  
    0                  0.521617          0.663594  
    1                  0.341246          0.604743  
    2                  0.702222          0.575130  
    3                  0.429850          0.503788  
    4                  0.513502          0.415646  
    5                  0.437409          0.559889  
    6                  0.514480          0.418163  
    7                  0.543474          0.433574  
    8                  0.538889          0.670103  
    9                  0.313889          0.636364  
    10                 0.482060          0.490050  
    11                 0.477165          0.666667  
    12                 0.534950          0.609195  
    13                 0.509744          0.744186  
    14                 0.295175          0.562753  
    15                 0.473285          0.459542  
    17                 0.374314          0.624679  
    18                 0.423611          0.689320  
    20                 0.506535          0.390638  
    21                 0.284211          0.510256  
    22                 0.533658          0.427305  
    23                 0.396402          0.674797  
    24                 0.331640          0.560000  
    25                 0.313095          0.572581  
    26                 0.560696          0.562691  
    27                 0.480778          0.385452  
    28                 0.517984          0.619247  
    29                 0.375049          0.490934  
    30                 0.491428          0.482219  
    31                 0.531972          0.476038  
    ...                     ...               ...  
    39613              0.000000          0.000000  
    39614              0.385977          0.472158  
    39615              0.000000          0.000000  
    39616              0.000000          0.000000  
    39617              0.493241          0.459173  
    39618              0.425799          0.517454  
    39619              0.469273          0.512881  
    39620              0.482620          0.556675  
    39622              0.397040          0.439421  
    39623              0.540285          0.529412  
    39624              0.434468          0.514925  
    39625              0.384271          0.570136  
    39626              0.440992          0.567227  
    39627              0.521271          0.480702  
    39628              0.408462          0.552504  
    39629              0.486905          0.656863  
    39630              0.434965          0.454167  
    39631              0.500530          0.465306  
    39632              0.428833          0.506261  
    39633              0.469965          0.476033  
    39634              0.755769          0.666667  
    39635              0.391249          0.514039  
    39636              0.483535          0.348878  
    39637              0.440152          0.425711  
    39638              0.552041          0.653153  
    39639              0.482679          0.529052  
    39640              0.564374          0.696296  
    39641              0.510296          0.516355  
    39642              0.358578          0.539493  
    39643              0.517893          0.701987  
    
    [37253 rows x 10 columns]
    


```python
X.shape
```




    (37253, 10)



6-Prediction target variable(share) using random forest


```python
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df_new_2_1, labels, test_size = 0.3, random_state = 42)
```


```python
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

```

    Training Features Shape: (26077, 60)
    Training Labels Shape: (26077,)
    Testing Features Shape: (11176, 60)
    Testing Labels Shape: (11176,)
    


```python
# Instantiate model with 500 decision trees
rf = RandomForestRegressor(n_estimators = 500, random_state = 22)
# Train the model on training data
rf.fit(train_features, train_labels);
```


```python
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
print(errors)
```

    [ 0.  0.  0. ...,  0.  0.  0.]
    


```python
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
```


```python
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
```

    Accuracy: 99.77 %.
    
