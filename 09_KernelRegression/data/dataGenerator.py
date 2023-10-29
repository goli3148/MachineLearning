import numpy as np
import requests
def oneDimEx1() :
    return ([5,7,8,7,2,17,2,9,4,11,12,9,6],[99,86,87,88,111,86,103,87,94,78,77,85,86])

def oneDimEx2() :
    np.random.seed(1)
    # xwidth controls the range of x values.
    xwidth = 20
    x = np.arange(0,xwidth,1)
    # we want to add some noise to the x values so that dont sit at regular intervals
    x_residuals = np.random.normal(scale=0.2, size=[x.shape[0]])
    # new_x is the range of x values we will be using all the way through
    new_x = x + x_residuals
    # We generate residuals for y values since we want to show some variation in the data
    num_points = x.shape[0]
    residuals = np.random.normal(scale=2.0, size=[num_points])
    # We will be using fun_y to generate y values all the way through
    fun_y = lambda x: -(x*x) + residuals
    # Plot the x and y values 
    return (new_x, fun_y(new_x))

def oneDimEx3():
    x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
    y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
    return x,y

def twoDimEx1() :
    X_train = np.random.rand(2000).reshape(1000,2)*60
    y_train = (X_train[:, 0]**2)+(X_train[:, 1]**2)
    X_train = X_train.transpose()
    y_train = y_train.transpose()
    return X_train, y_train

def forexCurrencies(intervalSec=1, dataNum=60):
    y = [1827.49, 1800.8, 1807.995, 1925.07, 1897.77, 1846.025, 1811.2, 1772.105, 1697.525, 1662.13, 1646.885, 1802.435, 1823.905]
    x = [i for i in range(len(y))]
    return x, y
    import time
    
    url = "http://apilayer.net/api/historical"
    dates = []
    x = []
    y = []
    for i in range(1,13):
        if i<10: dates.append(f'2022-0{i}-01')
        else: dates.append(f'2022-{i}-01')
    dates.append("2023-01-01")
    counter = 0
    for date in dates:
        time.sleep(1)
        pa = {'access_key':'080ae986c4c869864ae53af4e09eb873', 'currencies' : "USD", 'source':"XAU", 'format':1, 'date':date}
        res = requests.get(url=url, params=pa)
        if res.status_code == 200:
            y.append(res.json()['quotes']['XAUUSD'])
            print(y[-1])
            x.append(counter)
            counter += 30
    return x,y

def forexPolyGon(TrainDataNumber='', TestDataNumber=''):
    from polygon import RESTClient
    prices = []
    x = [i for i in range(20)]
    client = RESTClient("EJ17S2Vp8OpJU4sv6_xSzUpdNrWLBSR_")
    aggs = []
    for a in client.list_aggs("SPEC", 1, "minute", "2023-10-27", "2023-10-27", limit=20):
        aggs.append(a)
        prices.append(a.close)
    print("GOT DATA")
    return x,prices
        
    
