#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
x=2*np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)


# In[21]:


xx=np.c_[np.ones((100,1)),x]
bestw=np.linalg.inv(xx.T.dot(xx)).dot(xx.T).dot(y)


# In[22]:


bestw


# In[23]:


xn=np.array([[0],[2]])
xn


# In[24]:


xxn=np.c_[np.ones((2,1)),xn]
yp=xxn.dot(bestw)


# In[25]:


yp


# In[29]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(xn,yp,'r-')
plt.plot(x,y,'b.')
plt.axis([0,2,0,15])
plt.show()


# In[30]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)
lin_reg.intercept_,lin_reg.coef_


# In[31]:


lin_reg.predict(xn)


# In[35]:


bestw_svd,residuals,rank,s=np.linalg.lstsq(xx,y,rcond=1e-6)
bestw_svd


# In[37]:


eta=0.1
n_iterations=1000
m=100

theta=np.random.randn(2,1)
for iteration in range(n_iterations):
    gradients=2/m*xx.T.dot(xx.dot(theta)-y)
    theta=theta-eta*gradients


# In[38]:


theta


# In[48]:


n_epochs=50
t0,t1=5,50

def learning_schedule(t):
    return t0/(t+t1)
theta=np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range (m):
        random_index=np.random.randint(m)
        xi=xx[random_index:random_index+1]
        yi=y[random_index:random_index+1]
        gradients=2*xi.T.dot(xi.dot(theta)-yi)
        eta=learning_schedule(epoch*m+i)
        theta=theta-eta*gradients


# In[49]:


theta


# In[65]:


from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2,include_bias=False)
x_poly=poly_features.fit_transform(x)
x
x_poly


# In[66]:


lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)
lin_reg.intercept_,lin_reg.coef_


# In[ ]:





# In[ ]:





# In[121]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model,x,y):
    x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.3)
    train_errors,val_errors=[],[]
    for m in range(1,len(x_train)):
        model.fit(x_train[:m],y_train[:m])
        y_train_predict=model.predict(x_train[:m])
        y_val_predict=model.predict(x_val)
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
        val_errors.append(mean_squared_error(y_val,y_val_predict))
        
    plt.plot(np.sqrt(train_errors),'r-+',linewidth=2,label='train')
    plt.plot(np.sqrt(val_errors),'b-',linewidth=3,label='val')


# In[124]:


x_train[:m]


# In[123]:


y_train


# In[112]:


lin_reg=LinearRegression()
plot_learning_curves(lin_reg,x,y)


# In[113]:


from sklearn.linear_model import Ridge


# In[114]:


rr=Ridge(alpha=0.1,solver='cholesky')
rr.fit(x,y)
rr.predict([[0.5]])


# In[115]:


from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor(penalty='l2')
sgdr.fit(x,y.ravel())
sgdr.predict([[1.5]])


# In[116]:


from sklearn.linear_model import Lasso
lassor=Lasso(alpha=1)
lassor.fit(x,y)
lassor.predict([[0.5]])


# In[117]:


from sklearn.linear_model import ElasticNet
en=ElasticNet(alpha=0.98,l1_ratio=0.5)
en.fit(x,y)
en.predict([[1.5]])


# In[119]:


from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
poly_scaler=Pipeline([('poly_features',PolynomialFeatures(degree=90,include_bias=False)),('std_scaler',StandardScaler())])
x_train_poly_scaled=poly_scaler.fit_transform(x_train)
x_val_poly_scaled=poly_scaler.transform(x_val)
sgdr=SGDRegressor(max_iter=1,tol=-np.infty,warm_start=True,penalty=None,learning_rate='constant',eta=0.0005)
minimum_val_error=float('inf')
best_epoch=None
best_model=None
for epoch in range(1000):
    sgdr.fit(x_train_poly_scaled,y_train)
    y_val_predict=sgdr.predict(x_train_poly_scaled)
    val_error=mean_squared_error(y_val,y_val_predict)
    if val_error<minimum_val_error:
        minimum_val_error=val_error
        best_epoch=epoch
        best_model=clone(sgdr)


# In[125]:


from sklearn import datasets
iris=datasets.load_iris()
list(iris.keys())


# In[133]:


iris


# In[126]:


x=iris['data'][:,3:]
y=(iris['target']==2).astype(np.int)
x,y


# In[131]:


from sklearn.linear_model import LogisticRegression
logr=LogisticRegression()
logr.fit(x,y)


# In[134]:


xn=np.linspace(0,3,1000).reshape(-1,1)
yp=logr.predict_proba(xn)
plt.plot(xn,yp[:,1],'g--',label='iris veroni')
plt.plot(xn,yp[:,0],'b-',label='not iris veroni')


# In[140]:


logr.predict([[1.68],[2.0]])


# In[ ]:




