import pandas as pd
import matplotlib.pyplot as plt

#reading
data=pd.read_csv('economic_data.csv')
print(data)

#drawing
plt.scatter(data['Year'], data['GDP'])

#defining y=mx+b (x,y)
x=data.iloc[:,:1]
y=data.iloc[:,1]
print(x)
print(y)
#m,b
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

print(model.coef_)
print(model.intercept_)

plt.scatter(x,y)
plt.plot(x,y,linestyle='dashed',color="blue")
plt.xlabel("Year")
plt.ylabel("GDP")
model.predict([[24]])
model.score(x,y)
