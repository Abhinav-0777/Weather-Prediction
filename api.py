import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# df = pd.DataFrame([[1,2,3,4]],
                #   columns=['a','b','c','d'])

# print(df)
# list1 = [[9,8,7,6]]
# df1 = pd.DataFrame(list1,
                #    columns = df.columns)

# print(np.sort(df1))
# print(df1)

# print(sorted(list(zip([5,4,7,2], list(df.columns)))))

# a = [(2, 'd'), (4, 'b'), (5, 'a'), (7, 'c')]

data = pd.DataFrame({
    "age":    [22, 25, 47, 52, 46, 56, 28, 30, 34, 40],
    "salary": [20000, 25000, 60000, 52000, 58000, 65000, 30000, 32000, 40000, 45000],
    "buy":    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]  # target
})
# print(list(data.columns))
arr = np.array([[1,2,3]])
print(pd.DataFrame(arr, columns=list(data.columns)))
# X = data.drop(columns = ['buy'])
# y = data['buy']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# model = LogisticRegression()

# model.fit(X_scaled,y)

# # print(model.predict([[30,35000]]))

# explainer = shap.LinearExplainer(model, X_scaled)

# input = pd.DataFrame([[30,35000]], columns=X.columns)
# input_scaled = scaler.transform(input)

# shap_values = explainer(input_scaled)

# print(shap_values.values.shape)


# a = np.array([1,2,3])
# print(a[0])

# b = ['a','b','c']
# c = np.array([[3,2,1]])
# print(sorted(list(zip(c,b)),reverse=True))
# print(pd.DataFrame(c, columns=[]))