import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC

# %pylab inline
# %matplotlib inline
iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
df.head()

"""Step-2: Classes of Data"""

df["target"].value_counts()

"""Step-3: Feature Selection"""

X = df.iloc[:, :-1]  # Features
y = df['target']  # Target
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)
selected_features = np.array(iris['feature_names'])[selector.get_support()]
print(f"Selected features: {selected_features}")

"""Step 4: Split Data into Training and Testing Sets"""

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
print("Rows and Columns : ",X_train.shape)

"""Step 5: Scatter plot of predicted data"""

sns.FacetGrid(df, hue="target",height=8).map(plt.scatter, "petal length (cm)", "petal width (cm)").add_legend()

"""fit a SVM model to the data"""

from sklearn import svm
model = svm.SVC(kernel='linear', C=0.1)
model.fit(iris.data, iris.target)

"""Step 6: Train and Evaluate the SVM Model"""

##  Normalize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear', C=0.1)
model.fit(X_train, y_train)
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

"""Step 7: Detailed Evaluation"""

predicted = model.predict(X_test)
print(metrics.classification_report(y_test, predicted))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, predicted))

# Visualize decision boundaries (optional, for 2D features)
def plot_decision_boundaries(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("Decision Boundaries")
    plt.show()

plot_decision_boundaries(X_test, y_test, model)
# model.score(iris.data, iris.target)