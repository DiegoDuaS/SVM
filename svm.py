from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def breif_clustering(X, n_clusters):

    X_pca = PCA(n_components=2).fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X_pca)

    X['Cluster'] = km.fit_predict(X_pca)
    centroides = km.cluster_centers_
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=X['Cluster'], palette="tab10", legend="full")
    # Plot centroids
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='X', s=200, label="Centroids")

    plt.title(f"K-Means Clustering (Reducido con PCA) - 3 Clusters")
    plt.legend()
    plt.show()
    return X

def drop_many_nulls(df):
    df = df.copy()  # Evitar modificar el dataframe original
    
    # Eliminar las variables que no queremos en el análisis de clusters
    drop_columns = [
        'Id', 'PoolArea', 'MiscVal', 'BsmtFinSF2', 'BsmtFinSF1', 'MasVnrArea',
        'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Alley', 'ExterCond',
        'BsmtHalfBath', 'KitchenAbvGr', 'PoolQC', 'Fence', 'MiscFeature', 'MiscFeature',
        'FireplaceQu', 'MasVnrType', 
    ]
    df = df.drop(columns=drop_columns, errors='ignore')
    
    return df

def trans_categorical(df):
    # Variables ordinales con asignación de valores
    ordinal_mappings = {
        'ExterQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'BsmtQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3},
    }

    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping).fillna(0)

    # Variables nominales -> Label Encoding (sin One-Hot)
    nominal_cols = [
        'LotShape', 'LotConfig', 'Neighborhood', 'BldgType',
        'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
        'Foundation', 'BsmtFinType1', 'GarageType', 'MSZoning', 'Street',
        'LandContour', 'Utilities', 'LandSlope', 'Condition1', 'Condition2', 'RoofMatl',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional',
        'SaleCondition', 'SaleType', 'GarageQual', 'GarageCond', 'CentralAir', 'PavedDrive'
    ]

    for col in nominal_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

def metrics_and_cm(y_pred, y_test):
    # Presicion
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()