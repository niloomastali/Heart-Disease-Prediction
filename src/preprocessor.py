import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess(X_train, X_test):
    # Impute missing values (ca and thal have '?' encoded as NaN)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Standardise all features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return X_train_scaled, X_test_scaled
