import pandas as pd
import numpy as np



def train_test_split(X:pd.core.frame.DataFrame,
                     y:pd.core.frame.DataFrame,
                     train_size:int = None,
                     test_size:int = None
                    ) -> tuple[pd.core.frame.DataFrame]:
    
    if train_size is test_size is None:
        raise TypeError('You must specify either train_size or test_size')
    elif train_size is None:
        train_size = 1 - test_size
    elif test_size is None:
        test_size = 1 - train_size

    indices = np.array(y.index)
    np.random.shuffle(indices)
    
    break_at = int(len(indices)*0.2)
    
    test_indices = indices[:break_at]
    train_indices = indices[break_at:]
    
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]

    # y_train and y_test are converted to DataFrames for future use when we perform arithmetical operations on the matrices.
    return (
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True).to_frame(),
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True).to_frame()
    )





def calc_zscore(feature):
    mean = np.mean(feature)
    std = np.std(feature)
        
    z_score = (feature-mean)/std
    
    return z_score





def covar(
    X: pd.core.series.Series,
    y: pd.core.series.Series
) -> int:
    covar = np.mean(X*y) - np.mean(X)*np.mean(y)
    return covar






def clean_data(X:pd.core.frame.DataFrame,
               y:pd.core.frame.DataFrame,
               threshold:float=3.5
              ) -> tuple:
    '''This function will remove outliers(if any) with the help of Z-score value.
According to National Institute of Standards and Technology, Z-scores with an absolute value of greater than 3.5 be labeled as potential outliers.'''
    
    drop_indices = np.empty(0)
    
    for feature_name in X:
        
        feature = X[feature_name]

        z_score = calc_zscore(feature)
        
        outliers = feature.loc[np.abs(z_score)>threshold]
        drop_indices = np.append(outliers.index, drop_indices)

    return X.drop(drop_indices, inplace=False).reset_index(drop=True), y.drop(drop_indices, inplace=False).reset_index(drop=True)
        





class Standarize:
    
    mean = std = None
    
    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()
        
        
    def fit_transform(self, X):
        self.fit(X)
        X_scaled = (X - self.mean)/self.std
        return X_scaled
        

    def transform(self, X):
        try:
            X_scaled = (X - self.mean)/self.std
            return X_scaled
        except TypeError:
            raise TypeError('No data has been provided to calculate mean and standard deviation')




def r2_score(
    y:np.ndarray[float],
    y_pred:np.ndarray[float]
) -> float:

    # From the formula, it is clear that mean of sum of squared residuals = 2*(minimum value of cost_func)
    RSS = np.mean((y - y_pred)**2)
    
    # from the formula, Mean of Sum of Squared of variation from mean is same as variation
    variance = covar(y, y)

    r2_score = 1 - RSS/variance
    return r2_score




def poly_transformation(degree, n_features):
     def convert(number, base, n_features):
         res = ''
         while number != 0:
             res += str(number%base)
             number //= base
    
         result = res[::-1]
         return '0'*(n_features - len(result)) + result
    
     powers = []
     for i in range((degree+1)**n_features):
         power = tuple(map(int, list(convert(i, degree+1, n_features))))
         if sum(power) <= degree:
             powers.append(power)
    
     return powers