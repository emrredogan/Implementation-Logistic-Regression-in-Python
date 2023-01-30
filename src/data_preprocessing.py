import pandas as pd 
from sklearn.preprocessing import LabelEncoder

class DataPreprocessing:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def check_missing(df):
        return df.isnull().sum()

    @staticmethod
    def df_info(df):
        return df.info()

    @staticmethod
    def describe(df):
        return df.describe().T

    @staticmethod
    def control_discrete_object_feat(df):
        for feat in df.columns:
            if df[feat].dtype == 'object':
                print("\n",feat, "object :")
                print("\n" ,df[feat].value_counts())
            else:
                print("there is no object type")

    @staticmethod
    def encoding_object_feat(df):
        for feat in df.columns:
            if df[feat].dtype == 'object':
                le =LabelEncoder().fit(df[feat])
                df[feat+"_encoded"] = le.transform(df[feat])
            else:
                continue
        return df
    

    @staticmethod
    def drop_categorical_cols(df):
        categorical_col = [col for col in df.columns if df[col].dtype == 'object']
        df2 = df.drop(categorical_col, axis=1)
        return df2


    @staticmethod
    def multicollinearity_elimination(df):
        corr = df.corr(method='pearson')
        drop_col_list = []

        for i, col in enumerate(df.columns):
            highly_corr_feat = df.columns[(corr[col].abs() > 0.80) & (corr[col].index != col)]
            highly_corr_feat = list(highly_corr_feat)

            if len(drop_col_list) > 0:
                drop_col_list.extend(highly_corr_feat)
            else:
                drop_col_list = highly_corr_feat
        
        drop_col_list = list(drop_col_list)

        for column in drop_col_list:
            if column in df.columns:
                other_col = [col for col in drop_col_list if col != column][0]

                try:
                    if abs(corr['y_encoded'][column] > abs(corr['y_encoded'][other_col])):
                        df = df.drop(columns= other_col)
                    else:
                        df = df.drop(columns=column)
                except:
                    pass
        
        return df



