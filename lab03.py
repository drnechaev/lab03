import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_info_columns', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option("mode.chained_assignment", None)



# ALL FULL PREDICTROTOS with max non null dataset
predictors = [ "AGE", "PACK", "CR_PROD_CNT_IL", "REST_DYNAMIC_PAYM_3M", "TURNOVER_CC", "TURNOVER_PAYM", "CR_PROD_CNT_CC","REST_DYNAMIC_FDEP_3M","REST_DYNAMIC_IL_1M","REST_DYNAMIC_CUR_1M",
                   "REST_AVG_PAYM", "LDEAL_GRACE_DAYS_PCT_MED", "REST_DYNAMIC_CUR_3M", "TURNOVER_DYNAMIC_CUR_1M","REST_DYNAMIC_IL_3M", "CR_PROD_CNT_TOVR",
                   "TURNOVER_DYNAMIC_IL_3M","REST_DYNAMIC_PAYM_1M","TURNOVER_DYNAMIC_CUR_3M","CLNT_SETUP_TENOR","TURNOVER_DYNAMIC_PAYM_3M", "TURNOVER_DYNAMIC_PAYM_1M",
                   "REST_DYNAMIC_CC_1M","TURNOVER_DYNAMIC_CC_1M","REST_DYNAMIC_CC_3M","TURNOVER_DYNAMIC_CC_3M","CR_PROD_CNT_PIL","CR_PROD_CNT_CCFP","TURNOVER_DYNAMIC_IL_1M", "REST_DYNAMIC_FDEP_1M",
                   "REST_DYNAMIC_SAVE_3M", "CR_PROD_CNT_VCU", "REST_AVG_CUR",
                   "AMOUNT_RUB_CLO_PRC","AMOUNT_RUB_NAS_PRC", "AMOUNT_RUB_SUP_PRC", "TRANS_COUNT_SUP_PRC","TRANS_COUNT_NAS_PRC",
                    'APP_REGISTR_RGN_CODE',
                    'CLNT_TRUST_RELATION', 'APP_EDUCATION', 'APP_EMP_TYPE', 'APP_KIND_OF_PROP_HABITATION',
                   'APP_MARITAL_STATUS', 'APP_POSITION_TYPE','APP_TRAVEL_PASS',
                    "TRANS_AMOUNT_TENDENCY3M","TRANS_CNT_TENDENCY3M","SUM_TRAN_ATM_TENDENCY3M","CNT_TRAN_ATM_TENDENCY3M", "TARGET"]



com_predictors = ['CLNT_TRUST_RELATION','PACK', 'APP_EDUCATION', 'APP_EMP_TYPE', 'APP_KIND_OF_PROP_HABITATION',
                   'APP_MARITAL_STATUS', 'APP_POSITION_TYPE','APP_TRAVEL_PASS']



data_train = pd.read_csv("lab03_n.csv",  delimiter='\t', header='infer')

data_train = data_train[predictors]

data_learn = data_train["TARGET"]
data_train = data_train.drop(columns="TARGET")

transformer = make_column_transformer((OneHotEncoder(), com_predictors),remainder='passthrough')
transformed = transformer.fit_transform(data_train)
data_train = pd.DataFrame(transformed,columns=transformer.get_feature_names_out())


X_train, X_test, y_train, y_test = train_test_split(data_train,data_learn ,
                                   random_state=42,
                                   test_size=0.05,
                                   shuffle=True)


ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


from catboost import CatBoostClassifier

print ("Start learning ...")
cls = CatBoostClassifier(iterations=150, random_seed=45)

cls.fit(X_train, y_train)

print("Teached... \n")

acr = cls.score(X_train, y_train)
print("Accurasy for  \"{}\" is {}".format( type(cls).__name__, acr * 100))

lr_probs = cls.predict_proba(X_train)[:,1]
lr_auc = roc_auc_score(y_train, lr_probs)
print('ROC AUC(self)=%.3f' % (lr_auc))



"""
feats = {}
for feature, importance in zip(data_train.columns, cls.feature_importances_):
    feats[feature] = importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
importances = importances.sort_values(by='Gini-Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})
print(importances)
"""


acr = cls.score(X_test, y_test)
print("Accurasy for  \"{}\" is {}".format( type(cls).__name__, acr * 100))

lr_probs = cls.predict_proba(X_test)[:,1]
lr_auc = roc_auc_score(y_test, lr_probs)
print('ROC AUC(test)=%.3f' % (lr_auc))


"""

from sklearn.model_selection import KFold, cross_val_score
#scores = cross_val_score(cls, data_train, data_learn, cv=5)
#decision_tree_result = cross_validation(cls, X, encoded_y, 5)
#print(decision_tree_result)
k_folds = KFold(n_splits = 5)

scores = cross_val_score(cls, data_train, data_learn, cv = k_folds)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
"""

"""
sns.set(font_scale = 5)
sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
fig, ax = plt.subplots()
fig.set_size_inches(30,15)
sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
plt.xlabel('Importance', fontsize=25, weight = 'bold')
plt.ylabel('Features', fontsize=25, weight = 'bold')
plt.title('Feature Importance', fontsize=25, weight = 'bold')
print(plt.show())

"""


import makeData as md
test_data = pd.read_csv("test.csv",  delimiter=None, header='infer')
test_data = md.makeJobPosition(test_data)
predictors.remove("TARGET")
td = test_data[predictors]

td = md.makeCategorical(td)
td[md.fill_zero_pred] = md.setZero(td,md.fill_zero_pred)  #td[fill_zero_pred].fillna(0)
td[md.mean_transf_pred] = md.setMedian(td,md.mean_transf_pred) #imp_mean.fit_transform(td[mean_transf_pred])

transformed = transformer.transform(td)
td = pd.DataFrame(transformed,columns=transformer.get_feature_names_out())

td = ss.transform(td)

out = cls.predict_proba(td)[:,1]

print(out)

out_data = pd.DataFrame()
out_data['id'] = test_data['ID']
out_data['target'] = out

out_data.to_csv("lab03.csv",sep='\t',index=False)
