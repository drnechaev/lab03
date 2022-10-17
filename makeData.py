import pandas as pd
from pandas.io.formats.info import DataFrameInfo
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif




def setAPPPosition(seq):
    if seq['APP_POSITION_TYPE'] is np.NAN and seq['CLNT_JOB_POSITION'] is not np.NAN:
        if seq['CLNT_JOB_POSITION'] in ['TOP_MANAGER', 'MANAGER', 'SELF_EMPL']:
            seq['APP_POSITION_TYPE'] = seq['CLNT_JOB_POSITION']
        else:
            seq['APP_POSITION_TYPE'] = 'specialist'

    return seq


def makeJobPosition(df):
    pos = df.loc[:, 'CLNT_JOB_POSITION'].str.lower()

    pos = pos.replace(['директор', 'генеральный директор', 'ген.директор', 'ген. директор', 'коммерческий директор',
                       'финансовый директор',
                       'исполнительный директор', 'заместитель директора', 'зам.директора', 'зам. директора',
                       'директор по развитию',
                       'зам.ген.директора', 'зам. ген. директора', 'комерческий директор', 'ген директор',
                       'технический директор', 'заместитель генерального директора', 'зам директора',
                       'директор по маркетингу', 'директор по продажам', 'директор филиала',
                       'зам. генерального директора', 'ком.директор', 'зам.генерального директора',
                       'гениральный директор',
                       'директор магазина', 'директор департамента', 'ком. директор',
                       'зам.директора по эк-ке и развитию',
                       'начальник отдела', 'руководитель отдела', 'руководитель отдела продаж',
                       'начальник отдела продаж',
                       'руководитель проектов', 'начальник участка', 'ведущий специалист', 'главный специалист ',
                       'главный бухгалтер', 'гл. бухгалтер',
                       'руководитель', 'главный инженер', 'руководитель группы', 'помощник руководителя', 'начальник',
                       'начальник управления',
                       'старший специалист', 'начальник отдела кадров', 'рук. отделения', 'зам. начальника отд. пто',
                       'глав.спец.отдела междунар.проектов', 'управляющий склада', 'начальник отдела розничных продаж',
                       'вед.разработчик прогр.обесп.', 'руководитель отдела по работе с кл.', 'руководитель отк',
                       'инжинер програмист', 'начальник отдела инсталяции', 'руководитель произв. отдела',
                       'рук-тель отд-а по работе с клиентам'], 'TOP_MANAGER')

    pos = pos.replace(
        ['предприниматель', 'ип', 'индивидуальный предприниматель', 'учредитель', 'индивдуальный предприниматель'],
        'SELF_EMPL')

    pos = pos.replace(
        ['менеджер по продажам', 'управляющий', 'региональный менеджер', 'старший менеджер', 'офис-менеджер',
         'супервайзер', 'менеджер по работе с клиентами', 'менеджер по персоналу', 'менеджер по рекламе',
         'страховой агент', 'ведущий менеджер', 'менеджер отдела продаж', 'менеджер проектов',
         'менеджер по закупкам', 'агент', 'специалист по продажам', 'финансовый менеджер', 'советник',
         'менеджер по лизингу', 'менеджер по вэб'], 'MANAGER')

    pos = pos.replace(['студентка'], 'none')
    df['CLNT_JOB_POSITION'] = pos


    return df.apply(setAPPPosition, axis=1)

def setMedian(df,param,method='S'):
    imp_mean = SimpleImputer(strategy='median')  # для импутации медианой замените 'mean' на 'median'
    return imp_mean.fit_transform(df[param])

def setZero(df, param):
    return df[param].fillna(0)

def setNone(df,param):
    pos = df.loc[:, param].str.lower()
    pos = pos.replace(' ','none')
    pos = pos.fillna('none')
    return pos

def makeRelation(df):
    pos = df.loc[:, 'CLNT_TRUST_RELATION'].str.lower()
    pos = pos.replace(['близкий ро', 'жена', 'муж'], 'relative')
    pos = pos.replace('друг', 'friend')
    pos = pos.replace(['мать', 'мама'], 'mother')
    pos = pos.replace('дальний ро', 'other')
    pos = pos.replace('сестра', 'sister')
    pos = pos.replace('отец', 'father')
    pos = pos.replace('брат', 'brother')
    pos = pos.replace('дочь', 'daughter')
    pos = pos.replace('сын', 'son')
    pos = pos.fillna('none')
    return pos


def makeCategorical(df):
    df['CLNT_TRUST_RELATION'] = makeRelation(df)

    df['APP_EDUCATION'] = setNone(df, 'APP_EDUCATION')
    df['APP_EMP_TYPE'] = setNone(df, 'APP_EMP_TYPE')
    df['APP_MARITAL_STATUS'] = setNone(df, 'APP_MARITAL_STATUS')
    df['APP_POSITION_TYPE'] = setNone(df, 'APP_POSITION_TYPE')

    pos = df.loc[:, 'APP_KIND_OF_PROP_HABITATION'].str.lower()
    pos = pos.fillna('other')
    df['APP_KIND_OF_PROP_HABITATION'] = pos

    pos = df.loc[:, 'APP_TRAVEL_PASS'].str.lower()
    pos = pos.fillna('n')
    df['APP_TRAVEL_PASS'] = pos

    pos = df.loc[:, 'APP_REGISTR_RGN_CODE']
    pos = pos.fillna(77)
    df['APP_REGISTR_RGN_CODE'] = pos

    return df





fill_zero_pred =["TRANS_CNT_TENDENCY3M","SUM_TRAN_ATM_TENDENCY3M","CNT_TRAN_ATM_TENDENCY3M"]

mean_transf_pred = ["AMOUNT_RUB_SUP_PRC", "TRANS_AMOUNT_TENDENCY3M", "TRANS_COUNT_SUP_PRC",
                        "AMOUNT_RUB_CLO_PRC", "TRANS_COUNT_NAS_PRC", "AMOUNT_RUB_NAS_PRC"]


if __name__ == "__main__":
    data = pd.read_csv("data.csv",  delimiter=None, header='infer')

    data = makeJobPosition(data)


    pd.set_option('display.max_rows', 2000)
    pd.set_option('display.max_info_columns', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 2000)
    pd.set_option("mode.chained_assignment", None)




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


    data_train = data[predictors]

    data_train = makeCategorical(data_train)



    data_train[fill_zero_pred] = setZero(data_train,fill_zero_pred)
    #mean_transf_pred.extend(fill_zero_pred)

    data_train[mean_transf_pred] = setMedian(data_train,mean_transf_pred)
    data_train = data_train.drop(320763)
    data_train.to_csv("lab03_n.csv",sep='\t',index=False)

