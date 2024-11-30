
import warnings

warnings.filterwarnings('ignore')

# ==============Загрузка данных===============
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.max_rows = 300

# loaded_data = pd.read_csv('dataset.csv', sep='\t')
# loaded_data = pd.read_excel('Baza_glaukoma_issled_31_08_2023.xlsx', index_col=0)
loaded_data1 = pd.read_excel('new_database_impersonal.xlsx')

# loaded_data1.to_excel('new_database_test.xlsx', index=False)

# loaded_data2 = loaded_data1.sample(frac=1)
# loaded_data2.to_excel('new_database_test1.xlsx', index=False)
#
# loaded_data3 = loaded_data2.sample(frac=1)
# loaded_data3.to_excel('new_database_test1.xlsx', index=False)
#
# loaded_data4 = loaded_data3.sample(frac=1)
# loaded_data4.to_excel('new_database_test1.xlsx', index=False)

# Вариант загруженного датасета
loaded_data = loaded_data1


# Добавление колонки с возрастом
# loaded_data['age'] = (loaded_data['date_of_examination'] - loaded_data['Date_of_Birth']) / np.timedelta64 ( 1 , 'y')
# print(loaded_data)
loaded_data = loaded_data.assign(temp_age = (loaded_data['date_of_examination'] - loaded_data['Date_of_Birth']) // np.timedelta64 ( 365 , 'D'))


# Удаление колонок
# loaded_data = loaded_data.loc[:, ~loaded_data.columns.str.contains("full_name")]

print("\nКоличество пропусков")
print(loaded_data.isna().sum())
print("\nКоличество пропусков в процентом соотношении")
for col in loaded_data.columns:
    print(f'{col}: {loaded_data[col].isna().sum() / loaded_data.shape[0] * 100:.2f}%')

# import seaborn as sns
# import matplotlib.pyplot as plt

# Отрисовка тепловой диаграммы
# plt.figure(figsize=(20,12))
# sns.heatmap(loaded_data.isna().transpose())
# plt.show()


# =====================================================================================================
# # # Удаляем строки, где все пропуски
# # loaded_data = loaded_data.dropna(how='all', axis=0)
# print("\nОставляем строки, где N значений")
# print(loaded_data.shape)
# loaded_data = loaded_data.dropna(thresh=5, axis=0)
# print(loaded_data.shape)
# =====================================================================================================

# print("\nУдаляем строку, если значение не заполнено")
# print(loaded_data.shape)
# loaded_data = loaded_data.dropna(subset=['eye'], axis=0)
# print(loaded_data.shape)


# ==============Кодирование категориальных признаков===============
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
print("\nЗакодированные категориальные признаки")
loaded_data_new = labelencoder.fit_transform(loaded_data['eye'])
loaded_data['eye'] = loaded_data_new
print("Столбец 'eye'", loaded_data_new[:20])


# =============Индикаторный метод: Создание нового признака, который помечает пропуск через Pandas=============
loaded_data['diagnosis_nan'] = 0
loaded_data['stage_nan'] = 0
loaded_data['first_nan'] = 0
loaded_data['medication_nan'] = 0
loaded_data['comparable_diagnosis_nan'] = 0
loaded_data['sex_nan'] = 0

loaded_data.loc[loaded_data['diagnosis'].isna(), 'diagnosis_nan'] = 1
loaded_data.loc[loaded_data['stage'].isna(), 'stage_nan'] = 1
loaded_data.loc[loaded_data['first'].isna(), 'first_nan'] = 1
loaded_data.loc[loaded_data['medication'].isna(), 'medication_nan'] = 1
loaded_data.loc[loaded_data['comparable_diagnosis'].isna(), 'comparable_diagnosis_nan'] = 1
loaded_data.loc[loaded_data['sex'].isna(), 'sex_nan'] = 1

loaded_data['sex'].fillna("unknown", inplace=True)
loaded_data['diagnosis'].fillna("unknown", inplace=True)
loaded_data['stage'].fillna(0, inplace=True)
loaded_data['first'].fillna(0, inplace=True)
loaded_data['medication'].fillna(0, inplace=True)
loaded_data['comparable_diagnosis'].fillna("unknown", inplace=True)
# =============================================================================================================


# ========================================классифицирование====================================================
loaded_data_new = labelencoder.fit_transform(loaded_data['full_name'])
loaded_data['full_name'] = loaded_data_new
print("Столбец 'full_name'", loaded_data_new[:20])
loaded_data_new = labelencoder.fit_transform(loaded_data['diagnosis'])
loaded_data['diagnosis'] = loaded_data_new
print("Столбец 'diagnosis'", loaded_data_new[:20])
loaded_data_new = labelencoder.fit_transform(loaded_data['stage'])
loaded_data['stage'] = loaded_data_new
print("Столбец 'stage'", loaded_data_new[:20])
loaded_data_new = labelencoder.fit_transform(loaded_data['first'])
loaded_data['first'] = loaded_data_new
print("Столбец 'first'", loaded_data_new[:20])
loaded_data_new = labelencoder.fit_transform(loaded_data['medication'])
loaded_data['medication'] = loaded_data_new
print("Столбец 'medication'", loaded_data_new[:20])
loaded_data_new = labelencoder.fit_transform(loaded_data['comparable_diagnosis'])
loaded_data['comparable_diagnosis'] = loaded_data_new
print("Столбец 'comparable_diagnosis'", loaded_data_new[:20])
loaded_data_new = labelencoder.fit_transform(loaded_data['sex'])
loaded_data['sex'] = loaded_data_new
print("Столбец 'sex'", loaded_data_new[:20])

# print("\nЗакодированные категориальные признаки")
# print(loaded_data['eye'])
# print(loaded_data['diagnosis'])
# print(loaded_data['stage'])
# print(loaded_data['first'])
# print(loaded_data['medication'])
# print(loaded_data['comparable_diagnosis'])

# =============================================================================================================


# Заполнение пропусков
print("Описание данных по колонкам:")
print(loaded_data.info())

loaded_data['eye'] = loaded_data['eye'].fillna(loaded_data['eye'].median())
# loaded_data['Date_of_Birth'] = loaded_data['Date_of_Birth'].fillna(loaded_data['Date_of_Birth'].median())                       #По идее, заполнен
# loaded_data['date_of_examination'] = loaded_data['date_of_examination'].fillna(loaded_data['date_of_examination'].median())     #По идее, заполнен
# loaded_data['diagnosis'] = loaded_data['diagnosis'].fillna(loaded_data['diagnosis'].median())
# loaded_data['stage'] = loaded_data['stage'].fillna(loaded_data['stage'].median())                                               #Вопрос
# loaded_data['first'] = loaded_data['first'].fillna(loaded_data['first'].median())                                               #Вопрос
# loaded_data['medication'] = loaded_data['medication'].fillna(loaded_data['medication'].median())                                #Вопрос
# loaded_data['comparable_diagnosis'] = loaded_data['comparable_diagnosis'].fillna(loaded_data['comparable_diagnosis'].median())  #Вопрос


# =======================================================Заполнение пропусков по медиане===============================================================
# loaded_data['cct'] = loaded_data['cct'].fillna(loaded_data['cct'].median())
# loaded_data['KRT_H'] = loaded_data['KRT_H'].fillna(loaded_data['KRT_H'].median())
# loaded_data['KRT_V'] = loaded_data['KRT_V'].fillna(loaded_data['KRT_V'].median())
# loaded_data['ORA_IOPcc'] = loaded_data['ORA_IOPcc'].fillna(loaded_data['ORA_IOPcc'].median())
# loaded_data['ORA_IOPg'] = loaded_data['ORA_IOPg'].fillna(loaded_data['ORA_IOPg'].median())
# loaded_data['ORA_CH'] = loaded_data['ORA_CH'].fillna(loaded_data['ORA_CH'].median())
# loaded_data['ORA_CRF'] = loaded_data['ORA_CRF'].fillna(loaded_data['ORA_CRF'].median())
# loaded_data['Perimetry_MD'] = loaded_data['Perimetry_MD'].fillna(loaded_data['Perimetry_MD'].median())
# loaded_data['Perimetry_SLV'] = loaded_data['Perimetry_SLV'].fillna(loaded_data['Perimetry_SLV'].median())
# loaded_data['Perimetry_DD'] = loaded_data['Perimetry_DD'].fillna(loaded_data['Perimetry_DD'].median())
# loaded_data['Perimetry_LD'] = loaded_data['Perimetry_LD'].fillna(loaded_data['Perimetry_LD'].median())
# loaded_data['OCT_RNFL_total'] = loaded_data['OCT_RNFL_total'].fillna(loaded_data['OCT_RNFL_total'].median())
# loaded_data['OCT_RNFL_superior'] = loaded_data['OCT_RNFL_superior'].fillna(loaded_data['OCT_RNFL_superior'].median())
# loaded_data['OCT_RNFL_inferior'] = loaded_data['OCT_RNFL_inferior'].fillna(loaded_data['OCT_RNFL_inferior'].median())
# loaded_data['OCT_Disc_area'] = loaded_data['OCT_Disc_area'].fillna(loaded_data['OCT_Disc_area'].median())
# loaded_data['OCT_Cup_area'] = loaded_data['OCT_Cup_area'].fillna(loaded_data['OCT_Cup_area'].median())
# loaded_data['OCT_Rim_area'] = loaded_data['OCT_Rim_area'].fillna(loaded_data['OCT_Rim_area'].median())
# loaded_data['OCT_C/D_area_ratio'] = loaded_data['OCT_C/D_area_ratio'].fillna(loaded_data['OCT_C/D_area_ratio'].median())
# loaded_data['OCT_Linear_CDR'] = loaded_data['OCT_Linear_CDR'].fillna(loaded_data['OCT_Linear_CDR'].median())
# loaded_data['OCT_Vertical_CDR'] = loaded_data['OCT_Vertical_CDR'].fillna(loaded_data['OCT_Vertical_CDR'].median())
# loaded_data['OCT_Cup_volume'] = loaded_data['OCT_Cup_volume'].fillna(loaded_data['OCT_Cup_volume'].median())
# loaded_data['OCT_Rim_volume'] = loaded_data['OCT_Rim_volume'].fillna(loaded_data['OCT_Rim_volume'].median())
# loaded_data['OCT_Horizontal_D.D'] = loaded_data['OCT_Horizontal_D.D'].fillna(loaded_data['OCT_Horizontal_D.D'].median())
# loaded_data['OCT_Vertical_D.D.'] = loaded_data['OCT_Vertical_D.D.'].fillna(loaded_data['OCT_Vertical_D.D.'].median())
# loaded_data['OCT_GCL+ST'] = loaded_data['OCT_GCL+ST'].fillna(loaded_data['OCT_GCL+ST'].median())
# loaded_data['OCT_GCL+S'] = loaded_data['OCT_GCL+S'].fillna(loaded_data['OCT_GCL+S'].median())
# loaded_data['OCT_GCL+SN'] = loaded_data['OCT_GCL+SN'].fillna(loaded_data['OCT_GCL+SN'].median())
# loaded_data['OCT_GCL+IN'] = loaded_data['OCT_GCL+IN'].fillna(loaded_data['OCT_GCL+IN'].median())
# loaded_data['OCT_GCL+I'] = loaded_data['OCT_GCL+I'].fillna(loaded_data['OCT_GCL+I'].median())
# loaded_data['OCT_GCL+IT'] = loaded_data['OCT_GCL+IT'].fillna(loaded_data['OCT_GCL+IT'].median())

# =======================================================Заполнение пропусков по 0 и "unknown"===============================================================
loaded_data['cct_nan'] = 0
loaded_data['KRT_H_nan'] = 0
loaded_data['KRT_V_nan'] = 0
loaded_data['ORA_IOPcc_nan'] = 0
loaded_data['ORA_IOPg_nan'] = 0
loaded_data['ORA_CH_nan'] = 0
loaded_data['ORA_CRF_nan'] = 0
loaded_data['Perimetry_MD_nan'] = 0
loaded_data['Perimetry_SLV_nan'] = 0
loaded_data['Perimetry_DD_nan'] = 0
loaded_data['Perimetry_LD_nan'] = 0
loaded_data['OCT_RNFL_total_nan'] = 0
loaded_data['OCT_RNFL_superior_nan'] = 0
loaded_data['OCT_RNFL_inferior_nan'] = 0
loaded_data['OCT_Disc_area_nan'] = 0
loaded_data['OCT_Cup_area_nan'] = 0
loaded_data['OCT_Rim_area_nan'] = 0
loaded_data['OCT_C/D_area_ratio_nan'] = 0
loaded_data['OCT_Linear_CDR_nan'] = 0
loaded_data['OCT_Vertical_CDR_nan'] = 0
loaded_data['OCT_Cup_volume_nan'] = 0
loaded_data['OCT_Rim_volume_nan'] = 0
loaded_data['OCT_Horizontal_D.D_nan'] = 0
loaded_data['OCT_Vertical_D.D._nan'] = 0
loaded_data['OCT_GCL+ST_nan'] = 0
loaded_data['OCT_GCL+S_nan'] = 0
loaded_data['OCT_GCL+SN_nan'] = 0
loaded_data['OCT_GCL+IN_nan'] = 0
loaded_data['OCT_GCL+I_nan'] = 0
loaded_data['OCT_GCL+IT_nan'] = 0

loaded_data.loc[loaded_data['cct'].isna(), 'cct_nan'] = 1
loaded_data.loc[loaded_data['KRT_H'].isna(), 'KRT_H_nan'] = 1
loaded_data.loc[loaded_data['KRT_V'].isna(), 'KRT_V_nan'] = 1
loaded_data.loc[loaded_data['ORA_IOPcc'].isna(), 'ORA_IOPcc_nan'] = 1
loaded_data.loc[loaded_data['ORA_IOPg'].isna(), 'ORA_IOPg_nan'] = 1
loaded_data.loc[loaded_data['ORA_CH'].isna(), 'ORA_CH_nan'] = 1
loaded_data.loc[loaded_data['ORA_CRF'].isna(), 'ORA_CRF_nan'] = 1
loaded_data.loc[loaded_data['Perimetry_MD'].isna(), 'Perimetry_MD_nan'] = 1
loaded_data.loc[loaded_data['Perimetry_SLV'].isna(), 'Perimetry_SLV_nan'] = 1
loaded_data.loc[loaded_data['Perimetry_DD'].isna(), 'Perimetry_DD_nan'] = 1
loaded_data.loc[loaded_data['Perimetry_LD'].isna(), 'Perimetry_LD_nan'] = 1
loaded_data.loc[loaded_data['OCT_RNFL_total'].isna(), 'OCT_RNFL_total_nan'] = 1
loaded_data.loc[loaded_data['OCT_RNFL_superior'].isna(), 'OCT_RNFL_superior_nan'] = 1
loaded_data.loc[loaded_data['OCT_RNFL_inferior'].isna(), 'OCT_RNFL_inferior_nan'] = 1
loaded_data.loc[loaded_data['OCT_Disc_area'].isna(), 'OCT_Disc_area_nan'] = 1
loaded_data.loc[loaded_data['OCT_Cup_area'].isna(), 'OCT_Cup_area_nan'] = 1
loaded_data.loc[loaded_data['OCT_Rim_area'].isna(), 'OCT_Rim_area_nan'] = 1
loaded_data.loc[loaded_data['OCT_C/D_area_ratio'].isna(), 'OCT_C/D_area_ratio_nan'] = 1
loaded_data.loc[loaded_data['OCT_Linear_CDR'].isna(), 'OCT_Linear_CDR_nan'] = 1
loaded_data.loc[loaded_data['OCT_Vertical_CDR'].isna(), 'OCT_Vertical_CDR_nan'] = 1
loaded_data.loc[loaded_data['OCT_Cup_volume'].isna(), 'OCT_Cup_volume_nan'] = 1
loaded_data.loc[loaded_data['OCT_Rim_volume'].isna(), 'OCT_Rim_volume_nan'] = 1
loaded_data.loc[loaded_data['OCT_Horizontal_D.D'].isna(), 'OCT_Horizontal_D.D_nan'] = 1
loaded_data.loc[loaded_data['OCT_Vertical_D.D.'].isna(), 'OCT_Vertical_D.D._nan'] = 1
loaded_data.loc[loaded_data['OCT_GCL+ST'].isna(), 'OCT_GCL+ST_nan'] = 1
loaded_data.loc[loaded_data['OCT_GCL+S'].isna(), 'OCT_GCL+S_nan'] = 1
loaded_data.loc[loaded_data['OCT_GCL+SN'].isna(), 'OCT_GCL+SN_nan'] = 1
loaded_data.loc[loaded_data['OCT_GCL+IN'].isna(), 'OCT_GCL+IN_nan'] = 1
loaded_data.loc[loaded_data['OCT_GCL+I'].isna(), 'OCT_GCL+I_nan'] = 1
loaded_data.loc[loaded_data['OCT_GCL+IT'].isna(), 'OCT_GCL+IT_nan'] = 1

loaded_data['cct'].fillna(0, inplace=True)
loaded_data['KRT_H'].fillna(0, inplace=True)
loaded_data['KRT_V'].fillna(0, inplace=True)
loaded_data['ORA_IOPcc'].fillna(0, inplace=True)
loaded_data['ORA_IOPg'].fillna(0, inplace=True)
loaded_data['ORA_CH'].fillna(0, inplace=True)
loaded_data['ORA_CRF'].fillna(0, inplace=True)
loaded_data['Perimetry_MD'].fillna(0, inplace=True)
loaded_data['Perimetry_SLV'].fillna(0, inplace=True)
loaded_data['Perimetry_DD'].fillna(0, inplace=True)
loaded_data['Perimetry_LD'].fillna(0, inplace=True)
loaded_data['OCT_RNFL_total'].fillna(0, inplace=True)
loaded_data['OCT_RNFL_superior'].fillna(0, inplace=True)
loaded_data['OCT_RNFL_inferior'].fillna(0, inplace=True)
loaded_data['OCT_Disc_area'].fillna(0, inplace=True)
loaded_data['OCT_Cup_area'].fillna(0, inplace=True)
loaded_data['OCT_Rim_area'].fillna(0, inplace=True)
loaded_data['OCT_C/D_area_ratio'].fillna(0, inplace=True)
loaded_data['OCT_Linear_CDR'].fillna(0, inplace=True)
loaded_data['OCT_Vertical_CDR'].fillna(0, inplace=True)
loaded_data['OCT_Cup_volume'].fillna(0, inplace=True)
loaded_data['OCT_Rim_volume'].fillna(0, inplace=True)
loaded_data['OCT_Horizontal_D.D'].fillna(0, inplace=True)
loaded_data['OCT_Vertical_D.D.'].fillna(0, inplace=True)
loaded_data['OCT_GCL+ST'].fillna(0, inplace=True)
loaded_data['OCT_GCL+S'].fillna(0, inplace=True)
loaded_data['OCT_GCL+SN'].fillna(0, inplace=True)
loaded_data['OCT_GCL+IN'].fillna(0, inplace=True)
loaded_data['OCT_GCL+I'].fillna(0, inplace=True)
loaded_data['OCT_GCL+IT'].fillna(0, inplace=True)

# ======================================================================================================================
# primer zapolnenia propuska
# loaded_data['cct_nan'] = 0
# loaded_data.loc[loaded_data['cct'].isna(), 'cct_nan'] = 1
# loaded_data['cct'].fillna(0, inplace=True)

# Удаление колонок
loaded_data = loaded_data.loc[:, ~loaded_data.columns.str.contains("Date_of_Birth")]
loaded_data = loaded_data.loc[:, ~loaded_data.columns.str.contains("date_of_examination")]

print("\nКоличество пропусков")
print(loaded_data.isna().sum())
# print("\nПолучившиеся данные")
# print(loaded_data)

# Отрисовка тепловой диаграммы
# plt.figure(figsize=(20,12))
# sns.heatmap(loaded_data.isna().transpose())
# plt.show()
loaded_data.to_excel('new_database_not_null.xlsx')

# ==============Разделение данных на обучающую и тестовую===============
# Создание целевого значения
y_data = loaded_data['diagnosis']
y_data.to_excel('y_data.xlsx')

# Удаление колонок
# cols = [x for x in loaded_data.columns if not "diagnosis" in x]
# x_data = loaded_data[cols]
x_data = loaded_data.loc[:, ~loaded_data.columns.str.contains("diagnosis")]
x_data.to_excel('x_data.xlsx')

print("\nВывод целевого значения")
print(x_data)
print(y_data)

from sklearn.model_selection import train_test_split
# train_test_split разбивает выборку случайныи образом на 2 подвыборки: обучающую и тестовую
# запускам разбиение с параметрами по умолчанию




#
# Кроме того, мы можем явно задать некоторые специальные параметры
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.33, random_state=0, shuffle=False)
x_train.to_excel('x_train.xlsx')
x_test.to_excel('x_test.xlsx')
y_train.to_excel('y_train.xlsx')
y_test.to_excel('y_test.xlsx')

# ==============Стандартизация===============
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train_std = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
x_test_std = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

# ==============Нормализация===============
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(x_train)
normalized_x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
normalized_x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

# print('До нормализации:')
# print(x_test[:3])
# print()
# print('После нормализации:')
# print(normalized_x_test[:3])


# ==============Загрузка и настройка моделей===============
from sklearn.naive_bayes import GaussianNB


# print("x_train y_train")
# print(x_train)
# print("========")
# print(y_train)

print("\n\n======= Обучение и предсказание =======")
# ===========================================ЗАДАЧА РЕГРЕССИИ=========================================
# Линейная регрессия
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
lr_model_y_predict = lr_model.predict(x_test)
# print('\nPredict LinearRegression: ', lr_model_y_predict)
# print(type(lr_model_y_predict))
# print("========================================")
# print('Y_test?: ', y_test)
# print(type(y_test))
# # Оценка модели внутренним методом score
# print("\nМетод score модели lr_model")
# print(lr_model.score(x_test, y_test))

# Ridge
from sklearn.linear_model import Ridge
ridge_model = Ridge()
ridge_model.fit(x_train,y_train)
ridge_model_y_predict = ridge_model.predict(x_test)
# print('\nPredict Ridge: ', ridge_model_y_predict)
# print(type(ridge_model_y_predict))
# print("========================================")
# print('Y_test?: ', y_test)
# print(type(y_test))
# print("\nМетод score модели ridge_model")
# print(ridge_model.score(x_test, y_test))

# Lasso
from sklearn.linear_model import Lasso
lasso_model = Lasso()
lasso_model.fit(x_train,y_train)
lasso_model_y_predict = lasso_model.predict(x_test)
# print('\nPredict Lasso: ', lasso_model_y_predict)
# print(type(lasso_model_y_predict))
# print("========================================")
# print('Y_test?: ', y_test)
# print(type(y_test))
# print("\nМетод score модели lasso_model")
# print(lasso_model.score(x_test, y_test))


# ElasticNet
from sklearn.linear_model import ElasticNet
elasticNet_model = ElasticNet()
elasticNet_model.fit(x_train,y_train)
elasticNet_model_y_predict = elasticNet_model.predict(x_test)
# print('\nPredict ElasticNet: ', elasticNet_model_y_predict)
# print(type(elasticNet_model_y_predict))
# print("========================================")
# print('Y_test?: ', y_test)
# print(type(y_test))
# print("\nМетод score модели elasticNet_model")
# print(elasticNet_model.score(x_test, y_test))
# ==================================================================================================



# ===================================ЗАДАЧА КЛАССИФИКАЦИИ===========================================
# Метод опорных векторов
from sklearn.svm import SVC
svc_model = SVC(kernel = 'linear', degree=11, coef0=60) # kernel указывает тип ядра ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
svc_model.fit(normalized_x_train, y_train)
svc_model_y_predict = svc_model.predict(normalized_x_test)
print('\nPredict SVC: ', svc_model_y_predict)
print(type(svc_model_y_predict))
print("========================================")
print('Y_test?: ', y_test)
print(type(y_test))
print("\nМетод score модели svc_model")
print(svc_model.score(normalized_x_test, y_test))
#
#
# Метод опорных векторов. Похож на SVC с параметром kernel='linear', но реализован в liblinear, а не libsvm
from sklearn.svm import LinearSVC
linear_svc_model = LinearSVC(tol=1e-5)
linear_svc_model.fit(normalized_x_train, y_train)
linear_svc_model_y_predict = linear_svc_model.predict(normalized_x_test)
print('\nPredict LinearSVC: ', linear_svc_model_y_predict)
print(type(linear_svc_model_y_predict))
print("========================================")
print('Y_test?: ', y_test)
print(type(y_test))
print("\nМетод score модели linearSVC_model")
print(linear_svc_model.score(normalized_x_test, y_test))
#
#
# # Наивный байесовский классификатор
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(x_train, y_train)
gnb_model_y_predict = gnb_model.predict(x_test)
print('\nPredict gnb: ', gnb_model_y_predict)
print(type(gnb_model_y_predict))
print("========================================")
print('Y_test?: ', y_test)
print(type(y_test))
print("\nМетод score модели gnb_model")
print(gnb_model.score(x_test, y_test))


# Метод k-ближайших соседей
from sklearn import neighbors
knn_model = neighbors.KNeighborsClassifier()
knn_model.fit(normalized_x_train, y_train)
knn_model_y_predict = knn_model.predict(normalized_x_test)
print('\nPredict knn: ', knn_model_y_predict)
print(type(knn_model_y_predict))
print("========================================")
print('Y_test?: ', y_test)
print(type(y_test))
print("\nМетод score модели knn_model")
print(knn_model.score(normalized_x_test, y_test))
# ==================================================================================================




# ==============Оценка качества модели===============
y_predicted_data = knn_model_y_predict

print("\n\n======= Метрики оценки качества =======")

#convert series to NumPy array
y_test = y_test.to_numpy()

# ------Для классификации------
from sklearn.metrics import accuracy_score
print('(нормализация; тестовая выборка: 33%; заполнение через "0")\nМетрика accuracy_score для оценки качества модели')
print(accuracy_score(y_test, y_predicted_data))
print()

from sklearn.metrics import balanced_accuracy_score
print('========\nМетрика balanced_accuracy_score для оценки качества модели')
print(balanced_accuracy_score(y_test, y_predicted_data))
print()

from sklearn.metrics import f1_score
print('========\nМетрика f1_score для оценки качества модели "macro"')
print(f1_score(y_test, y_predicted_data, average="macro")) # f1 f1_micro f1_macro f1_weighted f1_samples   average{‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None, default=’binary’
print()

# print('Метрика f1_score для оценки качества модели "micro"')
# print(f1_score(y_test, y_predicted_data, average='micro')) # f1 f1_micro f1_macro f1_weighted f1_samples   average{‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None, default=’binary’
# print()
#
# print('Метрика f1_score для оценки качества модели "weighted"')
# print(f1_score(y_test, y_predicted_data, average='weighted')) # f1 f1_micro f1_macro f1_weighted f1_samples   average{‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None, default=’binary’
# print()

from sklearn.metrics import precision_score
print('========\nМетрика precision_score для оценки качества модели "macro"')
print(precision_score(y_test, y_predicted_data, average='macro'))
print()

# print('Метрика precision_score для оценки качества модели')
# print(precision_score(y_test, y_predicted_data, average='weighted'))
# print()

from sklearn.metrics import recall_score
print('======\nМетрика recall_score для оценки качества модели')
print(recall_score(y_test, y_predicted_data, average='macro'))
print()

# -----Для регрессии-----
# from sklearn.metrics import mean_absolute_error
# print('Метрика mean_absolute_error для оценки качества модели')
# print(mean_absolute_error(y_test, y_predicted_data))
# print()
#
# from sklearn.metrics import mean_squared_error
# print('Метрика mean_squared_error для оценки качества модели')
# print(mean_squared_error(y_test, y_predicted_data))
# print()
#
# from sklearn.metrics import r2_score
# print('Метрика r2_score для оценки качества модели')
# print(r2_score(y_test, y_predicted_data))
# print()



# =====================Describe()=======================
# Стандартизация
print(x_train_std.describe().to_excel('describe_x_train_std.xlsx'))
print()
# Нормализация
print(normalized_x_train.describe().to_excel('describe_normalized_x_train.xlsx'))
# ======================================================


# ==============Кросс-валидация===============
from sklearn.model_selection import cross_validate

print('\nCross Validation LinearSVC_model:')
results = cross_validate(linear_svc_model, x_data, y_data, cv=3)
print(results['test_score'])
#
# print('\nCross Validation ridge_model:')
# results = cross_validate(ridge_model, x_data, y_data, cv=3)
# print(results['test_score'])
#
# print('\nCross Validation lasso_model:')
# results = cross_validate(lasso_model, x_data, y_data, cv=3)
# print(results['test_score'])
#
# print('\nCross Validation elasticNet_model:')
# results = cross_validate(elasticNet_model, x_data, y_data, cv=3)
# print(results['test_score'])
#
#
#
# print('\nCross Validation svc_model:')
# results = cross_validate(svc_model, x_data, y_data, cv=3)
# print(results['test_score'])
#
# print('\nCross Validation gnb_model:')
# results = cross_validate(gnb_model, x_data, y_data, cv=3)
# print(results['test_score'])
#
# print('\nCross Validation knn_model:')
# results = cross_validate(knn_model, x_data, y_data, cv=3)
# print(results['test_score'])











# ВЫВОД
# # Посмотрим на размер данных (количество строк, колонок):
# print("Размер таблицы:", loaded_data.shape)
# print('=====================================================================================')
# # describe all the columns
# print("Описание данных в строках:")
# print(loaded_data.describe(include = "all"))
# print('=====================================================================================')
# # look at the info
# print("Описание данных по колонкам:")
# print(loaded_data.info())
# print('=====================================================================================')
#
# # Evaluating for Missing Data
# print("Незаполненные значения:")
# missing_data = loaded_data.isnull()
# print(missing_data)
# print('=====================================================================================')
# print(missing_data.isnull().sum())




#
#
#
# # from sklearn.preprocessing import Binarizer
# # X = 10 * np.random.random((5,5)) - 5
# # binarizer = Binarizer(threshold = 0.0).fit(X) # в данном случае порог 0.0
# # binary_X = binarizer.transform(X)
# # print('До бинаризации:')
# # print(X[:5])
# # print()
# # print('После бинаризации:')
# # binary_X[:5]
#
#
#
#
# from sklearn.linear_model._perceptron import Perceptron
#
# model = Perceptron().fit(x_train, y_train)





