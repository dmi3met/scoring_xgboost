import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
import xgboost as xgb

# для оценки будем использовать нормализованный коэффициент Джини
# эта метрика позволяет минимизировать ошибки 2 рода - отказ в кредите ответственному заемщику 
# то есть глобально, она направлена на увеличение потенциальной прибыли от выданных кредитов

def norm_gini(actual, predict):
    return 2*metrics.roc_auc_score(actual, predict)-1

print('загрузка данных')
credits = pd.read_excel('credits.xlsx', header=1, index_col=0)
#print(credits.head())
#print(credits.info())
# нулевых данных нет, предобработка не нужна



# разделим датасет на обучающую и тестовую выборки
train = credits[credits.is_test == 0]
test = credits[credits.is_test == 1]

#уберём лишние столбцы, нам нужны только столбцы с первого по 'res_npl_15'
y_col = 'res_npl_15'
train = train.loc[:, :y_col]
train_y = train[y_col]
train_X = train.drop(columns=y_col)

test = test.loc[:, :y_col]
test_X = test.drop(columns=y_col)

#print(X_train, y_train)

print('готовим метрику')
gini_scorer = metrics.make_scorer(norm_gini, greater_is_better = True)


xgb_model = xgb.XGBClassifier(use_label_encoder=False)

print('подбираем лучшие параметры модели')
parameters = {'nthread':[2],
              'objective':['binary:logistic'],
              'learning_rate': [0.05],
              'max_depth': [4,5],
              'min_child_weight': [11],
              'eval_metric': ['logloss'],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [100,200],
              'seed': [42]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                    
                   scoring=gini_scorer,
                   verbose=3, refit=True)

clf.fit(train_X, train_y)
print('лучшие параметры: ', clf.best_params_)

print('готовим предсказания')
test_probs = clf.predict(test_X)
test_X[y_col] = test_probs
test_X.to_excel('Submission.xlsx')
