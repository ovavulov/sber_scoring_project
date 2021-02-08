# НАБОР ФУНКЦИЙ ДЛЯ ПОСТРОЕНИЯ МОДЕЛИ


def getConfig(path="."):
    import json, os
    return json.load(open(os.path.join(path, "config.json"), 'r'))


def getPath(config, to):
    """
    Вывод путей к папкам проекта
    """
    import os
    home_path = os.path.expanduser("~")
    assert to in ["data", "test", "reports", "selections", "results", "catcols"]

    # директория загрузки данных
    if to == "data":
        return os.path.join(home_path, "datasets", config["MODEL_ID"]) if config["PATH"]["DATA"] == "DEFAULT" \
            else config["PATH"]["DATA"]

    # директория загрузки тестовых данных
    elif to == "test":
        return os.path.join(home_path, ''.join([config["MODEL_NO"], "_", config["MODEL_ID"]]), "test") if \
        config["PATH"]["TEST"] == "DEFAULT" \
            else config["PATH"]["TEST"]

    # директория сохранения промежуточных отчётов
    elif to == "reports":
        return os.path.join(home_path, ''.join([config["MODEL_NO"], "_", config["MODEL_ID"]]), "reports") if \
        config["PATH"]["REPORTS"] == "DEFAULT" \
            else config["PATH"]["REPORTS"]

    elif to == "selections":
        return os.path.join(home_path, ''.join([config["MODEL_NO"], "_", config["MODEL_ID"]]), "selections") if \
        config["PATH"]["RESULTS"] == "DEFAULT" \
            else config["PATH"]["RESULTS"]

    # директория сохранения результатов моделирования
    elif to == "results":
        return os.path.join(home_path, ''.join([config["MODEL_NO"], "_", config["MODEL_ID"]]), "results") if \
        config["PATH"]["RESULTS"] == "DEFAULT" \
            else config["PATH"]["RESULTS"]

    # директория со списком категориальных признаков
    elif to == "catcols":
        return os.path.join(home_path, 'feature_engineering') if config["PATH"]["RESULTS"] == "DEFAULT" \
            else config["PATH"]["CATCOLS"]


def setProject(config):
    """
    Создание необходимыех папок
    """
    import os

    # директория загрузки данных
    data_path = getPath(config, to="data")
    if not os.path.exists(data_path): os.mkdir(data_path)

    # директория загрузки тестовых данных
    test_path = getPath(config, to="test")
    if not os.path.exists(test_path): os.mkdir(test_path)

    # директория сохранения промежуточных отчётов
    reports_path = getPath(config, to="reports")
    if not os.path.exists(reports_path): os.mkdir(reports_path)

    # директория сохранения результатов отбора признаков
    selections_path = getPath(config, to="selections")
    if not os.path.exists(selections_path): os.mkdir(selections_path)

    # директория сохранения результатов моделирования
    results_path = getPath(config, to="results")
    if not os.path.exists(results_path): os.mkdir(results_path)


def washProject(config):
    """
    Очистить все артефакты в проекте
    """
    import os
    if input("Are you sure you want to wash this project? [y/n]:").strip() == 'y':
        for folder in ["test"]:
            path = getPath(config, to=folder)
            files = os.listdir(path)
            for file in files:
                os.system("".join(["rm -r ", os.path.join(path, file)]))


def getReportDate(string):
    """
    Выводит подходящую отчётную дату для сбора признаков по строковому описанию периода наблюдения
    """
    import calendar
    from datetime import date
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    monthDict = {months[i]: i+1 for i in range(len(months))}
    month_str, year_str = string.split("/")
    month = monthDict[month_str.lower()] - 2 + 12*(monthDict[month_str.lower()] < 3)
    year = 2000 + int(year_str) - (monthDict[month_str.lower()] < 3)
    day = calendar.monthrange(year, month)[1]
    return str(date(year, month, day))


def makeDataReport(config, data, long, mode):
    """
    Построение отчёта по данным или модели
    """
    import pandas, os
    suffix = config["VERSION_ID"]
    report = pandas.DataFrame(
        columns=['volume', 'n_response', 'trg_rate'],
        index=[str(x)[:-9] for x in data.groupby(by='report_dt').indices.keys()]
    )
    report['trg_rate'] = data.groupby(by='report_dt')['trg_flag'].mean().values
    report['volume'] = data.groupby(by='report_dt')['trg_flag'].count().values
    report['n_response'] = data.groupby(by='report_dt')['trg_flag'].sum().values
    report_ = report.T
    report_['TOTAL'] = [
        len(data), data['trg_flag'].sum(), data.trg_flag.mean()
    ]
    report = report_.T
    report.volume = report.volume.apply(int)
    report.n_response = report.n_response.apply(int)
    reportName = f"""data{"_long" if long else ""}_report.csv"""
    if mode == "TEST":
        report.to_csv(os.path.join(getPath(config, to="test"), reportName))
    else:
        report.to_csv(os.path.join(getPath(config, to="reports"), reportName))


def makeTest(config, long=True):
    """
    Создание тестового набора данных для отладки кода построения модели
    """
    from string import ascii_uppercase as letters
    from sklearn.datasets import make_classification
    import pandas, numpy, os
    suffix = config["VERSION_ID"]
    pickleName = f"""{config["MODEL_ID"]}_data{"_long" if long else ""}{("_" + suffix) if suffix else ""}.pkl"""
    X, y = make_classification(
        weights=(1 - config["TESTPARAMS"]["WEIGHT"],), **config["TESTPARAMS"]["DICT"]
    )
    cols = ['feature_%i' % (i + 1) for i in range(config["TESTPARAMS"]["DICT"]["n_features"])]
    client_dk = numpy.random.randint(1e6, 1e8, size=(config["TESTPARAMS"]["DICT"]["n_samples"], 1))
    ootMonth = list(map(getReportDate, config["OOT"].split()))[0]
    ootPart = config["TESTPARAMS"]["WEIGHT"]
    dates = [
        "9999-01-31", "9999-02-28", "9999-03-31", "9999-04-30", "9999-05-31", "9999-06-30",
        "9999-07-31", "9999-08-31", "9999-09-30", "9999-10-31", "9999-11-30", "9999-12-31"
    ]
    dates = dates[:config["TESTPARAMS"]["NDATES"]]
    report_dt = numpy.random.choice(
        dates + [ootMonth],
        size=(config["TESTPARAMS"]["DICT"]["n_samples"], 1),
        p=[(1 - ootPart)/len(dates)]*len(dates) + [ootPart]
    )
    test = pandas.DataFrame(
        data=numpy.concatenate([client_dk, report_dt, y.reshape(config["TESTPARAMS"]["DICT"]["n_samples"], 1), X],
                               axis=1)
        , columns=config["IDCOLUMNS"].split() + ["trg_flag"] + cols
    )
    for col in cols:
        test[col] = test[col].astype(float)
    for i in range(config["TESTPARAMS"]["NCAT"]):
        catvalues = numpy.random.choice([l for l in letters]+[numpy.nan], size=len(test))
        cattitle = f"catfeature_{i+1}_cd"
        test[cattitle] = catvalues

    test.to_pickle(os.path.join(getPath(config, to="test"), pickleName))


def getData(config, long=True, mode="TEST", verb="TRUE"):
    """
    Загрузка данных
    """
    import teradatasql, os, sys, pandas, pickle
    verbose = verb == "TRUE"
    suffix = config["VERSION_ID"]
    pickleName = f"""{config["MODEL_ID"]}_data{"_long" if long else ""}{("_" + suffix) if suffix else ""}.pkl"""
    if mode == "TEST":
        if not os.path.exists(os.path.join(getPath(config, to="test"), pickleName)):
            makeTest(config, long=long)
        if verbose: print("\n\nSynthetic data has been created\n\n")
        return pandas.read_pickle(os.path.join(getPath(config, to="test"), pickleName))
    try:
        with open(os.path.join(getPath(config, to="data"), pickleName), 'rb') as f:
            up = pickle.Unpickler(f)
            data = up.load()
        if verbose: print("\n\nData has been loaded from bran storage\n\n")
    except FileNotFoundError:
        home_path = os.path.expanduser("~")
        sys.path.insert(0, home_path)
        sys.path.insert(0, os.path.join(home_path, "mpp_toolbox"))
        from mpp_toolbox.account import Account
        acc = Account()
        tableName = f"""sbx_retail_mp_ds.sbol_banners_{config["MODEL_ID"]}_data{"_long" if long else ""}"""
        query = f"select * from {tableName}"
        with teradatasql.connect(host=acc.host, user=acc.usr, password=acc.pwd) as connection:
            data = pandas.read_sql(query, connection)
        with open(os.path.join(getPath(config, to="data"), pickleName), 'wb') as f:
            pickle.dump(data, f, protocol=4)
        makeDataReport(config, data=data, long=long, mode=mode)
        if verbose: print("\n\nData has been loaded from Teradata\n\n")

    return data


def getCatIndexes(config, columns):
    """
    Вывод индексов категориальных признаков по списку имён полей
    """
    import os
    columns = list(columns)
    with open(os.path.join(getPath(config, to="catcols"), 'cat_columns.txt'), 'r') as f:
        knowledge_cats = [x.strip() for x in f.readlines()]
    cat_columns_idxs = [
        i for i in range(len(columns))
        if columns[i].endswith('cd') or columns[i].endswith('nflag') \
           or columns[i].endswith('group') or columns[i].endswith('segm') \
           or columns[i] in knowledge_cats
    ]
    return cat_columns_idxs


def preprocessing(config, dataset, verb="TRUE", lgb=True):
    """
    Подготовка датасетов для подачи в бустинг, выделение категориальных и числовых признаков
    """
    import numpy
    from tqdm import tqdm
    verbose = verb == "TRUE"
    # устаревшие атрибуты
    cred_columns = [col for col in dataset.columns if col.startswith('cred_')]
    drop_columns = [col for col in dataset.columns if col in config["DROPCOLUMNS"].split()]
    df = dataset.drop(columns=cred_columns+drop_columns)
    # выделим индексы категориальных признаков
    cat_columns_idxs = getCatIndexes(config, df.columns)
    num_columns_idxs = list(set(range(len(df.columns))) - set(cat_columns_idxs) - set(config["IDCOLUMNS"].split()))

    # приводим пропуски в категориях к строковому типу и float к int в dummy-переменных (catboost)
    # def catboost_transformer(x):
    #     if str(x) == 'nan' or x is None:
    #         return 'nan'
    #     else:
    #         return int(x) if type(x) is float else x

    for col in tqdm(df.columns[cat_columns_idxs]) if verbose else df.columns[cat_columns_idxs]:
        # df[col] = df[col].apply(lambda x: catboost_transformer(x))
        df[col] = df[col].astype(str)
    if lgb:
        def lightgbm_transformer(x):
            return numpy.nan if x is None else x

        for col in tqdm(df.columns[num_columns_idxs]) if verbose else df.columns[num_columns_idxs]:
            df[col] = df[col].apply(lambda x: lightgbm_transformer(x))
    return df, [x - 1 for x in num_columns_idxs], [x - 1 for x in cat_columns_idxs]


def makePools(config, data, verb="TRUE", lgb=True):
    """
    Разделение данных на train/val/test/oot + предобработка
    """
    from sklearn.model_selection import train_test_split
    verbose = verb == "TRUE"
    if verbose: print('\nData preprocessing is in progress...\n\n')
    dev_data = data[data.report_dt.apply(
        lambda x:
            str(x) not in list(map(getReportDate, config["OOT"].split())) if config["DEV"] == "ALL"
            else str(x) in list(map(getReportDate, config["DEV"].split()))
    )]
    oot_data = data[data.report_dt.apply(lambda x: str(x) in list(map(getReportDate, config["OOT"].split())))]
    dates = list(map(str, data.report_dt.value_counts().index))
    if verbose:
        print(f"""\nDEV DATES: {
            " ".join(sorted(set(dates) - set(map(getReportDate, config["OOT"].split())))) 
            if config["DEV"] == "ALL" else " ".join(map(getReportDate, config["DEV"].split()))
        }\n""")
    df_dev, num_columns_idxs, cat_columns_idxs = preprocessing(config, dev_data, verb=verb, lgb=lgb)
    if verbose: print('\nDEV data is ready\n\n')
    if verbose:
        print(f"""\nOOT DATES: {" ".join(map(getReportDate, config["OOT"].split()))}\n""")
    df_oot, num_columns_idxs, cat_columns_idxs = preprocessing(config, oot_data, verb=verb, lgb=lgb)
    if verbose: print('\nOOT data is ready\n\n')
    if verbose: print('Total categorical features number:', len(cat_columns_idxs), '\n\n')
    X_dev, X_test, y_dev, y_test = train_test_split(
        df_dev.drop(columns=["trg_flag"]), df_dev["trg_flag"]
        , test_size=config["TESTSIZE"]
        , random_state=config["RS"]
    )
    X_oot, y_oot = df_oot.drop(columns=["trg_flag"]), df_oot["trg_flag"]
    pools = [(X_dev, y_dev), (X_test, y_test), (X_oot, y_oot)]
    labels = ["DEV", "TEST", "OOT"]
    if verbose: print(f"""\nData are preprocessed and splitted into {'/'.join(labels)}:\n""")
    for i in range(len(pools)):
        print(f"{labels[i]}\tSHAPE: ", pools[i][0].shape)
    print("\n\n")
    return pools, cat_columns_idxs


def getGini(config, model, pools, features=None):
    """
    Расчёт коэффициента Джини предсказаний модели
    """
    from sklearn.metrics import roc_auc_score
    result = []
    for X, y_true in pools:
        y_pred = model.predict_proba(X[features] if features else X.drop(columns=config["IDCOLUMNS"].split()))[:, 1]
        result.append(round((2 * roc_auc_score(y_true, y_pred) - 1) * 100, 2))
    return result


def makePreselection(config, pools, catIndexesLong, mode="TEST"):
    """
    Грубый предварительный отбор признаков с помощью CatBoost
    """
    import pandas, os
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    verbose = config["STAGE"]["PRESEL"]["VERBOSE"] == "TRUE"
    preselImportancePath = os.path.join(getPath(config, to="test" if mode == "TEST" else "reports"),
                                        "preselImportance.csv")
    preselReportPath = os.path.join(getPath(config, to="test" if mode == "TEST" else "reports"), "preselReport.csv")
    ##################
    preselReportPath_ext = os.path.join(getPath(config, to="test" if mode == "TEST" else "reports"),
                                        "preselReport_ext.csv")
    ##################
    preselSQLPath = os.path.join(getPath(config, to="test" if mode == "TEST" else "reports"), 'topFeaturesSQL.txt')
    num_top_features = config["STAGE"]["PRESEL"]["NFEATS"]
    X_dev, y_dev = pools.pop(0)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_dev, y_dev, test_size=config["STAGE"]["PRESEL"]["VALID"], random_state=config["RS"]
    )
    trainPool = X_train, y_train
    validPool = X_valid, y_valid
    estimator = CatBoostClassifier(
        iterations=config["STAGE"]["PRESEL"]["NITER"]
        , random_state=config["RS"]
        , use_best_model=True
        , verbose=verbose
        , eval_metric='AUC'
        , thread_count=config["STAGE"]["PRESEL"]["NJOBS"]
    )
    fitParams = {
        "use_best_model": True
        , "early_stopping_rounds": config["STAGE"]["PRESEL"]["STOP"]
        , "verbose": 10 if verbose else 0
    }

    if not os.path.exists(preselImportancePath):
        preselReport = pandas.DataFrame(index=["train", "valid"])
        ##################
        preselReport_ext = pandas.DataFrame(index=["train", "valid", "test", "oot"])
        ##################
        # обучаем бустинг на полном наборе признаков
        estimator.fit(
            X_train.drop(columns=config["IDCOLUMNS"].split()), y_train
            , cat_features=[idx - 2 for idx in catIndexesLong]
            , eval_set=(X_valid.drop(columns=config["IDCOLUMNS"].split()), y_valid)
            , **fitParams
        )
        preselReport["LONG"] = getGini(config, estimator, [trainPool, validPool])
        ##################
        preselReport_ext["LONG"] = getGini(config, estimator, [trainPool, validPool] + pools[-2:])
        ##################
        # оцениваем важность признаков
        imp_df = pandas.DataFrame(index=range(len(X_train.drop(columns=config["IDCOLUMNS"].split()).columns)))
        imp_df['feature'] = X_train.drop(columns=config["IDCOLUMNS"].split()).columns
        imp_df['importance'] = estimator.feature_importances_
        imp_df.sort_values('importance', ascending=False) \
            .reset_index(drop=True) \
            .to_csv(preselImportancePath)
    # отбираем топ признаков
    imp_df = pandas.read_csv(preselImportancePath)
    imp_features = imp_df.head(num_top_features)['feature'].values
    imp_features = [f for f in imp_features if not f.endswith('_dt') and not f.startswith('lifestyle')] \
                   + [f for f in imp_features if f.endswith('_dt')] \
                   + [f for f in imp_features if f.startswith('lifestyle')]
    with open(preselSQLPath, 'w') as f:
        for feat in imp_features:
            f.write(', b.' + feat + '\n')
    # обучаем бустинг на отобранных признаках
    # выделим индексы категориальных признаков
    catIndexes = getCatIndexes(config, imp_features)
    estimator.fit(
        X_train[imp_features], y_train
        , cat_features=catIndexes
        , eval_set=(X_valid[imp_features], y_valid)
        , **fitParams
    )
    if os.path.exists(preselReportPath):
        recentReport = pandas.read_csv(preselReportPath)
        preselReport = pandas.DataFrame(index=["train", "valid"])
        preselReport["LONG"] = recentReport["LONG"].values
    preselReport[f"TOP {num_top_features}"] = getGini(config, estimator, [trainPool, validPool], imp_features)
    preselReport.to_csv(preselReportPath, index_label="POOL")
    ##################
    if os.path.exists(preselReportPath_ext):
        recentReport_ext = pandas.read_csv(preselReportPath_ext)
        preselReport_ext = pandas.DataFrame(index=["train", "valid", "test", "oot"])
        preselReport_ext["LONG"] = recentReport_ext["LONG"].values
    preselReport_ext[f"TOP {num_top_features}"] = getGini(config, estimator, [trainPool, validPool] + pools[-2:],
                                                          imp_features)
    preselReport_ext.to_csv(preselReportPath_ext, index_label="POOL")
    ##################


def makeCrossvalidation(
        config, X, y, n_splits, val_size, model, model_type, cat_features, fit_params,
        sampler, confbands, n_bs, random_state, verb, sampler_type="other", round_to=2
):
    import pandas as pd
    import numpy as np
    from numba import njit
    from tqdm import tqdm
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    report = {
        'scores': None
        , 'gini_test_agg': None
        , 'gini_train': []
        , 'median_gini_train': None
        , 'gini_test': []
        , 'median_gini_test': None
    }
    assert model_type in ['catboost', 'xgboost', 'lightgbm', 'other']
    assert sampler_type == 'smote' or sampler_type == 'other'
    dummy = False
    if model_type not in ['catboost', 'lightgbm'] or sampler_type == 'smote': dummy = True
    def make_dummies(X, cat_idx):
        return pd.concat([X.drop(columns=X.columns[cat_idx]), pd.get_dummies(X[X.columns[cat_idx]])], axis=1)
    if cat_features:
        if dummy:
            X = make_dummies(X, cat_features)
    for train_val_idx, test_idx in tqdm(skf.split(X, y)) if verb else skf.split(X, y):
        x_train_val = X.iloc[train_val_idx, :]
        y_train_val = y.iloc[train_val_idx]
        x_test = X.iloc[test_idx, :]
        y_test = y.iloc[test_idx]
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        cols = sorted(set(x_train.columns) & set(x_val.columns) & set(x_test.columns))
        cat_features = getCatIndexes(config, cols)
        x_train = x_train[cols]
        x_val = x_val[cols]
        x_test = x_test[cols]
        if sampler_type == 'smote' or model_type == 'other':
            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            imputer = imputer.fit(x_train)
            x_train = pd.DataFrame(data=imputer.transform(x_train), index=x_train.index, columns=x_train.columns)
            x_val = pd.DataFrame(data=imputer.transform(x_val), index=x_val.index, columns=x_val.columns)
            x_test = pd.DataFrame(data=imputer.transform(x_test), index=x_test.index, columns=x_test.columns)
        if sampler:
            x_train, y_train = sampler.fit_sample(x_train, y_train)
        if model_type == 'xgboost':
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], **fit_params)
        elif model_type == 'catboost':
            model.fit(x_train, y_train, cat_features=cat_features, eval_set=(x_val, y_val), **fit_params)
        elif model_type == 'lightgbm':
            def lgb_transformer(x_train, x_list, cat_features):
                for col in x_train.columns[cat_features]:
                    #                         values = np.unique(x_train[col].values)
                    values = x_train[col].value_counts().index.values
                    map_dict = {values[i]: i + 1 for i in range(len(values))}
                    map_dict['nan'] = -1
                    x_train[col] = x_train[col].map(map_dict)
                    for x_val in x_list:
                        x_val[col] = x_val[col].map(map_dict)
                return x_train, x_list[0], x_list[1]

            x_train, x_val, x_test = lgb_transformer(x_train, [x_val, x_test], cat_features)
            model.fit(x_train, y_train, categorical_feature=cat_features, eval_set=(x_val, y_val), **fit_params)
        else:
            model.fit(x_train, y_train, **fit_params)
        y_test_p = model.predict_proba(x_test)[:, 1]
        y_train_p = model.predict_proba(x_train)[:, 1]
        report['scores'] = pd.concat([pd.Series(data=y_test_p, index=y_test.index), report['scores']])
        report['gini_train'].append(100 * (roc_auc_score(y_train, y_train_p) * 2 - 1))
        report['gini_test'].append(100 * (roc_auc_score(y_test, y_test_p) * 2 - 1))
    report['median_gini_train'] = np.median(report['gini_train'])
    report['median_gini_test'] = np.median(report['gini_test'])
    y_true = y.sort_index()
    y_pred = report['scores'].sort_index()
    gini_agg = 100 * (2 * roc_auc_score(y_true, y_pred) - 1)
    report['gini_test_agg'] = gini_agg
    if confbands:
        @njit
        def fast_auc(y_true, y_prob):
            y_true = np.asarray(y_true)
            y_true = y_true[np.argsort(y_prob)]
            nfalse = 0
            auc = 0
            n = len(y_true)
            for i in range(n):
                y_i = y_true[i]
                nfalse += (1 - y_i)
                auc += y_i * nfalse
            auc /= (nfalse * (n - nfalse))
            return auc

        np.random.seed(random_state)
        gini_list = []
        for _ in tqdm(range(n_bs)) if verb else range(n_bs):
            idxs = np.random.randint(0, len(y), len(y))
            gini = (2 * roc_auc_score(y_true.values[idxs], y_pred.values[idxs]) - 1) * 100
            gini_list.append(gini)
        gini_list = sorted(gini_list)
        gini_low = gini_list[int(n_bs * 0.025)]
        gini_upp = gini_list[int(n_bs * 0.975)]

    result = list(map(
        lambda x: round(x, round_to)
        , [
            report['gini_test_agg']
            , gini_low if confbands else np.nan
            , gini_upp if confbands else np.nan
        ]))
    return result, report


def makePipeline(config, pools, catIndexes, mode="TEST"):
    """
    Выбор пайплайна моделирования: семплирование + градиентный бустинг
    """
    import pandas, os
    from collections import namedtuple
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    n_estimators = config["STAGE"]["PIPELINE"]["NITER"]
    early_stop = config["STAGE"]["PIPELINE"]["STOP"]
    n_jobs = config["STAGE"]["PIPELINE"]["NJOBS"]
    verbose = config["STAGE"]["PIPELINE"]["VERBOSE"] == "TRUE"
    confbands = config["STAGE"]["PIPELINE"]["BANDS"] == "TRUE"
    modelsToCheck = [name.strip() for name in config["STAGE"]["PIPELINE"]["MODELS"].split()]
    samplersToCheck = [name.strip() for name in config["STAGE"]["PIPELINE"]["SAMPLERS"].split()]
    fracsToCheck = [int(frac) / 100 for frac in config["STAGE"]["PIPELINE"]["FRACS"].split()]

    cvParams = {
        "config": config,
        "n_splits": config["STAGE"]["PIPELINE"]["NSPLITS"],
        "val_size": config["STAGE"]["PIPELINE"]["VALID"],
        "confbands": confbands,
        "n_bs": config["STAGE"]["PIPELINE"]["NBS"],
        "random_state": config["RS"],
        "verb": verbose
    }

    X_train, y_train = pools[0]
    pipelineData = X_train.copy()
    pipelineData['trg_flag'] = y_train
    pipelineData = pipelineData.sample(frac=config["STAGE"]["PIPELINE"]["SAMPLE"], random_state=config["RS"])
    X, y = pipelineData.drop(columns=config["IDCOLUMNS"].split() + ["trg_flag"]), pipelineData['trg_flag']

    pipelineReport = pandas.DataFrame(index=['GINI', 'GINI_low', 'GINI_upp'])
    pipelineReportPath = os.path.join(getPath(config, to="test" if mode == "TEST" else "reports"), "pipelineReport.csv")

    boosters = []
    Booster = namedtuple('booster', 'prelabel type object fitparams')

    # базовые модели
    # xgboost
    if "XGBOOST" in modelsToCheck:
        xgb = XGBClassifier(
            n_estimators=n_estimators, reg_alpha=1, reg_lambda=100, max_depth=4
            , gamma=10, n_jobs=n_jobs, random_state=config["RS"]
        )
        xgb_fit_params = {
            "eval_metric": "auc",
            "early_stopping_rounds": early_stop,
            "verbose": False
        }
        xgb_booster = Booster('XGB_', 'xgboost', xgb, xgb_fit_params)
        boosters.append(xgb_booster)

    # lightgbm
    if "LIGHTGBM" in modelsToCheck:
        lgb = LGBMClassifier(
            boosting_type='gbdt', objective='binary', learning_rate=0.001, n_estimators=n_estimators
            , reg_alpha=1, reg_lambda=100, max_depth=4, random_state=config["RS"], n_jobs=n_jobs
        )
        lgb_fit_params = {
            "eval_metric": "auc",
            "early_stopping_rounds": early_stop,
            "verbose": False
        }
        lgb_booster = Booster('LGB_', 'lightgbm', lgb, lgb_fit_params)
        boosters.append(lgb_booster)

    # catboost
    if "CATBOOST" in modelsToCheck:
        cb = CatBoostClassifier(
            n_estimators=n_estimators, eval_metric='AUC', random_state=config["RS"]
            , use_best_model=True, thread_count=n_jobs
        )
        cb_fit_params = {
            'use_best_model': True
            , 'verbose': False
            , 'early_stopping_rounds': early_stop
        }
        cb_booster = Booster('CB_', 'catboost', cb, cb_fit_params)
        boosters.append(cb_booster)

    for booster in boosters:
        # применение базовой модели на оригинальном распределении таргета
        label = booster.prelabel + 'ORIG'
        pipelineReport[label] = makeCrossvalidation(
            X=X, y=y, model=booster.object, model_type=booster.type, fit_params=booster.fitparams
            , cat_features=[idx - 2 for idx in catIndexes], sampler=False, **cvParams
        )[0]
        if verbose:
            print(pipelineReport.loc[:, [label]].T)
            print(booster.prelabel + 'ORIG IS READY\n\n')

        # UNDER SAMPLING    # RandomUnderSampler
        if "UNDER" in samplersToCheck:
            for us_frac in fracsToCheck:
                label = booster.prelabel + 'US_' + str(
                    int(us_frac * 100) if us_frac * 100 >= 1 else us_frac * 100) + '%'
                underSampler = RandomUnderSampler(sampling_strategy=us_frac / (1 - us_frac), random_state=config["RS"])
                pipelineReport[label] = makeCrossvalidation(
                    X=X, y=y, model=booster.object, model_type=booster.type, fit_params=booster.fitparams
                    , cat_features=[idx - 2 for idx in catIndexes], sampler=underSampler, **cvParams
                )[0]
                if verbose:
                    print(pipelineReport.loc[:, [label]].T)
                    print('\n')
            if verbose:
                print(booster.prelabel + 'RandomUnderSampler IS READY\n\n')

        # OVER SAMPLING     # RandomOverSampler
        if "OVER" in samplersToCheck:
            for os_frac in fracsToCheck:
                label = booster.prelabel + 'OS_' + str(
                    int(os_frac * 100) if os_frac * 100 >= 1 else os_frac * 100) + '%'
                overSampler = RandomOverSampler(sampling_strategy=os_frac / (1 - os_frac), random_state=config["RS"])
                pipelineReport[label] = makeCrossvalidation(
                    X=X, y=y, model=booster.object, model_type=booster.type, fit_params=booster.fitparams
                    , cat_features=[idx - 2 for idx in catIndexes], sampler=overSampler, **cvParams
                )[0]
                if verbose:
                    print(pipelineReport.loc[:, [label]].T)
            if verbose:
                print(booster.prelabel + 'RandomOverSampler IS READY\n\n')
    if confbands:
        pipelineReport = pipelineReport.T.sort_values('GINI_low', ascending=False)
    else:
        pipelineReport = pipelineReport.T.sort_values('GINI', ascending=False)[['GINI']]
    pipelineReport.to_csv(pipelineReportPath)


def makeDummies(X, catIndexes):
    """
    Dummy-кодирование датасета
    """
    import pandas as pd
    return pd.concat([X.drop(columns=X.columns[catIndexes]), pd.get_dummies(X[X.columns[catIndexes]])],
                     axis=1) if catIndexes else X


def makeFeatures(config, X_list, catIndexes):
    """
    Подготовка признаков для подачи в бустинг
    """
    import numpy
    params = config["STAGE"]["PIPELINE"]["RESULT"]
    if params["MODEL"] == "XGBOOST":
        cols = None
        for i in range(len(X_list)):
            X_list[i] = makeDummies(X_list[i], catIndexes)
            if cols:
                cols &= set(X_list[0].columns)
            else:
                cols = set(X_list[0].columns)
        cols = sorted(cols)
        for i in range(len(X_list)):
            X_list[i] = X_list[i][cols]
    elif params["MODEL"] == "LIGHTGBM":
        X_train = X_list[0]
        for col in X_train.columns[catIndexes]:
            values = numpy.unique(X_train[col].values)
            map_dict = {values[i]: i + 1 for i in range(len(values))}
            map_dict['nan'] = -1
            X_train[col] = X_train[col].map(map_dict)
            for X_val in X_list[1:]:
                X_val[col] = X_val[col].map(map_dict)
    return X_list


def trainPipeline(config, pools, catIndexes, bestFeatures=None, bestParams=None, selection=False):
    """
    Обучение пайплайна моделирования: семплирование + градиентный бустинг
    """
    import pandas
    from sklearn.model_selection import train_test_split
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    params = config["STAGE"]["PIPELINE"]["RESULT"]
    finStop = config["STAGE"]["TUNING"]["FINALSTOP"]
    finValid = config["STAGE"]["TUNING"]["FINALVALID"]
    verbose = False if selection else config["STAGE"]["TUNING"]["VERBOSE"] == "TRUE"
    if len(pools) == 3:
        X_dev, y_dev = pools[0]
        X_test, y_test = pools[1]
        X_oot, y_oot = pools[2]
        X_t, X_valid, y_t, y_valid = train_test_split(
            X_dev, y_dev
            , test_size=finValid if finValid > 0 and bestFeatures and bestParams else params["VALID"]
            , random_state=config["RS"])
        if params["SAMPLER"] != "FALSE":
            frac = bestParams["frac"] if bestParams else int(params["FRAC"]) / 100
            samplerParams = {
                "sampling_strategy": float(frac / (1 - frac))
                , "random_state": config["RS"]
            }
        if params["SAMPLER"] == "UNDER":
            sampler = RandomUnderSampler(**samplerParams)
        elif params["SAMPLER"] == "OVER":
            sampler = RandomOverSampler(**samplerParams)
        try:
            X_train, y_train = sampler.fit_sample(X_t, y_t)
            if verbose: print("\nTrain dataset has been resampled\n\n")
        except NameError:
            X_train, y_train = X_t, y_t
    else:
        X_train, y_train = pools[0]
        X_valid, y_valid = pools[1]
        X_test, y_test = pools[2]
        X_oot, y_oot = pools[3]

    X_train, X_valid, X_test, X_oot = makeFeatures(config, [X_train, X_valid, X_test, X_oot], catIndexes)
    if verbose:
        print("\nDatasets has been transformed for model training\n\n")

    if params["MODEL"] == "CATBOOST":
        if bestParams:
            model = CatBoostClassifier(
                iterations=params["NITER"], eval_metric='AUC', thread_count=params["NJOBS"], random_state=config["RS"]
                , **{x: y for (x, y) in bestParams.items() if x not in ['frac', 'fspace_idx']}
            )
        else:
            model = CatBoostClassifier(
                iterations=params["NITER"], eval_metric='AUC', thread_count=params["NJOBS"], random_state=config["RS"]
            )
        fitParams = {
            "use_best_model": True
            , "early_stopping_rounds": finStop if finStop > 0 and bestFeatures and bestParams else params["STOP"]
            , "verbose": 10 if verbose else 0
        }
        model.fit(
            X_train[bestFeatures] if bestFeatures else X_train.drop(columns=config["IDCOLUMNS"].split()), y_train
            , eval_set=(
            X_valid[bestFeatures] if bestFeatures else X_valid.drop(columns=config["IDCOLUMNS"].split()), y_valid)
            , cat_features=[idx - 2 for idx in catIndexes], **fitParams
        )
        if verbose: print("\nModel is trained\n\n")
    elif params["MODEL"] == "XGBOOST":
        if bestParams:
            model = XGBClassifier(
                n_estimators=params["NITER"], n_jobs=params["NJOBS"], random_state=config["RS"]
                , max_depth=int(bestParams['max_depth'])
                , **{x: y for (x, y) in bestParams.items() if x not in ['max_depth', 'frac', 'fspace_idx']}
            )
        else:
            model = XGBClassifier(
                n_estimators=params["NITER"], n_jobs=params["NJOBS"], random_state=config["RS"]
                , reg_alpha=1, reg_lambda=100, max_depth=4, gamma=10
            )
        fitParams = {
            "eval_metric": "auc"
            , "early_stopping_rounds": finStop if finStop > 0 and bestFeatures and bestParams else params["STOP"]
            , "verbose": 10 if verbose else 0
        }
        model.fit(
            X_train[bestFeatures] if bestFeatures else X_train.drop(columns=config["IDCOLUMNS"].split()), y_train
            , eval_set=[(X_valid[bestFeatures] if bestFeatures else X_valid.drop(columns=config["IDCOLUMNS"].split()),
                         y_valid)], **fitParams
        )
        if verbose: print("\nModel is trained\n\n")
    elif params["MODEL"] == "LIGHTGBM":
        if bestParams:
            model = LGBMClassifier(
                boosting_type='gbdt', objective='binary', n_estimators=params["NITER"], n_jobs=params["NJOBS"]
                , **{x:y for (x, y) in bestParams.items() if x not in ['frac', 'num_leaves', 'max_depth', 'fspace_idx']}
                , num_leaves=int(bestParams["num_leaves"]), max_depth=int(bestParams["max_depth"])
                , random_state=config["RS"]
            )
        else:
            model = LGBMClassifier(
                boosting_type='gbdt', objective='binary', n_estimators=params["NITER"], n_jobs=params["NJOBS"]
                , learning_rate=0.001, reg_alpha=1, reg_lambda=100, max_depth=4, random_state=config["RS"]
            )
        fitParams = {
            "eval_metric": "auc"
            , "early_stopping_rounds": finStop if finStop > 0 and bestFeatures and bestParams else params["STOP"]
            , "verbose": 10 if verbose else 0
        }
        model.fit(
            X_train[bestFeatures] if bestFeatures else X_train.drop(columns=config["IDCOLUMNS"].split()), y_train
            , eval_set=[
                (X_valid[bestFeatures] if bestFeatures else X_valid.drop(columns=config["IDCOLUMNS"].split()), y_valid)]
            , categorical_feature=[idx - 2 for idx in catIndexes], **fitParams
        )
        if verbose: print("\nModel is trained\n\n")
    if bestFeatures:
        X_train = pandas.concat([X_train[config["IDCOLUMNS"].split()], X_train[bestFeatures]], axis=1)
        X_valid = pandas.concat([X_valid[config["IDCOLUMNS"].split()], X_valid[bestFeatures]], axis=1)
        X_test = pandas.concat([X_test[config["IDCOLUMNS"].split()], X_test[bestFeatures]], axis=1)
        X_oot = pandas.concat([X_oot[config["IDCOLUMNS"].split()], X_oot[bestFeatures]], axis=1)
    resPools = [(X_train, y_train), (X_valid, y_valid), (X_test, y_test), (X_oot, y_oot)]
    labels = ["TRAIN", "VALID", "TEST", "OOT"]
    if verbose: print(f"""\nData are preprocessed and splitted into {'/'.join(labels)}:\n""")
    if verbose:
        for i in range(len(resPools)):
            print(f"{labels[i]}\tSHAPE: ", resPools[i][0].shape)
        print("\n\n")
    return model, fitParams, resPools


def getModelPath(config, mode="TEST"):
    """
    Получение пути к директории с новой версией модели
    """
    import os
    sinkPath = getPath(config, to="test" if mode == "TEST" else "results")
    try:
        modelNumbers = sorted([int(name.split("_")[1]) for name in os.listdir(sinkPath) if "model_" in name])
        lastNumber = modelNumbers[-1]
    except IndexError:
        lastNumber = 0
    actualNumber = lastNumber + 1
    return os.path.join(sinkPath, f"model_{actualNumber}")


def getSelectonPath(config, mode="TEST"):
    """
    Получение пути к директории с новой версией модели
    """
    import os
    sinkPath = getPath(config, to="test" if mode == "TEST" else "selections")
    try:
        selectionNumbers = sorted([int(name.split("_")[1]) for name in os.listdir(sinkPath) if "selection_" in name])
        lastNumber = selectionNumbers[-1]
    except IndexError:
        lastNumber = 0
    actualNumber = lastNumber + 1
    return os.path.join(sinkPath, f"selection_{actualNumber}")


def makeUpliftTest(config, model, fitParams, pools, bestFeatures, modelPath, final=False):
    """
    Проведение Gini Uplift теста
    """
    import numpy, os, json
    from copy import deepcopy
    from collections import deque, namedtuple
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score
    fitparams = deepcopy(fitParams)
    del fitparams["verbose"]
    verbose = config["STAGE"]["TUNING"]["VERBOSE"] == "TRUE"
    upliftStack = deque(bestFeatures)
    pools = deque(pools)
    resultFeatures = []
    resultSteps = []
    Step = namedtuple('Step', 'feature trainScore valScore testScore ootScore')
    testFlg = False
    ootFlg = False
    X_train, y_train = pools.popleft()
    X_valid, y_valid = pools.popleft()
    if pools:
        X_test, y_test = pools.popleft()
        testFlg = True
    if pools:
        X_oot, y_oot = pools.popleft()
        ootFlg = True
    if verbose: pbar = tqdm(total=len(bestFeatures))
    while upliftStack:
        featureSet = resultFeatures.copy()
        feature = upliftStack.popleft()
        featureSet.append(feature)
        modelType = config["STAGE"]["PIPELINE"]["RESULT"]["MODEL"]
        if modelType == "XGBOOST":
            model.fit(
                X_train[featureSet], y_train, eval_set=[(X_valid[featureSet], y_valid)], verbose=False, **fitparams
            )
        if modelType == "CATBOOST":
            model.fit(
                X_train[featureSet], y_train, eval_set=[(X_valid[featureSet], y_valid)], verbose=False, **fitparams,
                cat_features=getCatIndexes(config, featureSet)
            )
        if modelType == "LIGHTGBM":
            model.fit(
                X_train[featureSet], y_train, eval_set=[(X_valid[featureSet], y_valid)], verbose=False, **fitparams,
                categorical_feature=getCatIndexes(config, featureSet)
            )
        stepTrain = (2 * roc_auc_score(y_train, model.predict_proba(X_train[featureSet])[:, 1]) - 1) * 100
        stepValid = (2 * roc_auc_score(y_valid, model.predict_proba(X_valid[featureSet])[:, 1]) - 1) * 100
        if testFlg: stepTest = (2 * roc_auc_score(y_test, model.predict_proba(X_test[featureSet])[:, 1]) - 1) * 100
        if ootFlg: stepOot = (2 * roc_auc_score(y_oot, model.predict_proba(X_oot[featureSet])[:, 1]) - 1) * 100
        resultFeatures.append(feature)
        resultSteps.append(
            Step(feature, stepTrain, stepValid, stepTest if testFlg else numpy.nan, stepOot if ootFlg else numpy.nan)
        )
        if verbose: pbar.update(1)
    if verbose: pbar.close()

    def checkCrieria(resultSteps, sample, final):
        assert sample in "train valid test oot"
        matcher = {"train": 1, "valid": 2, "test": 3, "oot": 4}
        gini_max = resultSteps[-1][matcher[sample]]
        yellow = 0.99 * gini_max
        red = 1.05 * gini_max
        nfeats = len(resultSteps)
        red_cnt = 0
        yel_cnt = 0
        subsets = []
        for i in range(nfeats):
            score = resultSteps[i][matcher[sample]]
            if score >= red:
                subsets.append("\t".join([str(i + 1), str(round(score/gini_max, 4)), "FAILED"]))
                red_cnt += 1
            elif score >= yellow and i + 1 <= int(0.6 * nfeats):
                subsets.append("\t".join([str(i + 1), str(round(score/gini_max, 4)), "YELLOW"]))
                yel_cnt += 1
        # результат
        resultPath = os.path.join(modelPath,
                                  f"""GiniUpliftResult_{sample.upper()}{"_FINAL" if final else ""}.txt""")
        with open(resultPath, "w") as f:
            if red_cnt == 0 and yel_cnt == 0:
                f.write("GREEN")
            else:
                f.write("FAILED\n" if red_cnt > 0 else "YELLOW\n")
                f.write("\n".join(subsets))
        if verbose:
            print(f"""\nGini Uplift Test {"final " if final else ""}result for {sample.upper()}{" (FINAL)" if final else ""}:\n""")
            os.system(f"cat {resultPath}")
            print("\n\n")
    checkCrieria(resultSteps, "train", final)
    if ootFlg:
        checkCrieria(resultSteps, "valid", final)
        checkCrieria(resultSteps, "oot", final)
    else:
        checkCrieria(resultSteps, "valid", final)
    # финальный uplift на всех выборках
    if final:
        checkCrieria(resultSteps, "test", final)
    # графики
    plt.figure(figsize=(9, 7))
    plt.plot(range(1, len(resultFeatures) + 1), [step.trainScore for step in resultSteps])
    plt.plot(range(1, len(resultFeatures) + 1), [step.valScore for step in resultSteps])
    plt.plot(range(1, len(resultFeatures) + 1), [step.testScore for step in resultSteps])
    plt.plot(range(1, len(resultFeatures) + 1), [step.ootScore for step in resultSteps])
    plt.legend(['TRAIN', 'VALID', 'TEST', 'OOT'])
    plt.ylabel('GINI')
    plt.xlabel('Features')
    plt.xticks(range(1, len(resultFeatures) + 1))
    plt.scatter(range(1, len(resultFeatures) + 1), [step.trainScore for step in resultSteps])
    plt.scatter(range(1, len(resultFeatures) + 1), [step.valScore for step in resultSteps])
    plt.scatter(range(1, len(resultFeatures) + 1), [step.testScore for step in resultSteps])
    plt.scatter(range(1, len(resultFeatures) + 1), [step.ootScore for step in resultSteps])
    plt.grid()
    picPath = os.path.join(
        modelPath
        , f"""GiniUpliftPic_{"OOT" if ootFlg else "VALID"}{"_FINAL" if final else ""}.png"""
    )
    plt.savefig(picPath)
    # пошаговая характеристика
    resultSteps = {x + 1: y for x, y in enumerate(resultSteps)}
    stepsPath = os.path.join(
        modelPath
        , f"""GiniUpliftSteps_{"OOT" if ootFlg else "VALID"}{"_FINAL" if final else ""}.json"""
    )
    with open(stepsPath, "w") as f:
        json.dump(resultSteps, f)


def getKFoldScore(params, config=None, X=None, y=None, catIndexes=None):
    """
    Расчёт оптимизируемой функции: средний Джини на кросс-валидации + штраф за переобучение
    """
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    settings = config["STAGE"]["PIPELINE"]["RESULT"]
    iterations = settings["NITER"]
    early_stop = settings["STOP"]
    n_jobs = settings["NJOBS"]
    valSize = settings["VALID"]
    n_splits = config["STAGE"]["TUNING"]["HYPEROPT"]["NSPLITS"]
    random_state = config["RS"]

    if settings["MODEL"] == "CATBOOST":
        model = CatBoostClassifier(
            iterations=iterations
            , depth=params['depth']
            , l2_leaf_reg=params['l2_leaf_reg']
            , rsm=params['rsm']
            , thread_count=n_jobs
            , eval_metric='AUC'
            , use_best_model=True
            , random_state=random_state
        )
        fitParams = {
            'use_best_model': True
            , 'verbose': False
            , 'early_stopping_rounds': early_stop
        }
        modelType = "catboost"
    elif settings["MODEL"] == "XGBOOST":
        model = XGBClassifier(
            n_estimators=iterations
            , learning_rate=params['learning_rate']
            , reg_alpha=params['reg_alpha']
            , reg_lambda=params['reg_lambda']
            , max_depth=int(params['max_depth'])
            , gamma=params['gamma']
            , subsample=params['subsample']
            , n_jobs=n_jobs
            , random_state=random_state
        )
        fitParams = {
            "eval_metric": "auc",
            "early_stopping_rounds": early_stop,
            "verbose": False,
        }
        modelType = "xgboost"
    elif settings["MODEL"] == "LIGHTGBM":
        model = LGBMClassifier(
            boosting_type='gbdt'
            , num_leaves=int(params['num_leaves'])
            , max_depth=int(params['max_depth'])
            , learning_rate=params['learning_rate']
            , n_estimators=iterations
            , objective='binary'
            , subsample=params['subsample']
            , reg_alpha=params['reg_alpha']
            , reg_lambda=params['reg_lambda']
            , random_state=random_state
            , n_jobs=n_jobs
        )
        fitParams = {
            'eval_metric': 'auc'
            , 'verbose': False
            , 'early_stopping_rounds': early_stop
        }
        modelType = "lightgbm"
    sampler = False
    if settings["SAMPLER"] == "UNDER":
        sampler = RandomUnderSampler(sampling_strategy=params['frac'] / (1 - params['frac']), random_state=random_state)
    elif settings["SAMPLER"] == "OVER":
        sampler = RandomOverSampler(sampling_strategy=params['frac'] / (1 - params['frac']), random_state=random_state)

    def combine(cv_function):
        def wrapper(*args, **kwargs):
            report = cv_function(*args, **kwargs)[1]
            giniTrain = report['median_gini_train']
            giniTest = report['median_gini_test']
            return giniTrain, giniTest

        return wrapper

    @combine
    def makeCrossvalidation_upd(*args, **kwargs):
        return makeCrossvalidation(*args, **kwargs)

    giniTrain, giniTest = makeCrossvalidation_upd(
        config, X, y, n_splits, valSize, model, modelType, catIndexes, fitParams, sampler
        , confbands=False, n_bs=2000, random_state=random_state, verb=False, sampler_type="other", round_to=8
    )
    return -giniTest + config["STAGE"]["TUNING"]["HYPEROPT"]["KFOLD_FINE"] * abs(giniTrain - giniTest)


def getStableScore(params, config=None, X=None, y=None, fSpace=None):
    """
    Расчёт оптимизируемой функции: ориентация на стабильность прогнозов
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    date_id = config["IDCOLUMNS"].split()[1]
    settings = config["STAGE"]["PIPELINE"]["RESULT"]
    iterations = settings["NITER"]
    early_stop = settings["STOP"]
    n_jobs = settings["NJOBS"]
    valSize = settings["VALID"]
    random_state = config["RS"]
    stableParams = config["STAGE"]["TUNING"]["HYPEROPT"]["STABLETUNING"]
    verbose = config["STAGE"]["TUNING"]["VERBOSE"] == "TRUE"

    if settings["MODEL"] == "CATBOOST":
        model = CatBoostClassifier(
            iterations=iterations
            , depth=params['depth']
            , l2_leaf_reg=params['l2_leaf_reg']
            , rsm=params['rsm']
            , thread_count=n_jobs
            , eval_metric='AUC'
            , use_best_model=True
            , random_state=random_state
        )
        fitParams = {
            'use_best_model': True
            , 'verbose': False
            , 'early_stopping_rounds': early_stop
        }
    elif settings["MODEL"] == "XGBOOST":
        model = XGBClassifier(
            n_estimators=iterations
            , learning_rate=params['learning_rate']
            , reg_alpha=params['reg_alpha']
            , reg_lambda=params['reg_lambda']
            , max_depth=int(params['max_depth'])
            , gamma=params['gamma']
            , subsample=params['subsample']
            , n_jobs=n_jobs
            , random_state=random_state
        )
        fitParams = {
            "eval_metric": "auc",
            "early_stopping_rounds": early_stop,
            "verbose": False,
        }
    elif settings["MODEL"] == "LIGHTGBM":
        model = LGBMClassifier(
            boosting_type='gbdt'
            , num_leaves=int(params['num_leaves'])
            , max_depth=int(params['max_depth'])
            , learning_rate=params['learning_rate']
            , n_estimators=iterations
            , objective='binary'
            , subsample=params['subsample']
            , reg_alpha=params['reg_alpha']
            , reg_lambda=params['reg_lambda']
            , random_state=random_state
            , n_jobs=n_jobs
        )
        fitParams = {
            'eval_metric': 'auc'
            , 'verbose': False
            , 'early_stopping_rounds': early_stop
        }
    dev_dates = np.unique(X[date_id].values)
    dev_scores = []
    oot_scores = []
    features = list(fSpace[params["fspace_idx"]])
    catIndexes = getCatIndexes(config, features)

    for i in range(1, len(dev_dates)):
        oot_dts = [dev_dates[i]]
        dev_dts = dev_dates[:i]
        x_oot = X[X[date_id].apply(lambda x: x in oot_dts)][features]
        y_oot = y[x_oot.index]
        x_dev = X[X[date_id].apply(lambda x: x in dev_dts)][features]
        y_dev = y[x_dev.index]
        x_t, x_val, y_t, y_val = train_test_split(x_dev, y_dev, test_size=valSize, random_state=random_state)

        if settings["SAMPLER"] != "FALSE":
            frac = params["frac"]
            samplerParams = {
                "sampling_strategy": float(frac / (1 - frac))
                , "random_state":random_state
            }
        if settings["SAMPLER"] == "UNDER":
            sampler = RandomUnderSampler(**samplerParams)
        elif settings["SAMPLER"] == "OVER":
            sampler = RandomOverSampler(**samplerParams)
        try:
            x_train, y_train = sampler.fit_sample(x_t, y_t)
        except NameError:
            x_train, y_train = x_t, y_t
        x_train, x_val = makeFeatures(config, [x_train, x_val], catIndexes)

        if settings["MODEL"] == "CATBOOST":
            model.fit(x_train, y_train, eval_set=(x_val, y_val), cat_features=catIndexes, **fitParams)
        elif settings["MODEL"] == "XGBOOST":
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], **fitParams)
        elif settings["MODEL"] == "LIGHTGBM":
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=catIndexes, **fitParams)

        y_dev_p = model.predict_proba(x_dev)[:, 1]
        dev_score = 200 * roc_auc_score(y_dev, y_dev_p) - 100
        y_oot_p = model.predict_proba(x_oot)[:, 1]
        oot_score = 200 * roc_auc_score(y_oot, y_oot_p) - 100
        dev_scores.append(dev_score)
        oot_scores.append(oot_score)
        del oot_dts, dev_dts, x_oot, y_oot, x_dev, y_dev, x_train, x_val, y_train, y_val

    stepReport = pd.DataFrame(index=[str(dt) for dt in dev_dates[1:]], columns=["DEV", "OOT"])
    stepReport["DEV"] = dev_scores
    stepReport["OOT"] = oot_scores

    def stabilityScore(devScores, ootScores):
        if np.mean(oot_scores) < stableParams["ACCEPT_LVL"]:
            return np.inf
        weights = np.linspace(1, stableParams["ACTUALITY"], len(dev_dates)-1)
        weights /= sum(weights)
        result = -sum(ootScores * weights)
        result += stableParams["STABLE_FINE"] * abs(np.mean(devScores) - np.mean(ootScores))
        return result

    if verbose:
        print(stepReport.T)
        print(f"""Stabilty Score: {stabilityScore(dev_scores, oot_scores)}""")

    return stabilityScore(dev_scores, oot_scores)


def getBestParams(config, X, y, catIndexes):
    """
    Подбор гиперпараметров пайплайна
    """
    import numpy
    from hyperopt import fmin, hp, tpe, atpe, anneal, Trials
    from itertools import combinations
    modelType = config["STAGE"]["PIPELINE"]["RESULT"]["MODEL"]
    verbose = config["STAGE"]["TUNING"]["VERBOSE"] == "TRUE"
    settings = config["STAGE"]["TUNING"]["HYPEROPT"]
    stableParams = settings["STABLETUNING"]
    frac = config["STAGE"]["PIPELINE"]["RESULT"]["FRAC"]
    fracAmp = settings["FRACAMP"]
    fracLower = max(1, frac - fracAmp) / 100
    fracUpper = min(50, frac + fracAmp) / 100
    fSpace = []
    features = list(X.columns)[1:] if config["IDCOLUMNS"].split()[1] in X.columns else list(X.columns)
    fSpaceUpper = min(stableParams["F_UPPER"], len(features))
    fSpaceLower = stableParams["F_LOWER"]
    for i in range(fSpaceLower, fSpaceUpper + 1):
        fSpace += list(combinations(features, i))
    fSpace = numpy.random.choice(fSpace, size=len(fSpace), replace=False)

    if modelType == "CATBOOST":
        space = {
            "fspace_idx": hp.choice("fspace_idx", list(range(len(fSpace)))),
            "frac": hp.uniform("frac", fracLower, fracUpper),
            "depth": hp.quniform("depth", 2, 13, 1),
            "l2_leaf_reg": hp.loguniform("l2_leaf_reg", -5, 2),
            "rsm": hp.uniform("rsm", 0, 1)
        }
    elif modelType == "XGBOOST":
        space = {
            "fspace_idx": hp.choice("fspace_idx", list(range(len(fSpace)))),
            "frac": hp.uniform("frac", fracLower, fracUpper),
            "learning_rate": hp.loguniform("learning_rate", -5, 2),
            "max_depth": hp.quniform("max_depth", 2, 13, 1),
            "reg_alpha": hp.loguniform("reg_alpha", -5, 5),
            "reg_lambda": hp.loguniform("reg_lambda", -5, 5),
            "subsample": hp.uniform("subsample", 0, 1),
            "gamma": hp.uniform("gamma", 0, 15)
        }
    elif modelType == "LIGHTGBM":
        space = {
            "fspace_idx": hp.choice("fspace_idx", list(range(len(fSpace)))),
            "frac": hp.uniform("frac", fracLower, fracUpper),
            "num_leaves": hp.quniform("num_leaves", 10, 100, 1),
            "max_depth": hp.quniform("max_depth", 2, 13, 1),
            "learning_rate": hp.loguniform("learning_rate", -5, 2),
            "subsample": hp.uniform("subsample", 0, 1),
            "reg_alpha": hp.loguniform("reg_alpha", -5, 5),
            "reg_lambda": hp.loguniform("reg_lambda", -5, 5)
        }
    if config["STAGE"]["PIPELINE"]["RESULT"]["SAMPLER"] == "FALSE":
        del space["frac"]
    if settings["DESIGN"] == "KFOLD":
        del space["fspace_idx"]

    algoDict = {"ANNEAL": anneal, "TPE": tpe, "ATPE": atpe}
    trials = Trials()

    def getLoss(params, config=config, X=X, y=y, catIndexes=catIndexes):
        if settings["DESIGN"] == "KFOLD":
            return getKFoldScore(params, config=config, X=X, y=y, catIndexes=catIndexes)
        else:
            return getStableScore(params, config=config, X=X, y=y, fSpace=fSpace)

    bestParams = fmin(
        fn=getLoss
        , space=space, algo=algoDict[settings["ALGO"]].suggest
        , max_evals=settings["NEVALS"], trials=trials, verbose=verbose
        , rstate=numpy.random.RandomState(config["RS"])
    )

    bestFeatures = features if settings["DESIGN"] == "KFOLD" else list(fSpace[bestParams["fspace_idx"]])

    if verbose:
        print(f"""\nBest pipeline hyperparameters are detected:\n""")
        print(bestParams)
        if settings["DESIGN"] == "STABLE":
            print(f"""\nBest feature combination: {" ".join(bestFeatures)}""")
        print("\n\n")

    return bestParams, trials, bestFeatures


def makeStabilityTest(config, giniList, modelPath):
    """
    Проведение тета на стабильность коэффициента Джини на различных выборках
    """
    import os
    verbose = config["STAGE"]["TUNING"]["VERBOSE"] == "TRUE"
    trainScore, validScore, testScore, ootScore = giniList
    # train VS test
    trainTestDelta = trainScore - testScore
    trainTestRatio = trainTestDelta / trainScore
    if trainTestDelta > 15 and trainTestRatio > 0.25:
        trainVStest = "FAILED"
    elif trainTestDelta > 10 and trainTestRatio > 0.15:
        trainVStest = "YELLOW"
    else:
        trainVStest = "GREEN"
    # train VS oot
    trainOotDelta = trainScore - ootScore
    trainOotRatio = trainOotDelta / ootScore
    if trainOotDelta > 15 and trainOotRatio > 0.30:
        trainVSoot = "FAILED"
    elif trainOotDelta > 10 and trainOotRatio > 0.20:
        trainVSoot = "YELLOW"
    else:
        trainVSoot = "GREEN"
    # oot VS test
    ootTestDelta = ootScore - testScore
    ootTestRatio = ootTestDelta / ootScore
    if abs(ootTestDelta) > 10 and abs(ootTestRatio) > 0.20:
        ootVStest = "FAILED"
    elif abs(ootTestDelta) > 5 and abs(ootTestRatio) > 0.15:
        ootVStest = "YELLOW"
    else:
        ootVStest = "GREEN"
    result = [trainVStest, trainVSoot, ootVStest]
    with open(os.path.join(modelPath, "GiniStabilityResult.txt"), "w") as f:
        if "FAILED" in result:
            f.write("FAILED\n")
        elif "YELLOW" in result:
            f.write("YELLOW\n")
        else:
            f.write("GREEN\n")
        f.write(f"""trainVStest: {trainVStest}\n""")
        f.write(f"""trainVSoot: {trainVSoot}\n""")
        f.write(f"""ootVStest: {ootVStest}""")
    if verbose:
        print("\nGini Stability Test result:\n")
        os.system(f"""cat {os.path.join(modelPath, "GiniStabilityResult.txt")}""")
        print("\n\n")


def getSelections(config):
    """
    Формирование сводной таблицы по результатам отбора признаков
    """
    import os, pandas, numpy, json
    test = config["STAGE"]["FINESEL"]["MODE"] == "TEST"
    sinkPath = getPath(config, to="test" if test else "selections")
    selections = [name for name in os.listdir(sinkPath) if "selection_" in name]
    selectionReport = pandas.DataFrame(
        index=["bins", "size", "anneal_iter", "fast", "start_from", "vif_cutoff", "n_splits", "perm_iter"]
    )
    for selection in selections:
        selectionPath = os.path.join(sinkPath, selection)
        cfg = json.load(open(os.path.join(selectionPath, "selectionParams.json"), "r"))
        selectionReport[selection] = [
            cfg["BINNER"]["PARAMS"]["max_bins"], cfg["BINNER"]["PARAMS"]["min_size"],
            cfg["BINNER"]["PARAMS"]["n_iter"], cfg["BINNER"]["FAST"],
            cfg["BINNER"]["PARAMS"]["starts_from"] if cfg["BINNER"]["FAST"] == "TRUE" else "FALSE",
            cfg["VIF_CUTOFF"], cfg["PERMIMP"]["NSPLITS"], cfg["PERMIMP"]["NITER"]
        ]
    return selectionReport.T


def makeFineSelection(config, pools, catIndexes, mode="TEST"):
    """
    Финальная настройка модели: оценка значимости признаков
    """
    import os, json, pickle
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    from time import sleep
    from pybinning import GiniBinner
    from imblearn.under_sampling import RandomUnderSampler
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from eli5.permutation_importance import get_score_importances
    selectionPath = getSelectonPath(config, mode=mode)
    if not os.path.exists(selectionPath): os.mkdir(selectionPath)
    selectionReportPath = os.path.join(
        getPath(config, to="test" if mode == "TEST" else "reports"), "selectionReport.csv"
    )
    vifReportPath = os.path.join(selectionPath, "vifReport.csv")
    permImpReportPath = os.path.join(selectionPath, "permImpReport.csv")
    selectionParams = config["STAGE"]["FINESEL"]
    binningParams = selectionParams["BINNER"]["PARAMS"]
    permImpParams = selectionParams["PERMIMP"]
    verbose = selectionParams["VERBOSE"] == "TRUE"

    X_dev, y_dev = pools[0]
    binner = GiniBinner(
        **binningParams, fast=selectionParams["BINNER"]["FAST"] == "TRUE",
        skiplist=config["IDCOLUMNS"], catlist=list(X_dev.columns[catIndexes]),
        random_state=config["RS"], verbose=selectionParams["VERBOSE"] == "TRUE",
    )
    if selectionParams["SAMPLE"]["UNDER"] == "FALSE":
        X_dev_smp = X_dev.sample(frac=selectionParams["SAMPLE"]["FRAC"], random_state=config["RS"])
        y_dev_smp = y_dev[X_dev_smp.index]
    else:
        frac = selectionParams["SAMPLE"]["FRAC"]
        samplerParams = {
            "sampling_strategy": float(frac / (1 - frac))
            , "random_state": config["RS"]
        }
        sampler = RandomUnderSampler(**samplerParams)
        X_dev_smp, y_dev_smp = sampler.fit_sample(X_dev, y_dev)
    if verbose: print("\nWoE binning is in progress...\n")
    X_dev_bnd = binner.fit_transform(X_dev_smp, y_dev_smp)
    binnerReport = binner.report
    pickle.dump(binnerReport, open(os.path.join(selectionPath, "binnerReport.pkl"), "wb"))
    if verbose: print("\nWoE binning is done!\n")

    def getVIF(X):
        X = X.dropna()
        X = X._get_numeric_data()
        X = X.assign(const=1)
        vif = pd.DataFrame(index=X.columns)
        vif["vif"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif

    correlated = pd.DataFrame(index=["vif"])
    df = X_dev_bnd.drop(columns=config["IDCOLUMNS"].split())
    if verbose: print("\nVIF-based iterative selection is in progress...\n")
    if verbose: pbar = tqdm()
    while True:
        vif = getVIF(df)
        worst_feat = vif.drop(index=["const"]).sort_values(by="vif", ascending=False).index[0]
        worst_vif = vif.loc[worst_feat, "vif"]
        if worst_vif >= selectionParams["VIF_CUTOFF"]:
            correlated[worst_feat[4:]] = [round(worst_vif, 4)]
            df = df.drop(columns=[worst_feat])
            if verbose:
                if verbose: pbar.update(1)
        else:
            if verbose: pbar.close()
            break
    vif.to_csv(vifReportPath, index_label="feature")
    if verbose:
        print(f"\nVIF analysis is done! {len(correlated.T)} highly correlated features are found:\n")
        print(correlated.T)
        print("\n")

    X_dev_res = X_dev_smp.drop(columns=correlated.columns)
    y_dev_res = y_dev_smp
    catIndexes = getCatIndexes(config, list(X_dev_res.columns))
    skf = StratifiedKFold(n_splits=permImpParams["NSPLITS"], shuffle=True, random_state=config["RS"])
    permImpReport = pd.DataFrame()
    permImpReport["feature"] = X_dev_res.drop(columns=config["IDCOLUMNS"].split()).columns
    permImpReport["importance"] = 0
    if verbose: print("\nPermutation importance is being estimating...\n\n")
    for train_idx, val_idx in tqdm(skf.split(X_dev_res, y_dev_res)) if verbose else skf.split(X_dev_res, y_dev_res):
        X_t = X_dev_res.iloc[train_idx, :]
        y_t = y_dev_res.iloc[train_idx]
        X_v = X_dev_res.iloc[val_idx, :]
        y_v = y_dev_res.iloc[val_idx]
        foldPools = [(X_t, y_t), (X_v, y_v), pools[1], pools[2]]
        foldModel, fitParams, foldPools = trainPipeline(config, foldPools, catIndexes, selection=True)
        X_v_modified = X_v.drop(columns=config["IDCOLUMNS"].split())
        def score(X, y):
            roc_auc = roc_auc_score(
                y, foldModel.predict_proba(pd.DataFrame(data=X, columns=X_v_modified.columns))[:, 1]
            )
            return 100 * (roc_auc * 2 - 1)
        base_score, score_decreases = get_score_importances(
            score, X_v_modified.values, y_v, n_iter=permImpParams["NITER"], random_state=config["RS"]
        )
        featureImportances = np.mean(score_decreases, axis=0)
        permImpReport["importance"] += featureImportances
    permImpReport["importance"] /= permImpParams["NSPLITS"]
    permImpReport.sort_values('importance', ascending=False) \
        .reset_index(drop=True) \
        .to_csv(permImpReportPath, index=False)
    json.dump(selectionParams, open(os.path.join(selectionPath, "selectionParams.json"), "w"))
    if verbose: print(f"""\nFeature fineselection is ready\n\n""")
    sleep(3)
    selectionReport = getSelections(config)
    selectionReport.to_csv(selectionReportPath, index_label="selections")
    if verbose:
        print(f"""\nReport updated:\n\n""")
        print(selectionReport)


def makeFineTuning(config, pools, catIndexes, mode="TEST"):
    """
    Финальная настройка модели: отсечение признаков + подбор гиперпараметров + тесты валидации
    """
    import os, numpy, pandas, pickle, json
    from eli5.permutation_importance import get_score_importances
    from sklearn.metrics import roc_auc_score
    verbose = config["STAGE"]["TUNING"]["VERBOSE"] == "TRUE"
    modelPath = getModelPath(config, mode=mode)
    selection = config["STAGE"]["FINESEL"]["RESULT"]
    assert selection > 0
    if not os.path.exists(modelPath): os.mkdir(modelPath)
    topFeatsReportPath = os.path.join(getPath(config, to="test" if mode == "TEST" else "reports"), "topFeatsReport.csv")
    ##################
    topFeatsReportPath_ext = os.path.join(
        getPath(config, to="test" if mode == "TEST" else "reports"), "topFeatsReport_ext.csv"
    )
    ##################
    topFeatsNumber = config["STAGE"]["PRESEL"]["NFEATS"]
    bestFeatsNumber = config["STAGE"]["TUNING"]["NFEATS"]
    permImpReportPath = os.path.join(
        getPath(config, to="test" if mode == "TEST" else "selections"), f"selection_{selection}", "permImpReport.csv"
    )
    GINIReportPath = os.path.join(modelPath, "GINIReport.csv")
    ##################
    GINIReportPath_ext = os.path.join(modelPath, "GINIReport_ext.csv")
    ##################
    tuningResultPath = os.path.join(modelPath, "tuningResult.pkl")
    finetunSQLPath = os.path.join(modelPath, "bestFeaturesSQL.txt")
    ootFeatureImpPath = os.path.join(modelPath, "OOTFeatureImp.csv")

    if config["STAGE"]["TUNING"]["OOTSAMPLE"] < 1:
        X, y = pools[-1]
        ootData = X.copy()
        ootData['trg_flag'] = y
        ootData = ootData.sample(
            frac=config["STAGE"]["TUNING"]["OOTSAMPLE"], random_state=config["RS"] * 2
        )
        X_oot, y_oot = ootData.drop(columns=["trg_flag"]), ootData['trg_flag']
        pools[-1] = X_oot, y_oot

    if verbose:
        print(f"""\n\nTraining modelling pipeline on TOP {topFeatsNumber} features (preselected)...\n""")
    topModel, fitParams, topPools = trainPipeline(config, pools, catIndexes)
    pickle.dump(
        topPools, open(os.path.join(getPath(config, to="test" if mode == "TEST" else "reports"), "pools.pkl"), "wb")
    )
    topGiniList = getGini(config, topModel, topPools)
    topFeatsReport = pandas.DataFrame(index=["train", "valid", "test", "oot"])
    topFeatsReport["GINI"] = topGiniList[:2] + [numpy.nan] * 2
    topFeatsReport.to_csv(topFeatsReportPath, index_label="POOL")
    ##################
    topFeatsReport_ext = pandas.DataFrame(index=["train", "valid", "test", "oot"])
    topFeatsReport_ext["GINI"] = topGiniList
    topFeatsReport_ext.to_csv(topFeatsReportPath_ext, index_label="POOL")
    ##################
    permImpReport = pandas.read_csv(permImpReportPath)
    bestFeatures = list(permImpReport.head(bestFeatsNumber)['feature'].values)
    bestCatIndexes = getCatIndexes(config, config["IDCOLUMNS"].split() + bestFeatures)
    modelType = config["STAGE"]["PIPELINE"]["RESULT"]["MODEL"]
    topPools = pickle.load(
        open(os.path.join(getPath(config, to="test" if mode == "TEST" else "reports"), "pools.pkl"), "rb"))
    devTopPools = topPools[:2]
    if verbose:
        print(f"""\n\nTraining modelling pipeline on TOP {bestFeatsNumber} features (fineselected)...\n""")
    bestModel, fitParams, bestPools = trainPipeline(config, topPools, bestCatIndexes, bestFeatures=bestFeatures)
    bestGiniList = getGini(config, bestModel, bestPools)
    GINIReport = pandas.read_csv(topFeatsReportPath, usecols=["GINI"])
    GINIReport.columns = [f"""TOP {topFeatsNumber}"""]
    GINIReport.index = ["train", "valid", "test", "oot"]
    GINIReport[f"""TOP {bestFeatsNumber}"""] = bestGiniList[:2] + [numpy.nan] * 2
    # if verbose:
    #     print(f"""\n\nGini Uplift test is in progress...\n""")
    # makeUpliftTest(config, bestModel, fitParams, devTopPools, bestFeatures, modelPath)
    ###################
    GINIReport_ext = pandas.read_csv(topFeatsReportPath_ext, usecols=["GINI"])
    GINIReport_ext.columns = [f"""TOP {config["STAGE"]["PRESEL"]["NFEATS"]}"""]
    GINIReport_ext.index = ["train", "valid", "test", "oot"]
    GINIReport_ext[f"""TOP {bestFeatsNumber}"""] = bestGiniList
    if verbose:
        print(f"""\n\nEXTENDED Gini Uplift test is in progress...\n""")
    makeUpliftTest(config, bestModel, fitParams, bestPools, bestFeatures, modelPath)
    ##################
    if config["STAGE"]["TUNING"]["HYPEROPT"]["STATUS"] == "TODO":
        if verbose:
            print(f"""\n\nHyperparameters tuning is in progress...\n""")
        X_dev, y_dev = pools[0]
        dataForTuning = X_dev.copy()
        dataForTuning['trg_flag'] = y_dev
        dataForTuning = dataForTuning.sample(
            frac=config["STAGE"]["TUNING"]["HYPEROPT"]["SAMPLE"], random_state=config["RS"] * 2
        )
        if config["STAGE"]["TUNING"]["HYPEROPT"]["DESIGN"] == "KFOLD":
            X, y = dataForTuning.drop(columns=config["IDCOLUMNS"].split() + ["trg_flag"]), dataForTuning['trg_flag']
            X = makeFeatures(config, [X], [idx - 2 for idx in catIndexes])[0][bestFeatures]
            bestParams, trials, bestFeatures = getBestParams(config, X, y, getCatIndexes(config, bestFeatures))
        else:
            X, y = dataForTuning.drop(columns=config["IDCOLUMNS"].split()[:1] + ["trg_flag"]), dataForTuning['trg_flag']
            X = makeFeatures(config, [X], [idx - 1 for idx in catIndexes])[0]
            X = X[config["IDCOLUMNS"].split()[1:] + bestFeatures]
            bestParams, trials, bestFeatures = getBestParams(
                config, X, y, getCatIndexes(config, config["IDCOLUMNS"].split()[1:] + bestFeatures)
            )
        bestCatIndexes = getCatIndexes(config, config["IDCOLUMNS"].split() + bestFeatures)
        with open(tuningResultPath, "wb") as f:
            pickle.dump((bestParams, trials, bestFeatures), f)
        if verbose:
            print(f"""\n\nTraining final modelling pipeline...\n""")
        finalModel, fitParams, finalPools = trainPipeline(
            config, pools, bestCatIndexes, bestFeatures=bestFeatures, bestParams=bestParams
        )
    elif config["STAGE"]["TUNING"]["HYPEROPT"]["STATUS"] == "DONE":
        if verbose:
            print(f"""\n\nHyperparameters tuning is already done\n""")
        bestParams = config["STAGE"]["TUNING"]["BESTPARAMS"]
        if verbose:
            print(f"""\n\nTraining final modelling pipeline...\n""")
        finalModel, fitParams, finalPools = trainPipeline(
            config, pools, bestCatIndexes, bestFeatures=bestFeatures, bestParams=bestParams
        )
    else:
        if verbose:
            print(f"""\n\nHyperparameters tuning is omitted\n""")
        if verbose:
            print(f"""\n\nTraining final modelling pipeline...\n""")
        finalModel, finalPools = bestModel, bestPools
    if modelType == "XGBOOST":
        bestCatFeatures = list(pools[0][0].columns[catIndexes])
        catValues = None
    elif modelType == "CATBOOST":
        bestCatFeatures = [feature for feature in bestFeatures if
                           bestFeatures.index(feature) in getCatIndexes(config, bestFeatures)]
        catValues = {f: {x: x for x in pools[0][0][f].value_counts().index} for f in bestCatFeatures}
    elif modelType == "LIGHTGBM":
        bestCatFeatures = [feature for feature in bestFeatures if
                           bestFeatures.index(feature) in getCatIndexes(config, bestFeatures)]
        catValues = {}
        X = finalPools[0][0]
        for col in X.columns[bestCatIndexes]:
            values = numpy.unique(X[col].values)
            map_dict = {values[i]: i + 1 for i in range(len(values))}
            map_dict['nan'] = -1
            catValues[col] = map_dict
    else:
        raise Exception
    sqlFeatures = [f for f in bestFeatures if not f.endswith('_dt') and not f.startswith('lifestyle')] \
                  + [f for f in bestFeatures if f.endswith('_dt')] \
                  + [f for f in bestFeatures if f.startswith('lifestyle')]
    with open(finetunSQLPath, 'w') as f:
        for feat in sqlFeatures:
            f.write(', b.' + feat + '\n')
    pickle.dump(finalModel, open(os.path.join(modelPath, "model.pkl"), "wb"))
    pickle.dump(finalPools, open(os.path.join(modelPath, "pools.pkl"), "wb"))
    scoringPackage = bestFeatures, bestCatFeatures, finalModel
    pickle.dump(scoringPackage, open(os.path.join(modelPath, "scoringPackage.pkl"), "wb"))
    json.dump(catValues, open(os.path.join(modelPath, "catValues.json"), "w"))
    finalGiniList = getGini(config, finalModel, finalPools)
    if verbose: print(f"""\nFinal model is trained and saved!\n\n""")
    if verbose: print(f"""\n\nFINAL RESULTS AND VALIDATION TESTS:\n\n""")
    GINIReport[f"""TOP {bestFeatsNumber} FINAL"""] = finalGiniList
    if verbose:
        print(f"""Gini Report for model\n""")
        print(GINIReport.T)
        print("\n\n")
    GINIReport.to_csv(GINIReportPath, index_label="POOL")
    ##################
    GINIReport_ext[f"""TOP {bestFeatsNumber} FINAL"""] = finalGiniList
    if verbose:
        print(f"""Extended Gini Report for model\n""")
        print(GINIReport_ext.T)
        print("\n\n")
    GINIReport_ext.to_csv(GINIReportPath_ext, index_label="POOL")
    ##################
    makeStabilityTest(config, finalGiniList, modelPath)
    X_oot, y_oot = finalPools[-1]
    def score(X, y):
        return 100 * (roc_auc_score(y, finalModel.predict_proba(
            pandas.DataFrame(data=X, columns=X_oot.drop(columns=config["IDCOLUMNS"].split()).columns)
        )[:, 1]) * 2 - 1)
    if verbose: print(f"""\nPermutation importances for final model on OOT is being estimating...\n\n""")
    base, decreases = get_score_importances(
        score, X_oot.drop(columns=config["IDCOLUMNS"].split()).values, y_oot, random_state=config["RS"]
    )
    if verbose: print(f"""\nPermutation importances for final model on OOT are ready\n\n""")
    ootScores = numpy.mean(decreases, axis=0)
    ootFeatureImp = pandas.DataFrame(index=range(1, len(ootScores)+1))
    ootFeatureImp['feature'] = X_oot.drop(columns=config["IDCOLUMNS"].split()).columns
    ootFeatureImp['importance'] = ootScores
    ootFeatureImp.sort_values('importance', ascending=False, inplace=True)
    ootTopFeatures = list(ootFeatureImp['feature'].values)
    print(ootFeatureImp)
    ootFeatureImp.to_csv(ootFeatureImpPath, index_label="initialImp")
    if verbose:
        print(f"""\n\nGini Uplift test is in progress...\n""")
    makeUpliftTest(config, finalModel, fitParams, finalPools, ootTopFeatures, modelPath, final=True)
    json.dump(config, open(os.path.join(modelPath, "cfg.json"), "w"))
    os.system(f"""pip3 freeze > {os.path.join(modelPath, "requirements.txt")}""")


def getResults(config):
    """
    Формирование сводной таблицы по результатам настройки пайплайна
    """
    import os, pandas, numpy, json
    test = config["STAGE"]["DEPLOY"]["MODE"] == "TEST"
    sinkPath = getPath(config, to="test" if test else "results")
    resultReportPath = os.path.join(getPath(config, to="test" if test else "reports"), "resultReport.csv")
    topFeatsReportPath = os.path.join(getPath(config, to="test" if test else "reports"), "topFeatsReport.csv")
    models = [name for name in os.listdir(sinkPath) if "model_" in name]
    baseModelName = f"""TOP {config["STAGE"]["PRESEL"]["NFEATS"]}"""
    resultReport = pandas.DataFrame(
        index=["NFEATS", "TRAIN", "VALID", "TEST", "OOT", "SEL", "STABILITY", "UPLIFT_TRAIN", "UPLIFT_TEST", "UPLIFT_OOT"]
        , columns=[baseModelName] + models
    )
    topFeatsReport = pandas.read_csv(topFeatsReportPath)
    resultReport[baseModelName] = [config["STAGE"]["PRESEL"]["NFEATS"]] + list(topFeatsReport["GINI"]) + [numpy.nan] * 5
    for model in models:
        modelPath = os.path.join(sinkPath, model)
        cfg = json.load(open(os.path.join(modelPath, "cfg.json"), "r"))
        nfeats = cfg["STAGE"]["TUNING"]["NFEATS"]
        selection = cfg["STAGE"]["FINESEL"]["RESULT"]
        giniList = list(pandas.read_csv(os.path.join(modelPath, "GINIReport.csv"))[f"""TOP {nfeats} FINAL"""].values)
        try:
            with open(os.path.join(modelPath, "GiniStabilityResult.txt"), "r") as f:
                stability = f.readline().strip()
        except FileNotFoundError:
            stability = numpy.nan
        try:
            with open(os.path.join(modelPath, "GiniUpliftResult_TRAIN_FINAL.txt"), "r") as f:
                upliftTrain = f.readline().strip()
        except FileNotFoundError:
            upliftTrain = numpy.nan
        try:
            with open(os.path.join(modelPath, "GiniUpliftResult_TEST_FINAL.txt"), "r") as f:
                upliftTest = f.readline().strip()
        except FileNotFoundError:
            upliftTest = numpy.nan
        try:
            with open(os.path.join(modelPath, "GiniUpliftResult_OOT_FINAL.txt"), "r") as f:
                upliftOot = f.readline().strip()
        except FileNotFoundError:
            upliftOot = numpy.nan
        resultReport[model] = [nfeats] + giniList + [selection, stability, upliftTrain, upliftTest, upliftOot]
    resultReport.T["NFEATS"] = resultReport.T["NFEATS"].astype(int)
    for col in ["TRAIN", "VALID", "TEST", "OOT"]:
        resultReport.T[col] = resultReport.T[col].apply(lambda x: round(x, 2))
    resultReport.T.to_csv(resultReportPath, index_label="MODELS")
    return resultReport.T


def makeDeploy(config):
    """
    Загрузка исполняемых файлов в директорию для скоринга
    """
    import os
    test = config["STAGE"]["DEPLOY"]["MODE"] == "TEST"
    modelToDeploy = config["STAGE"]["DEPLOY"]["MODEL"]
    modelPath = os.path.join(getPath(config, to="test" if test else "results"), f"model_{modelToDeploy}")
    if not os.path.exists(modelPath):
        print("This model doesn't exist")
        return
    homePath = os.path.expanduser("~")
    modelID = config["MODEL_ID"]
    scoringPath = os.path.join(getPath(config, to="test"), "projects", modelID) if test \
        else os.path.join(homePath, "projects", modelID)
    if not os.path.exists(scoringPath):
        if test and not os.path.exists(os.path.join(getPath(config, to="test"), "projects")):
            os.mkdir(os.path.join(getPath(config, to="test"), "projects"))
        os.mkdir(scoringPath)
        os.mkdir(os.path.join(scoringPath, "data"))
        os.mkdir(os.path.join(scoringPath, "logs"))
        os.mkdir(os.path.join(scoringPath, "models"))
    deployPath = os.path.join(scoringPath, "models")
    os.system(
        f"""cp {os.path.join(modelPath, "scoringPackage.pkl")} {os.path.join(deployPath, modelID + "_scoring_package.pkl")}""")
    os.system(
        f"""cp {os.path.join(modelPath, "catValues.json")} {os.path.join(deployPath, modelID + "_cat_values.json")}""")
