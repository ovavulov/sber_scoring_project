class GiniBinner(object):

    """
    This class provides optimal binning algorithm that maximizes Gini score for each input variable. Algorithm uses hyperopt library to build the best piecewise linear approximation of ROC for each variable. It utilizes MAE to estimate resulting fitness.
    
    PARAMETERS:
    max_bins: int, default=6
        The highest number of resulting bins for each variable. It mustn't be lower than 2 and upper than the observations number.
    min_size: float, default=0.05
        The smallest possible size for each bin like a fraction of whole observation number. It's set up by default that every bin must contain at least 5% of all observations. It mustn't be lower than 0.01 and upper than 0.5.
    
    method: {'tpe', 'atpe', 'anneal', 'rand'}, default='anneal'
        Optimization method which is used to build the ROC approximation. This parameter relies on methods that are implemented in hyperopt library.
    
    n_iter: int, default=100
        Iterations amount for optimization algorithm. The more this parameter is, the higher Gini score for this variable you can get and the more time you will spend for it. Varying this param make sure you find optimal time-quality trade-off for your own task.  
    
    fast: bool, default=False
        If it's set on True algorithm performance can be accelerated by preliminary equal-size binning execution (with equal amount of observations into each bin). You should tune param 'starts_from' to make this feature suitable for your binning task.
    starts_from: int, default=100
        Ignored when fast=False. Otherwise it defines numer of resulting bins after preliminary equal-size binning. The lower the value is, the faster overall performance is and the less precise inner ROC approximation is. Use it to tune time-quality trade-off for your task.   
    
    random_state: int, default=None
        Random seed for results reproducibility.
    verbose: bool, default=True
        Set on True to track algorithm execution with progress bar.
    
    ATTRIBUTES:
    report: dict
        Report with binning results and useful statistics for each variable:
            - opt_thresholds: list
              List of thresholds separating resulting bins
            - Result bins: list
              List of bins numbers
            - NaN bin: int
              Number of bin with missing values; if it's equal zero missing values are put into separate bin
            - woe: dict
              Dictionary which looks like {bin number: WoE value}
            - target_rate: dict
              Dictionary which looks like {bin number: fraction of positive examples in bin}
            - bin_volume: dict
              Dictionary which looks like {bin number: number of observations in bin}
            - iv: float
              Information value for variable
            - gini: float
              Gini coefficient for variable
            - loss: float
              MAE that was reached after ROC approximation
    na_feat: list
        List of variables with fraction of not missed values less than 'min_size'
    with_error: list
        List of variables which weren't binned cause of some error  
    EXAMPLE:
    from pybinning import GiniBinning
    binner = GiniBinner()
    X_train_binned = binner.fit_transform(X_train, y_train)
    X_val_binned = binner.transform(X_val)
    """
    
    report = {}
    nalist = []
    errorlist = []
    
    def __init__(
      self, max_bins = 6, min_size = 0.05, method = 'anneal', n_iter = 100
      , fast = False, starts_from = 100, random_state = None, skiplist=[], catlist=[], verbose = True):
        self.max_bins = max_bins
        self.min_size = min_size
        self.method = method
        self.n_iter = n_iter
        self.starts_from = starts_from
        self.fast = fast
        self.random_state = random_state
        self.verbose = verbose
        self.report = {}
        self.nalist = []
        self.errorlist = []
        self.skiplist = skiplist
        self.catlist = catlist
        assert self.method in ['tpe', 'atpe', 'anneal', 'rand']
        assert type(self.max_bins) is int and self.max_bins >= 2
        assert type(self.min_size) is float and 0.01 <= self.min_size <= 0.5
        assert type(self.n_iter) is int and self.n_iter >= 1
        assert type(self.fast) is bool
        assert type(self.starts_from) is int and self.starts_from >= 1
        if self.random_state is None:
            from random import randint
            self.random_state = int(randint(1, 1e10))
        assert type(self.random_state) is int
        assert type(self.verbose) is bool
        
        
    def fit_transform(self, X, y):
        
        assert self.max_bins <= len(X)
        
        #requirements
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        from sklearn.metrics import roc_curve, roc_auc_score
        try:
          from hyperopt import fmin, hp, tpe, atpe, rand, anneal, Trials
        except ImportError:
          from hyperopt import fmin, hp, tpe, rand, anneal, Trials

        try:
          import warnings
          warnings.filterwarnings('ignore')
        except ModuleNotFoundError:
          pass

        tdf = pd.DataFrame(index=y.index, columns=['target'])
        tdf['target'] = y.astype(int)
        
        df_good = tdf['target'].value_counts()[0]
        df_bad = tdf['target'].value_counts()[1]
        
        for feature in tqdm(X.columns) if self.verbose else X.columns:

            if feature in self.skiplist:
                tdf[feature] = X[feature]
                continue

            if feature in self.catlist:

                df = pd.DataFrame(index=y.index, columns=[feature, 'target'])
                df[feature] = X[feature].astype(str)
                df['target'] = y.astype(int)
                df[feature] = df[feature].apply(lambda x: "nan" if x is None or str(x) == "nan" else x)
                n_pos = df.groupby(feature)["target"].sum()
                n_total = df.groupby(feature)["target"].count()
                stat_df = pd.DataFrame(index=n_pos.index)
                stat_df["n_pos"] = n_pos.values
                stat_df["n_total"] = n_total.values
                stat_df["frac"] = n_total / len(df)
                stat_df["trg_rate"] = stat_df["n_pos"] / stat_df["n_total"]

                cats = list(stat_df.index)
                matching = {cat: cat for cat in cats}

                while len(stat_df[stat_df["frac"] < self.min_size]) > 0:
                    small_cat = list(stat_df[stat_df["frac"] < self.min_size].sort_values(by="frac").index)[0]
                    big_cats = list(stat_df.index).copy()
                    big_cats.remove(small_cat)
                    dist = stat_df.loc[big_cats, "trg_rate"].apply(
                        lambda x: abs(x - stat_df.loc[small_cat, "trg_rate"])
                    )
                    # print("\n\n")
                    # print(feature)
                    # print(dist)
                    # print(dist.index)
                    # print(np.argmin(dist))
                    # print(type(np.argmin(dist)))
                    # # near_cat = dist.index[np.argmin(dist)]
                    near_cat = np.argmin(dist)
                    new_cat = "_x_".join([near_cat, small_cat])
                    n_pos = stat_df.loc[small_cat, "n_pos"] + stat_df.loc[near_cat, "n_pos"]
                    n_total = stat_df.loc[small_cat, "n_total"] + stat_df.loc[near_cat, "n_total"]
                    stat_df_inv = stat_df.T
                    stat_df_inv[new_cat] = [n_pos, n_total, n_total / len(df), n_pos / n_total]
                    stat_df_inv.drop(columns=[small_cat, near_cat], inplace=True)
                    stat_df = stat_df_inv.T
                    for cat in [small_cat, near_cat]:
                        if cat in matching.keys():
                            matching[cat] = new_cat
                        else:
                            keys = [key[0] for key in matching.items() if key[1] == cat]
                            for key in keys:
                                matching[key] = new_cat


                while stat_df.shape[0] > self.max_bins:

                    cats = list(stat_df.index)
                    substract = pd.DataFrame(index=cats, columns=cats)
                    for i in range(len(cats)):
                        cat_1 = cats[i]
                        for j in range(i, len(cats)):
                            cat_2 = cats[j]
                            delta = abs(stat_df.loc[cat_1, "trg_rate"] - stat_df.loc[cat_2, "trg_rate"])
                            if i == j:
                                delta = np.inf
                            substract.iloc[i, j] = delta
                            substract.iloc[j, i] = delta

                    # print(substract)
                    # print(np.argmin(substract.values))
                    argmin = np.argmin(substract.values)
                    idxs = sorted([argmin // len(cats), argmin % len(cats)])
                    cat_1 = cats[idxs[0]]
                    cat_2 = cats[idxs[1]]
                    new_cat = "_x_".join([cat_1, cat_2])
                    n_pos = stat_df.loc[cat_1, "n_pos"] + stat_df.loc[cat_2, "n_pos"]
                    n_total = stat_df.loc[cat_1, "n_total"] + stat_df.loc[cat_2, "n_total"]
                    stat_df_inv = stat_df.T
                    stat_df_inv[new_cat] = [n_pos, n_total, n_total / len(df), n_pos / n_total]
                    stat_df_inv.drop(columns=[cat_1, cat_2], inplace=True)
                    stat_df = stat_df_inv.T
                    for cat in [cat_1, cat_2]:
                        if cat in matching.keys():
                            matching[cat] = new_cat
                        else:
                            keys = [key[0] for key in matching.items() if key[1] == cat]
                            for key in keys:
                                matching[key] = new_cat

                tdf['woe_%s' % feature] = df[feature].map(matching)
                uniq_vals = np.unique(tdf['woe_%s' % feature])
                woe_iv = {'woe': {}, 'iv': 0}
                bin_volume = tdf.groupby(by='woe_%s' % feature).count().iloc[:, 0]
                target_rate = tdf.groupby(by='woe_%s' % feature)['target'].mean()
                counter = tdf.groupby(by='woe_%s' % feature)['target'].value_counts()

                for i in uniq_vals:
                    try:
                        goods = counter[i][0]
                    except:
                        goods = 0.001
                    try:
                        bads = counter[i][1]
                    except:
                        bads = 0.001
                    woe = np.log((goods / df_good) / (bads / df_bad))
                    woe_iv['woe'][i] = woe

                for j in uniq_vals:
                    try:
                        goods = counter[j][0]
                    except:
                        goods = 0.001
                    try:
                        bads = counter[j][1]
                    except:
                        bads = 0.001
                    woe_iv['iv'] += woe_iv['woe'][j] * ((goods / df_good) - (bads / df_bad))

                def woe_transform(x):
                    for i in uniq_vals:
                        if x == i:
                            return woe_iv['woe'][i]

                tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: woe_transform(x))

                gini = abs(roc_auc_score(tdf['target'], tdf['woe_%s' % feature]) * 2 - 1)

                self.report[feature] = {
                    "type": "CATEGORICAL",
                    'opt_thresholds': matching,
                    'Result bins': list(uniq_vals),
                    'NaN bin': None,
                    'woe': woe_iv['woe'],
                    'target_rate': {i: target_rate[i] for i in uniq_vals},
                    'bin_volume': {i: bin_volume[i] for i in uniq_vals},
                    'iv': woe_iv['iv'],
                    'gini': gini,
                    'loss': None
                }
                continue
            
            df = pd.DataFrame(index=y.index, columns=[feature, 'target'])
            init_len = len(df)
            df[feature] = X[feature]
            df['target'] = y.astype(int)
            df_na = df[df[feature].apply(lambda x: np.isnan(x)==True)]
            na_len = len(df_na)
            df.dropna(inplace=True)
            if na_len > (1 - self.min_size)*init_len:
                self.nalist.append(feature)
                continue
            big_na_part = len(df_na) > self.min_size*len(y)
            
            df[feature] = df[feature].apply(lambda x: df[df[feature] != np.inf][feature].max() + 1 if x == np.inf else x)
            df[feature] = df[feature].apply(lambda x: df[df[feature] != -np.inf][feature].min() - 1 if x == -np.inf else x)
            t = df['target']; x = df[feature]
            
            if x.nunique() > self.starts_from and self.fast == True:
                data_ = pd.concat([t, x], axis=1).sort_values(by=feature, ascending=False).reset_index(drop=True)
                bins_num = self.starts_from
                uniq = np.array([])
                for i in range(bins_num):
                    if i != bins_num - 1:
                        uniq_vals = np.concatenate((uniq, np.array([i]*(len(data_)//bins_num))))
                    else:
                        uniq_vals = np.concatenate((uniq, np.array([i]*(len(data_) - (len(data_)//bins_num)*(bins_num - 1)))))
                data_['Bin'] = uniq
                fpr, tpr, thresholds = roc_curve(data_['target'], data_['Bin'])
            else:
                fpr, tpr, thresholds = roc_curve(t, x)
            br = t.value_counts()[1]/len(t)
            #weights
            tpr_lag, fpr_lag = np.zeros((1, len(thresholds)))[0], np.zeros((1, len(thresholds)))[0]
            tpr_lag[1:], fpr_lag[1:] = tpr[:-1], fpr[:-1]
            weights = br*abs(tpr - tpr_lag) + (1 - br)*abs(fpr - fpr_lag)
            #bottom compaction
            i = 0
            bot_edge_vol = 0
            for k in range(len(weights)):
                bot_edge_vol += weights[k]
                if bot_edge_vol > self.min_size:
                    break
                i += 1
            if i > 0:
                weights = np.insert(weights, i+1, sum(weights[0:i+1]))
                for i in range(1, i+1):
                    weights = np.delete(weights, 1)
                for i in range(1, i):
                    tpr, fpr, thresholds = np.delete(tpr, 1), np.delete(fpr, 1), np.delete(thresholds, 1)
            #upper compaction
            j = -1
            top_edge_vol = 0
            for k in reversed(range(len(weights))):
                top_edge_vol += weights[k]
                if top_edge_vol > self.min_size:
                    break
                j -= 1
            if j < 0:
                weights = np.insert(weights, j, sum(weights[j:]))
                for i in range(-j):
                    weights = np.delete(weights, -1)
                for i in range(-j-1):
                    tpr, fpr, thresholds = np.delete(tpr, -2), np.delete(fpr, -2), np.delete(thresholds, -2)
            def get_loss(break_point):
                points = sorted(np.insert(break_point, 0, values=[0, len(tpr)-1]))
                loss = 0
                for i in range(self.max_bins-1):
                    k = int(points[i]); l = int(points[i+1])
                    tpr_0 = tpr[k]; fpr_0 = fpr[k]
                    tpr_1 = tpr[l]; fpr_1 = fpr[l]
                    k = (tpr_1 - tpr_0)/(fpr_1 - fpr_0 + 1e-20)
                    b = tpr_1 - k*fpr_1
                    for j in range(int(points[i]+1), int(points[i+1])):
                        tpr_appr = k*fpr[j] + b
                        loss += abs(tpr[j] - tpr_appr)
                return loss/len(fpr)
            if len(tpr) > self.max_bins-1:
                def score(params, func=get_loss):
                    br_point = []
                    for i in range(self.max_bins-2):
                        br_point.append(params['point_%i' % (i+1)])
                    return func(br_point)
                
                space = {'point_%i' % (i+1): hp.quniform('point_%i' % (i+1), 1, len(tpr)-2, 1) for i in range(self.max_bins-2)}
                if self.method == 'tpe':
                    algorithm = tpe.suggest
                elif self.method == 'atpe':
                    algorithm = atpe.suggest
                elif self.method == 'anneal':
                    algorithm = anneal.suggest
                elif self.method == 'rand':
                    algorithm = rand.suggest
                
                best = fmin(
                    score, space
                    , algo= algorithm
                    , max_evals=self.n_iter
                    , verbose = False
                    , rstate= np.random.RandomState(self.random_state)
                )
                
                opt_idxs = sorted([int(x[1]) for x in best.items()])
                
                if x.nunique() > self.starts_from and self.fast == True:
                    opt_thresholds = sorted(data_[data_.Bin==int(i)][feature].mean() for i in sorted(thresholds[int(i)] for i in opt_idxs))
                else:
                    opt_thresholds = sorted(thresholds[int(i)] for i in opt_idxs)
                loss = get_loss(opt_idxs)
            else:
                loss = None
                opt_thresholds = sorted(thresholds[1:-1])
               
            opt_thresholds = sorted(set(opt_thresholds))
            def transform(x):
                if np.isnan(x):
                    return 0
                for i in reversed(range(len(opt_thresholds))):
                    if i > 0:
                        if x >= opt_thresholds[i]:
                            return i+2
                    else:
                        if x > opt_thresholds[i]:
                            return i+2
                return 1
            tdf['woe_%s' % feature] = X[feature].apply(lambda x: transform(x))
            uniq_vals = np.unique(tdf['woe_%s' % feature])
            # print("\n\n")
            # print(feature)
            # print(uniq_vals)
            # print(opt_thresholds)
            while list(uniq_vals) != list(range(1, 2+len(opt_thresholds))) and list(uniq_vals) != list(range(2+len(opt_thresholds))):
                for i in range(2, self.max_bins):
                    if i not in uniq_vals:
                        del opt_thresholds[i-1]
                        tdf['woe_%s' % feature] = X[feature].apply(lambda x: transform(x))
                        uniq_vals = np.unique(tdf['woe_%s' % feature])
                        break
            
            q = 0
            if not big_na_part and len(df_na) > 0:
                br_na = tdf.groupby(by='woe_%s' % feature)['target'].mean()[0]
                q = tdf.groupby(by='woe_%s' % feature)['target'].mean() - br_na
                q = abs(q[q!=0])
                q = q[q==q.min()].index[0]
                tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: q if x==0 else x)
            woe_iv = {'woe':{}, 'iv': 0}
            bin_volume = tdf.groupby(by='woe_%s' % feature).count().iloc[:,0]
            target_rate = tdf.groupby(by='woe_%s' % feature)['target'].mean()
            counter = tdf.groupby(by='woe_%s' % feature)['target'].value_counts()
            for i in np.unique(tdf['woe_%s' % feature]):
                try:
                    goods = counter[i][0]
                except:
                    goods = 0.001
                try:
                    bads = counter[i][1]
                except:
                    bads = 0.001
                woe = np.log((goods/df_good)/(bads/df_bad))
                woe_iv['woe'][i] = woe
            for j in np.unique(tdf['woe_%s' % feature]):
                try:
                    goods = counter[j][0]
                except:
                    goods = 0.001
                try:
                    bads = counter[j][1]
                except:
                    bads = 0.001
                woe_iv['iv'] += woe_iv['woe'][j]*((goods/df_good) - (bads/df_bad))
            uniq_vals = np.unique(tdf['woe_%s' % feature])
            def woe_transform(x):
                for i in uniq_vals:
                    if x == i:
                        return woe_iv['woe'][i]
            tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: woe_transform(x))
            gini = abs(roc_auc_score(tdf['target'], tdf['woe_%s' % feature])*2 - 1)
            self.report[feature] = {
                "type": "NUMERICAL",
                'opt_thresholds': opt_thresholds,
                'Result bins': list(uniq_vals),
                'NaN bin': q,
                'woe': woe_iv['woe'],
                'target_rate': {i: target_rate[i] for i in uniq_vals},
                'bin_volume': {i: bin_volume[i] for i in uniq_vals},
                'iv': woe_iv['iv'],
                'gini': gini,
                'loss': loss
            }
            # except:
            #     self.errorlist.append(feature)
            #     continue
            
        return tdf.drop(columns=['target'], axis=1)
    
    def transform(self, X):
        
        self.X = X
        
        #requirements
        import pandas as pd
        import numpy as np
        from tqdm import tqdm

        try:
          import warnings
          warnings.filterwarnings('ignore')
        except ModuleNotFoundError:
          pass

        tdf = pd.DataFrame(index=X.index)
        
        for feature in tqdm(X.columns) if self.verbose else X.columns:
            
            if feature in self.skiplist:
                tdf[feature] = X[feature]
                continue

            if feature in self.catlist:

                matching = self.report[feature]['opt_thresholds']
                woe = self.report[feature]['woe']

                def woe_transform(x):
                    for i in self.report[feature]['Result bins']:
                        if x == i:
                            return woe[i]

                tdf['woe_%s' % feature] = X[feature].map(matching)
                tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: woe_transform(x))
                continue

            if feature not in self.errorlist and feature not in self.nalist and feature not in self.catlist:

                def transform(x):
                    if np.isnan(x):
                        return 0
                    for i in reversed(range(len(opt_thresholds))):
                        if i > 0:
                            if x >= opt_thresholds[i]:
                                return i + 2
                        else:
                            if x > opt_thresholds[i]:
                                return i + 2
                    return 1

                def woe_transform(x):
                    for i in self.report[feature]['Result bins']:
                        if x == i:
                            return woe[i]

                opt_thresholds = self.report[feature]['opt_thresholds']
                woe = self.report[feature]['woe']
                q = self.report[feature]['NaN bin']
                tdf['woe_%s' % feature] = X[feature].apply(lambda x: transform(x))
                tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: q if x==0 else x)
                tdf['woe_%s' % feature] = tdf['woe_%s' % feature].apply(lambda x: woe_transform(x))
            
        return tdf.dropna()