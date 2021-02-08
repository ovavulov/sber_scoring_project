from functions import *
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

cfg = getConfig()

if cfg["STAGE"]["PRESEL"]["STATUS"] == "TODO":

    np.random.seed(cfg["RS"])
    init = datetime.now()
    setProject(cfg)
    currMode = cfg["STAGE"]["PRESEL"]["MODE"]
    verbose = cfg["STAGE"]["PRESEL"]["VERBOSE"]
    fullData = getData(cfg, long=True, mode=currMode, verb=verbose)
    pools, cat_columns_idxs = makePools(
        cfg, fullData, verb=verbose, lgb=False
    )
    makePreselection(cfg, pools, cat_columns_idxs, mode=currMode)
    fin = datetime.now()
    print(f"\n\nPreselection time: {fin - init}\n\n")

else:
    print(f"\n\nPreselection was already done\n\n")