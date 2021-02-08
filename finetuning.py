from functions import *
import numpy as np, os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

cfg = getConfig()

params = cfg["STAGE"]["TUNING"]
assert params["STATUS"] in ["TODO", "DONE"]

if cfg["STAGE"]["TUNING"]["STATUS"] == "TODO":

    np.random.seed(cfg["RS"])
    init = datetime.now()
    currMode = cfg["STAGE"]["TUNING"]["MODE"]
    verbose = cfg["STAGE"]["TUNING"]["VERBOSE"]
    impData = getData(cfg, long=False, mode=currMode, verb=verbose)
    pools, cat_columns_idxs = makePools(
        cfg, impData, verb=verbose, lgb=cfg["STAGE"]["PIPELINE"]["RESULT"]["MODEL"]=="LIGHTGBM"
    )
    makeFineTuning(cfg, pools, cat_columns_idxs, mode=currMode)
    fin = datetime.now()
    print(f"\n\nFinetuning time: {fin - init}\n\n")

else:
    print(f"\n\nFinetuning was already done\n\n")