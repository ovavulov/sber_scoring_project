from functions import *
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

cfg = getConfig()

if cfg["STAGE"]["FINESEL"]["STATUS"] == "TODO":

    np.random.seed(cfg["RS"])
    init = datetime.now()
    currMode = cfg["STAGE"]["FINESEL"]["MODE"]
    verbose = cfg["STAGE"]["FINESEL"]["VERBOSE"]
    impData = getData(cfg, long=False, mode=currMode, verb=verbose)
    pools, cat_columns_idxs = makePools(
        cfg, impData, verb=verbose, lgb=cfg["STAGE"]["PIPELINE"]["RESULT"]["MODEL"]=="LIGHTGBM"
    )
    makeFineSelection(cfg, pools, cat_columns_idxs, mode=currMode)
    fin = datetime.now()
    print(f"\n\nFineselection time: {fin - init}\n\n")

else:
    print(f"\n\nFineselection was already done\n\n")

