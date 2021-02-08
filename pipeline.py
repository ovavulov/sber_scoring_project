from functions import *
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

cfg = getConfig()

if cfg["STAGE"]["PIPELINE"]["STATUS"] == "TODO":

    np.random.seed(cfg["RS"])
    init = datetime.now()
    currMode = cfg["STAGE"]["PIPELINE"]["MODE"]
    verbose = cfg["STAGE"]["PIPELINE"]["VERBOSE"]
    impData = getData(cfg, long=False, mode=currMode, verb=verbose)
    pools, cat_columns_idxs = makePools(
        cfg, impData, verb=verbose
    )
    makePipeline(cfg, pools, cat_columns_idxs, mode=currMode)
    fin = datetime.now()
    print(f"\n\nPipeline building time: {fin - init}\n\n")

else:
    print(f"\n\nPipeline building was already done\n\n")