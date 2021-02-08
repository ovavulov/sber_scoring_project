from functions import *
import json, os
cfg = getConfig()
currMode = cfg["STAGE"]["DEPLOY"]["MODE"]

if cfg["STAGE"]["DEPLOY"]["STATUS"] == "TODO":
    if cfg["STAGE"]["DEPLOY"]["MODEL"] == 0:
        print("\n\nModel to deploy is not defined. Choose the appropriate model from your drafts\n\n")
        resultReport = getResults(cfg)
        models = [int(name.split("_")[1]) for name in list(resultReport.index) if "model_" in name]
        fullshow = input("To show instable models? (y/n):")
        assert fullshow in ["y", "n"]
        if fullshow == "y":
            print(resultReport)
        else:
            print(resultReport[resultReport["STABILITY"] != "FAILED"])
        try:
            modelToDeploy = int(input("\n\nType model number and press Enter: "))
            if modelToDeploy not in models:
                print("This model doesn't exist")
            else:
                cfg["STAGE"]["DEPLOY"]["MODEL"] = modelToDeploy
                makeDeploy(cfg)
                if currMode != "TEST":
                    cfg["STAGE"]["DEPLOY"]["STATUS"] = "DONE"
                    json.dump(cfg, open(os.path.join(".", "config.json"), "w"))
        except ValueError:
            print("Bad input")
    else:
        makeDeploy(cfg)
        if currMode != "TEST":
            cfg["STAGE"]["DEPLOY"]["STATUS"] = "DONE"
            json.dump(cfg, open(os.path.join(".", "config.json"), "w"))
else:
    print(f"\n\nModel was already deployed\n\n")

