# sber_scoring_project
Flexible pipeline for binary classification task automated solving

The model is built in four stages by running the python scripts:

1) preselection.py
Downloading a limited sample with a full set of features, highlighting the most significant features

2) pipeline.py
Downloading a complete sample with selected features, evaluating the quality of various pipelines modeling: sampling + gradient boosting, selecting the most effective pipeline

3) fineselection.py
Additional assessment of the importance of features

3) finetuning.py
Final feature selection, final tests, results saving
