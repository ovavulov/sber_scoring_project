# sber_scoring_project
Flexible pipeline for binary classification task automated solving

The model is built in four stages by running the python scripts:

1) __preselection.py__
Downloading a limited sample with a full set of features, highlighting the most significant features

2) __pipeline.py__
Downloading a complete sample with selected features, evaluating the quality of various pipelines modeling: sampling + gradient boosting, selecting the most effective pipeline

3) __fineselection.py__
Additional assessment of the importance of features

3) __finetuning.py__
Final feature selection, final tests, results saving
