import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

max_training_rows = 10000
target = 'buggy'
date_column = 'author_date'
id_column = 'commit_id'
ignored_days = 90
seconds_in_day = 60*60*24

experience = ['exp', 'rexp', 'sexp']
history = ['ndev', 'nuc', 'age']
size = ['la', 'ld', 'lt']
diffusion = ['ns', 'nd', 'nf', 'entropy']

features = experience + history + size + diffusion


def main():
    df = pd.read_csv("data/mybatis-3_5ffe1bc68e3f65b96a5eb9e2_metrics.csv")

    latest_date = df[date_column].max()
    oldest_ignored_date = latest_date - (ignored_days * 86400)
    df = df.loc[df[date_column] < oldest_ignored_date]

    df = df.dropna(subset=features + [target], axis=0)
    df = df.sort_values(date_column, ascending=False)
    df = df[:max_training_rows]

    X = df.loc[:, features]
    y = df.loc[:, target].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,)
    print("Training on {} rows, testing on {} rows".format(
        len(X_train), len(X_test))
    )

    param_ranges = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=2000, num=10)],
        'bootstrap': [True, False],
        'max_depth': [int(x) for x in np.linspace(5, 100, num=10)],
        'max_features': ['auto', None]
    }

    best_params = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                     param_distributions=param_ranges,
                                     scoring='roc_auc',
                                     cv=5,
                                     verbose=1,
                                     n_jobs=10).fit(X_train, y_train).best_params_
    print("best hyperparameters:", best_params)

    model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                   bootstrap=best_params['bootstrap'],
                                   max_depth=best_params['max_depth'],
                                   max_features=best_params['max_features'])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)


if __name__ == '__main__':
    main()
