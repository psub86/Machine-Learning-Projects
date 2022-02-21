from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, cross_validate
import pandas as pd
import numpy as np

from sklearn.svm import SVC


# AM BESTEN MIT DEM NOTEBOOK ARBEITEN; DAS IST BESSER BESCHRIEBEN UND HAT BESSER AUSGABEN


df = pd.read_csv("accidents2019.csv")

gnb = GaussianNB()
knc = KNeighborsClassifier()
rfc = RandomForestClassifier()

clfs = [gnb, knc, rfc]
df_highway = df.loc[df["road_cat"] == 1]

X, y = df_highway.drop(columns="Severity"), df_highway["Severity"]

print(PCA().fit(X).explained_variance_ratio_)

# Die ersten zwei Komponenten erklären bereits 99% der Varianz, wir können uns also auf diese beiden beschränken:

X = PCA(n_components=2).fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scores = []
for clf in clfs:
    clf.fit(X_train, y_train)
    cores.append((str(clf), clf.score(X_train, y_train), clf.score(X_test, y_test)))

print("Scores", scores)


sss = StratifiedShuffleSplit(n_splits=2, test_size=0.33, random_state=0)
X, y = df_highway.drop(columns="Severity").to_numpy(), df_highway["Severity"].to_numpy()

train_index, test_index = next(sss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

scores = []
for clf in clfs:
    clf.fit(X_train, y_train)
    cores.append((str(clf), clf.score(X_train, y_train), clf.score(X_test, y_test)))

print("Scores mit stratifizierten Trainingsmengen", scores)

cv_results = []

for clf in clfs:
    cv_results.append((str(clf), cross_validate(clf, X_test, y_test, cv=5)["test_score"].mean()))

print("Scores-Mittelwerte der CV", cv_results)

# Die Crossvaldidation zeigt, dass die oben berechneten Scores stimmen.


# gridsearch_scores:
gs_scores = []

# ## RandomForest
params_rf = {
    'n_estimators': [10, 100, 250, 500],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 6, 12],
    'n_jobs': [-1],
}

gsc_rf = GridSearchCV(RandomForestClassifier(), params_rf)
gsc_rf.fit(X_train, y_train)
bprfc = RandomForestClassifier(**gsc_rf.best_params_)

bprfc.fit(X_train, y_train)
bprfc.score(X_train, y_train)
bprfc.score(X_test, y_test)

gs_scores.append((bprfc.score(X_train, y_train), bprfc.score(X_test, y_test)))

# ## KNN

params_kn = {
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree'],
    'p': [1, 2, 10],
    'n_jobs': [-1],
}

gsc_kn = GridSearchCV(KNeighborsClassifier(), params_kn)
gsc_kn.fit(X_train, y_train)

gsc_kn.best_params_
bpknc = KNeighborsClassifier(**gsc_kn.best_params_)

bpknc.fit(X_train, y_train)

print(bpknc.score(X_train, y_train))
print(bpknc.score(X_test, y_test))

gs_scores.append((bpknc.score(X_train, y_train), bpknc.score(X_test, y_test)))

print(gs_scores)

# # VotingClassifier

clfs, clfs_names = [GaussianNB(), bpknc, bprfc], ["GNB", "KNN", "RFC"]

list(zip(clfs_names, clfs))

eclf1 = VotingClassifier(estimators=list(zip(clfs_names, clfs)), voting='hard')
eclf1.fit(X_train, y_train)
print(eclf1.score(X_train, y_train), eclf1.score(X_test, y_test))

eclf2 = VotingClassifier(estimators=list(zip(clfs_names, clfs)), voting='soft')

eclf2.fit(X_train, y_train)
print(eclf2.score(X_train, y_train), eclf2.score(X_test, y_test))

# ## Versuch mit SVC

clfs.append(SVC(C=1, probability=True))
clfs_names.append("SVC")

eclf3 = VotingClassifier(estimators=list(zip(clfs_names, clfs)), voting='soft')
eclf3.fit(X_train, y_train)

print(eclf3.score(X_train, y_train), eclf3.score(X_test, y_test))

# ## gewichtes Voting
clfs, clfs_names = [GaussianNB(), bpknc, bprfc], ["GNB", "KNN", "RFC"]
eclf4 = VotingClassifier(estimators=list(zip(clfs_names, clfs)), voting='soft', weights=[0.5, 0.5, 1.5])
eclf4.fit(X_train, y_train)

print(eclf4.score(X_train, y_train), eclf4.score(X_test, y_test))
