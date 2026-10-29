"""Reproducible Adult income benchmark. Run from the project root."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, make_scorer)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['age','workclass','fnlwgt','education','education_num','marital_status',
           'occupation','relationship','race','sex','capital_gain','capital_loss',
           'hours_per_week','native_country','income']
NUMERIC = ['age','education_num','capital_gain','capital_loss','hours_per_week']
BASE = 'https://archive.ics.uci.edu/static/public/2/'


def load_data():
    """Cache original UCI files; preserve official training/test partition."""
    import zipfile
    from io import BytesIO
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    if not all((folder / name).exists() for name in ['adult.data', 'adult.test']):
        print('Downloading Adult dataset from UCI...', flush=True)
        with urlopen(BASE + 'adult.zip', timeout=90) as response:
            archive = zipfile.ZipFile(BytesIO(response.read()))
        for name in ['adult.data', 'adult.test']:
            (folder / name).write_bytes(archive.read(name))
    frames = []
    for name in ['adult.data', 'adult.test']:
        frame = pd.read_csv(folder / name, names=COLUMNS, comment='|',
                            skipinitialspace=True, na_values='?')
        frame['income'] = frame['income'].str.strip().str.rstrip('.')
        if not frame['income'].isin(['<=50K', '>50K']).all():
            raise ValueError('Unexpected or missing income labels')
        frame['income'] = (frame['income'] == '>50K').astype(int)
        frames.append(frame)
    # Remove exact full-record duplicates, including cross-partition overlap.
    train = frames[0].drop_duplicates().reset_index(drop=True)
    # pandas hashing handles missing values consistently across the two files.
    train_hash = set(pd.util.hash_pandas_object(train, index=False))
    test = frames[1].drop_duplicates()
    test = test.loc[~pd.util.hash_pandas_object(test, index=False).isin(train_hash)]
    audit = {'raw_train_rows': len(frames[0]), 'raw_test_rows': len(frames[1]),
             'clean_train_rows': len(train), 'clean_test_rows': len(test),
             'sha256': {n: hashlib.sha256((folder/n).read_bytes()).hexdigest()
                        for n in ['adult.data','adult.test']}}
    return train, test.reset_index(drop=True), audit


def preprocess(X):
    categorical = [c for c in X.columns if c not in NUMERIC]
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                              ('encode', OneHotEncoder(handle_unknown='ignore',
                                                       sparse_output=False))]), categorical)])


def models(seed):
    return {
        'Baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=seed),
        'K Nearest Neighbors': KNeighborsClassifier(n_neighbors=15, n_jobs=1),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=1, random_state=seed),
        'Extra Trees': ExtraTreesClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=1, random_state=seed),
        'Gaussian Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(C=1, kernel='rbf', random_state=seed),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=seed),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=seed),
        'Histogram Gradient Boosting': HistGradientBoostingClassifier(random_state=seed),
    }


def scores(y, pred, score):
    return dict(accuracy=accuracy_score(y,pred), balanced_accuracy=balanced_accuracy_score(y,pred),
                precision=precision_score(y,pred,zero_division=0), recall=recall_score(y,pred,zero_division=0),
                f1=f1_score(y,pred,zero_division=0), roc_auc=roc_auc_score(y,score),
                average_precision=average_precision_score(y,score))


def decision(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:,1]
    return model.decision_function(X)


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Stratified 3000/1500-row demonstration; not final results')
    parser.add_argument('--tune', action='store_true', help='Tune top two non-baseline models on training CV only')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=1, help='CV parallel workers; start with 1 on a laptop')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error('--folds must be at least 2')
    out = ROOT / 'results' / ('quick' if args.quick else 'full')
    out.mkdir(parents=True, exist_ok=True)
    train, test, audit = load_data()
    if args.quick:
        train, _ = train_test_split(train, train_size=min(3000,len(train)-1), stratify=train.income, random_state=args.seed)
        test, _ = train_test_split(test, train_size=min(1500,len(test)-1), stratify=test.income, random_state=args.seed)
    # fnlwgt is a survey weight, not used as a predictor; education is redundant.
    X = train.drop(columns=['income','fnlwgt','education'])
    y = train.income
    Xt = test[X.columns]
    yt = test.income
    train.isna().sum().to_csv(out/'missing_values.csv')
    train.income.value_counts().rename(index={0:'<=50K',1:'>50K'}).plot.bar(title='Training income distribution')
    plt.ylabel('Records'); save_plot(out/'income_distribution.png')
    train.boxplot(column='hours_per_week', by='income')
    plt.suptitle(''); plt.title('Training work hours by income (0: <=50K; 1: >50K)')
    save_plot(out/'hours_by_income.png')
    cv = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(X,y))
    metrics = ['accuracy','balanced_accuracy','precision','recall','f1','roc_auc','average_precision']
    candidates = {}
    rows, fold_rows = [], []
    for name, estimator in models(args.seed).items():
        print('Cross-validating:', name, flush=True)
        pipe = Pipeline([('preprocess',preprocess(X)), ('model',estimator)])
        result = cross_validate(pipe, X,y, cv=cv, scoring={m: (make_scorer(precision_score, zero_division=0) if m=='precision' else m) for m in metrics}, n_jobs=args.jobs, error_score='raise')
        row = {'model':name, 'cv_fit_seconds_mean':result['fit_time'].mean()}
        for metric in metrics:
            row['cv_'+metric+'_mean'] = result['test_'+metric].mean()
            row['cv_'+metric+'_std'] = result['test_'+metric].std(ddof=1)
        for i in range(args.folds):
            fold_rows.append({'model':name,'fold':i+1,**{m:result['test_'+m][i] for m in metrics}})
        rows.append(row); candidates[name] = pipe
        pd.DataFrame(rows).to_csv(out/'cv_results.csv', index=False)
    results = pd.DataFrame(rows).sort_values('cv_f1_mean',ascending=False)
    pd.DataFrame(fold_rows).to_csv(out/'cv_folds.csv',index=False)
    chosen = results.loc[results.model!='Baseline','model'].iloc[0]
    best = candidates[chosen]
    if args.tune:
        tuned = []
        for name in results.loc[results.model!='Baseline','model'].head(2):
            estimator = candidates[name].named_steps['model']
            param = ('C',[0.1,1,10]) if hasattr(estimator,'C') else (
                ('n_neighbors',[5,15,31]) if hasattr(estimator,'n_neighbors') else (
                ('var_smoothing',[1e-11,1e-9,1e-7]) if hasattr(estimator,'var_smoothing') else (
                ('learning_rate',[0.03,0.1,0.3]) if hasattr(estimator,'learning_rate') else
                ('max_depth',[5,12,None]))))
            search = GridSearchCV(candidates[name], {'model__'+param[0]:param[1]},
                                  scoring='f1',cv=cv,n_jobs=args.jobs,error_score='raise')
            search.fit(X,y)
            pd.DataFrame(search.cv_results_).to_csv(out/(name.replace(' ','_')+'_tuning.csv'),index=False)
            tuned.append((search.best_score_,name,search.best_estimator_))
        _, chosen, best = max(tuned,key=lambda item:item[0])
    # Selection is now frozen. Test comparisons are descriptive, not selection criteria.
    test_rows = []
    fig_roc, ax_roc = plt.subplots()
    fig_pr, ax_pr = plt.subplots()
    for name, candidate in candidates.items():
        fitted = clone(best if name==chosen else candidate)
        start=perf_counter(); fitted.fit(X,y); fit_time=perf_counter()-start
        start=perf_counter(); pred=fitted.predict(Xt); elapsed=perf_counter()-start
        score=decision(fitted,Xt)
        test_rows.append({'model':name,**scores(yt,pred,score),'fit_seconds':fit_time,
                          'predict_seconds':elapsed,'selected_by_cv':name==chosen})
        RocCurveDisplay.from_predictions(yt,score,name=name,ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(yt,score,name=name,ax=ax_pr)
        if name==chosen:
            best=fitted; best_pred=pred
        pd.DataFrame(test_rows).to_csv(out/'test_results.csv',index=False)
    for fig, ax, name in [(fig_roc,ax_roc,'roc_curves'),(fig_pr,ax_pr,'precision_recall_curves')]:
        ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(out/(name+'.png'),dpi=160); plt.close(fig)
    results.set_index('model').cv_f1_mean.sort_values().plot.barh(
        xerr=results.set_index('model').cv_f1_std.reindex(results.set_index('model').cv_f1_mean.sort_values().index),
        figsize=(9,6),title='Training cross-validation F1 (error bars: fold SD)')
    plt.xlabel('F1 (>50K)'); save_plot(out/'cv_comparison.png')
    ConfusionMatrixDisplay.from_predictions(yt,best_pred,display_labels=['<=50K','>50K'])
    plt.title(chosen+' — held-out test'); save_plot(out/'confusion_matrix.png')
    importance=permutation_importance(best,Xt,yt,scoring='f1',n_repeats=3,random_state=args.seed,n_jobs=args.jobs)
    pd.DataFrame({'feature':X.columns,'importance':importance.importances_mean,
                  'std':importance.importances_std}).sort_values('importance',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    pd.Series(importance.importances_mean,index=X.columns).sort_values().plot.barh(title='Selected model: permutation importance (test F1)')
    save_plot(out/'feature_importance.png')
    subgroup=[]
    for attribute in ['sex','race']:
        for value in Xt[attribute].dropna().unique():
            mask=(Xt[attribute]==value).to_numpy()
            actual=yt.to_numpy()[mask]; predicted=best_pred[mask]
            negatives=actual==0
            subgroup.append({'attribute':attribute,'group':value,'n':int(mask.sum()),
                'positive_count':int(actual.sum()), 'actual_positive_rate':float(actual.mean()),
                'predicted_positive_rate':float(predicted.mean()),
                'recall':recall_score(actual,predicted,zero_division=0) if actual.sum() else np.nan,
                'false_positive_rate':float(predicted[negatives].mean()) if negatives.any() else np.nan,
                'f1':f1_score(actual,predicted,zero_division=0)})
    pd.DataFrame(subgroup).to_csv(out/'subgroup_metrics.csv',index=False)
    errors=Xt.copy(); errors['actual']=yt; errors['predicted']=best_pred
    errors.loc[errors.actual!=errors.predicted].to_csv(out/'misclassifications.csv',index=False)
    joblib.dump(best,out/'best_model.joblib')
    metadata={'arguments':vars(args),'data_audit':audit,'train_rows_used':len(X),'test_rows_used':len(Xt),
              'selected_model':chosen,'selection_metric':'training CV F1',
              'python':platform.python_version(),'sklearn':sklearn.__version__,
              'pandas':pd.__version__,'numpy':np.__version__,
              'parameters':{k:str(v) for k,v in best.get_params().items()}}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\nSelected using training CV:',chosen)
    print(pd.DataFrame(test_rows)[['model','f1','roc_auc','accuracy']].to_string(index=False))
    print('\nOutputs:',out)


if __name__=='__main__':
    main()
