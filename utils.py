from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    report = {}
    
    for name, model in models.items():
        grid = GridSearchCV(model, params.get(name, {}), cv=3, scoring='f1_weighted', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        report[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "best_params": grid.best_params_
        }

    return report
