from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


class ModelTrainer:
    def __init__(self):
        pass

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, evaluate_models_fn):
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'MultinomialNB': MultinomialNB(),
            'LinearSVC': LinearSVC()
        }

        params = {
            'LogisticRegression': {'C': [0.1, 1, 10]},
            'MultinomialNB': {'alpha': [0.1, 1.0, 10.0]},
            'LinearSVC': {'C': [0.1, 1, 10]}
        }

        report = evaluate_models_fn(X_train, X_test, y_train, y_test, models, params)

        best_model_name = max(report, key=lambda x: report[x]['f1_score'])
        best_model = models[best_model_name].set_params(**report[best_model_name]['best_params'])

        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        print(f"\nBest Model: {best_model_name}")
        print(classification_report(y_test, y_pred))

        return best_model, report
