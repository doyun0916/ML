from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


class randomForest:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        rf = RandomForestClassifier(random_state=42)
        params = {
                'n_estimators': [50, 100, 150, 200],
                'criterion': ['gini', 'entropy']}
        grf = GridSearchCV(rf, params)
        grf.fit(self.x_train, self.y_train)
        print("\nBest parameters for RandomForest:", grf.best_params_)
        return grf

class logisticRegression:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        lr = LogisticRegression(solver="saga", random_state=42, penalty='elasticnet')
        params = {
                'l1_ratio': [0, 0.5, 1],
                'max_iter': [10000, 20000, 30000],
                }
        glr = GridSearchCV(lr, params)
        glr.fit(self.x_train, self.y_train)
        print("Best parameters for LogisticRegression:", glr.best_params_)
        return glr

class XGBoosting:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        xgb = XGBClassifier(booster='gbtree',
                            max_depth=8,
                            n_estimators=50,
                            random_state=42)

        params = {
                'max_depth': [5, 10, 15],
                'n_estimators': [10, 30, 60],
                }

        gxgb = GridSearchCV(xgb, params)
        gxgb.fit(self.x_train, self.y_train, eval_set=[(self.x_test, self.y_test)], early_stopping_rounds=5)
        print("Best parameters for XGBoosting:", gxgb.best_params_)
        return gxgb

class SVM:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        svmm = svm.SVC(random_state=42, probability=True)
        params = [
            {'C': [0.1, 1, 10], 'degree': [2, 3, 5], 'kernel': ['poly']},
            {'C': [0.1, 1, 10], 'gamma': [0.0001, 0.001, 0.01], 'kernel': ['linear', 'rbf']}, ]
        gsvm = GridSearchCV(svmm, params)
        gsvm.fit(self.x_train, self.y_train)
        print("Best parameters for SVM:", gsvm.best_params_)
        return gsvm


class voting:
    def __init__(self, model1, model2, model3, model4, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def train(self):
        voting_soft = VotingClassifier(
            estimators=[('model1', self.model1), ('model2', self.model2), ('model3', self.model3), ('model4', self.model4)],
            voting='soft')
        voting_soft.fit(self.x_train, self.y_train)
        return voting_soft

class stacking:
    def __init__(self, model1, model2, model3, model4, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def train(self):
        estimators = [('model1', self.model1),
                      ('model2', self.model2),
                      ('model3', self.model3),
                      ('model4', self.model4)]
        stacking_clf = StackingClassifier(estimators=estimators,
                                          final_estimator=LogisticRegression(),
                                          cv=5)
        stacking_clf.fit(self.x_train, self.y_train)
        return stacking_clf


class validation_score:
    def __init__(self, model, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y
        self.model = model

    def score(self):
        pred = self.model.predict(self.test_x)
        clf_re = classification_report(self.test_y, pred)
        print("\n", clf_re, "\n")
        accuracy = accuracy_score(self.test_y, pred)
        print("\nAccuracy:", accuracy, "\n")


class final_eval:
    def __init__(self, model_intrusion, model_traffic, test_x, test_y):
        self.mod_intrusion = model_intrusion
        self.mod_traffic = model_traffic
        self.test_x = test_x
        self.test_y = test_y

    def score(self):
        test_x_intrusion = self.test_x.iloc[:, :20]
        test_x_intrusion.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5',
                                    'pc6', 'pc7', 'pc8', 'pc9', 'pc10',
                                    'pc11', 'pc12', 'pc13', 'pc14', 'pc15',
                                    'pc16', 'pc17', 'pc18', 'pc19', 'pc20']
        test_x_traffic = self.test_x.iloc[:, 20:]
        test_x_traffic.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5',
                                  'pc6', 'pc7', 'pc8', 'pc9', 'pc10',
                                  'pc11', 'pc12', 'pc13', 'pc14', 'pc15']
        test_y_intrusion = self.test_y.iloc[:10000]
        test_y_traffic = self.test_y.iloc[10000:]
        test_y_traffic = test_y_traffic.reset_index()
        test_y_traffic.drop(columns=['index'], inplace=True)

        intru_pred = self.mod_intrusion.predict(test_x_intrusion)
        accuracy_intru = accuracy_score(test_y_intrusion, intru_pred)
        traffic_pred = self.mod_traffic.predict(test_x_traffic)
        accuracy_traf = accuracy_score(test_y_traffic, traffic_pred)
        overall_accuracy = (accuracy_intru + accuracy_traf)/2
        print("\n")
        for i in range(50):
            if intru_pred[i] == 0:
                print("Normal state\n")
            elif intru_pred[i] == 1:
                if traffic_pred[i] == 0:
                    print("Intrusion state, Not with DNS port\n")
                elif traffic_pred[i] == 1:
                    print("Alert! Intrusion state, Malicious DoH traffic sent using dns2tcp\n")
                elif traffic_pred[i] == 2:
                    print("Alert! Intrusion state, Malicious DoH traffic sent using dnscat2\n")
                elif traffic_pred[i] == 3:
                    print("Alert! Intrusion state, Malicious DoH traffic sent using iodine\n")

        print("Overall accuracy: ", overall_accuracy)


