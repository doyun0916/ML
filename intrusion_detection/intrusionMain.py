import preprocessing as pp
import model

# data preparation
intrusion, intrusion_x_train, intrusion_x_test, intrusion_y_train, intrusion_y_test = pp.intrusion_data()
traffic, traffic_x_train, traffic_x_test, traffic_y_train, traffic_y_test = pp.traffic_data()
actual_test_data_x, actual_test_data_y = pp.overall_test_data(intrusion, traffic)

# Training with intrusion data
Rf = model.randomForest(intrusion_x_train, intrusion_y_train).train()          # RandomForest
model.validation_score(Rf, intrusion_x_test, intrusion_y_test).score()

Lr = model.logisticRegression(intrusion_x_train, intrusion_y_train).train()    # Logistic Regression
model.validation_score(Lr, intrusion_x_test, intrusion_y_test).score()

Xgb = model.XGBoosting(intrusion_x_train, intrusion_y_train, intrusion_x_test, intrusion_y_test).train()    # XGBoosting
model.validation_score(Xgb, intrusion_x_test, intrusion_y_test).score()

Svm = model.SVM(intrusion_x_train, intrusion_y_train).train()                     # SVM
model.validation_score(Svm, intrusion_x_test, intrusion_y_test).score()

Voting_soft = model.voting(Rf, Lr, Xgb, Svm, intrusion_x_train, intrusion_y_train).train()        # Voting
print("\nVoting_soft's score: \n")
model.validation_score(Voting_soft, intrusion_x_test, intrusion_y_test).score()

Stacking_intrusion = model.stacking(Rf, Lr, Xgb, Svm, intrusion_x_train, intrusion_y_train).train()   # Stacking
print("\nStacking's score: \n")
model.validation_score(Stacking_intrusion, intrusion_x_test, intrusion_y_test).score()

# Training with traffic data
Rf = model.randomForest(traffic_x_train, traffic_y_train).train()
Lr = model.logisticRegression(traffic_x_train, traffic_y_train).train()
Xgb = model.XGBoosting(traffic_x_train, traffic_y_train, traffic_x_test, traffic_y_test).train()
Svm = model.SVM(traffic_x_train, traffic_y_train).train()
Stacking_traffic = model.stacking(Rf, Lr, Xgb, Svm, traffic_x_train, traffic_y_train).train()

# Final test using both model at the same time for new 5 classification
model.final_eval(Stacking_intrusion, Stacking_traffic, actual_test_data_x, actual_test_data_y).score()
