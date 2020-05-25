from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix
from report import log


def evaluate(test):
    test['correct'] = test['location_int'] == test['pred_int']
    log("confusion_matrix: " +
        str(confusion_matrix(test['location_int'], test['pred_int'])))
    log("accuracy: " +
        str(accuracy_score(test['location_int'], test['pred_int'])))
    log("log_loss: " + str(log_loss(test['location_int'], test['pred'])))
    return test
