from sklearn.metrics import roc_auc_score, f1_score, r2_score, make_scorer, confusion_matrix


class ClassificationScoreInterface():
    def score(self, x, y, score_type='rocauc'):
        if score_type == 'rocauc':
            return make_scorer(roc_auc_score, needs_proba=True)(self, x, y)
        elif score_type == 'f1':
            return make_scorer(f1_score)(self, x, y)
    
    
class RegressorScoreInterface():
    def score(self, x, y):
        return make_scorer(r2_score)(self, x, y)