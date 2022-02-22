# generate ranndom duplicate integer
aa = np.random.permutation(100)[:95]
aa.sort()

# export random float
import pandas as pd
import numpy as np
r = []
for i in range(1819):
    r.append(np.random.uniform(0,1))
export = pd.DataFrame(r)
export.to_csv('risky.csv')

# export random integer
import pandas as pd
import numpy as np
r = []
for i in range(1819):
    r.append(np.random.randint(0,2))
export = pd.DataFrame(r)
export.to_csv('risky.csv')



#  https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3

# calculate accuracy
from sklearn.metrics import accuracy_score
print(f"Accuracy of the classifier is: {accuracy_score(y_test, predictions)}")


# calcuate confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
print(confusion_matrix(y_test, predictions))
plot_confusion_matrix(classifier, X_test, y_test)
plt.show()


# calculate precision
from sklearn.metrics import precision_score
print(f"Precision Score of the classifier is: {precision_score(y_test, predictions)}")

# calculate recall
from sklearn.metrics import recall_score
print(f"Recall Score of the classifier is: {recall_score(y_test, predictions)}")

# calcuate F1 score
from sklearn.metrics import f1_score
print(f"F1 Score of the classifier is: {f1_score(y_test, predictions)}")




# plot auc-roc curve
from sklearn.metrics import roc_curve, auc

class_probabilities = classifier.predict_proba(X_test)
preds = class_probabilities[:, 1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



# how to adjust threshold
from sklearn.preprocessing import binarize
y_pred_class = binarize(y_pred_prob, 0.3)[0]

# plot histogram
import matplotlib.pyplot as plt
plt.hist(data.target) 




# plot histogram 
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(437)

num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=True)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

# plot historgram using seaborn   https://data-flair.training/blogs/python-histogram-python-bar/
import seaborn as sn
df=sn.load_dataset('iris')
sn.distplot(df['sepal_length'])
plt.show()



# plot confusion matrix
from autoviml.Auto_NLP import plot_confusion_matrix
plot_confusion_matrix(y, y)


# plot heatmap
plt.figure(figsize = (10,7))
sns.heatmap(aa, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# plot histogram using pandas
label.hist()

# calcuate the correlation
y_test = np.expand_dims(y_test, axis=1)    
coeff = np.corrcoef(predict_y, y_test)




# NB using pgm libaray
import numpy as np
import pandas as pd
from pgmpy.models import NaiveBayes
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
model = NaiveBayes()
values = pd.DataFrame(np.random.uniform(low=0, high=2, size=(1000, 5)),
                      columns=['A', 'B', 'C', 'D', 'E'])
print(values)
model.fit(values, 'E')
model.get_cpds()
model.edges()
infer = VariableElimination(model)
# g_dist = infer.query(['E'])
g_dist = infer.query(['E'], evidence={'A':0.21, 'B':0, 'C':0.212, 'D':0.7})
print(g_dist)

#####################################################################################
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
student = BayesianNetwork([('diff', 'grades'), ('intel', 'grades')])
grades_cpd = TabularCPD('grades', 3, [[0.1,0.1,0.1,0.1,0.1,0.1],
                                      [0.1,0.1,0.1,0.1,0.1,0.1],
                                      [0.8,0.8,0.8,0.8,0.8,0.8]],
                        evidence=['diff', 'intel'], evidence_card=[2, 3])
student.add_cpds(grades_cpd)
print(student)



# categorical NB
from sklearn.naive_bayes import CategoricalNB
NB_model = CategoricalNB()
NB_model.fit(features, label)
y = NB_model.predict(features)













# data preprocessing using PGM library

from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from IPython.core.display import display, HTML
from pgmpy.models import BayesianNetwork
model = BayesianNetwork([('risky', 'value_rate_overdue'), ('risky','client_size'), ('risky', 'seniority_level')])

model.cpds = []
model.fit(data=df,
          estimator=BayesianEstimator,
          prior_type="BDeu",
          equivalent_sample_size=5,
          complete_samples_only=True)

print(f'Check model: {model.check_model()}\n')
for cpd in model.get_cpds():
    print(f'CPT of {cpd.variable}:')
    print(cpd, '\n')


TIERS_NUM = 3

def boundary_str(start, end, tier):
    return f'{tier}: {start:+0,.2f} to {end:+0,.2f}'

def relabel(v, boundaries):
    if v >= boundaries[0][0] and v <= boundaries[0][1]:
        return boundary_str(boundaries[0][0], boundaries[0][1], tier='A')
    elif v >= boundaries[1][0] and v <= boundaries[1][1]:
        return boundary_str(boundaries[1][0], boundaries[1][1], tier='B')
    elif v >= boundaries[2][0] and v <= boundaries[2][1]:
        return boundary_str(boundaries[2][0], boundaries[2][1], tier='C')
    else:
        return np.nan

def get_boundaries(tiers):
    prev_tier = tiers[0]
    boundaries = [(prev_tier[0], prev_tier[prev_tier.shape[0] - 1])]
    for index, tier in enumerate(tiers):
        if index is not 0:
            boundaries.append((prev_tier[prev_tier.shape[0] - 1], tier[tier.shape[0] - 1]))
            prev_tier = tier
    return boundaries

new_columns = {}
for i, content in enumerate(train_NB.items()):
    (label, series) = content
    values = np.sort(np.array([x for x in series.tolist() if not np.isnan(x)] , dtype=float))
    if values.shape[0] < TIERS_NUM:
        print(f'Error: there are not enough data for label {label}')
        break
    boundaries = get_boundaries(tiers=np.array_split(values, TIERS_NUM))
    new_columns[label] = [relabel(value, boundaries) for value in series.tolist()]

df = pd.DataFrame(data=new_columns)
nodes = ['risky','value_rate_overdue','client_size','seniority_level']
df.columns = nodes
# df.index = range(years["min"], years["max"] + 1)
df.head(10)