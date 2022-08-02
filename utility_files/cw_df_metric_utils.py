# -*- coding: utf-8 -*-

import spacy
import math
import decimal
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from spacytextblob.spacytextblob import SpacyTextBlob
from flair.models import TextClassifier
from flair.data import Sentence
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, tqdm_pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#!python -m spacy download en_core_web_sm

DEBUG = False

def expandColumn(df, columnName, showProgress=False, progress=500, spacyType='en_core_web_sm'):
    nlp = spacy.load(spacyType)
    totalRecords = len(df)
    for i, row in tqdm(df.iterrows(), desc='Expanding column: ' + columnName):
        if i % progress == 0 and showProgress:
            print(str(i) + " " + str("{:.1%}".format(i/totalRecords)) + " records processed for " + str(columnName))
        if (row[columnName] and len(str(row[columnName])) < 1000000):
            doc = nlp(str(row[columnName]))
            adjectives = []
            nouns = []
            verbs = []
            lemmas = []

            for token in doc:
                if not token.is_stop:
                    lemmas.append(token.lemma_)
                    if token.pos_ == "ADJ":
                        adjectives.append(token.lemma_)
                    if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                        nouns.append(token.lemma_)
                    if token.pos_ == "VERB":
                        verbs.append(token.lemma_)

            df.at[i, columnName + "_lemma"] = " ".join(lemmas)
            df.at[i, columnName + "_nouns"] = " ".join(nouns)
            df.at[i, columnName + "_adjectives"] = " ".join(adjectives)
            df.at[i, columnName + "_verbs"] = " ".join(verbs)
            df.at[i, columnName + "_nav"] = " ".join(nouns + adjectives + verbs)

def calcTextBlobSentiment(df, columnName, showProgress=False, progress=500, spacyType='en_core_web_sm'):
    nlp = spacy.load(spacyType)
    nlp.add_pipe('spacytextblob')

    totalRecords = len(df)
    for i, row in tqdm(df.iterrows(), desc='Calculating TextBlob Sentiment'):
        if i % progress == 0 and showProgress:
            print(str(i) + " " + str("{:.1%}".format(i / totalRecords)) + " records processed for " + str(columnName))
        if (row[columnName] and len(str(row[columnName])) < 1000000):
            doc = nlp(str(row[columnName]))

            df.at[i, columnName + "_tb_pol"] = doc._.polarity
            df.at[i, columnName + "_tb_subj"] = doc._.subjectivity
            df.at[i, columnName + "_tb_tokens"] = len(doc) #tokens including punctuation etc
            df.at[i, columnName + "_tb_length"] = len(str(doc)) #length of text including spaces

def isNaN(num):
    #TODO help me FIXME
    return num!= num

def binSpacyPolarity(polarity, numBins):
  if isNaN(polarity):
      return None

  if polarity == -1:
    return 1
  else:
    return math.ceil(((polarity + 1) / 2) * numBins)

def binPolarity(df, columnName, numBins=5):
    tqdm.pandas()
    tDf = df.copy()
    tDf[columnName + '_norm'] = tDf.progress_apply(
        lambda x: binSpacyPolarity(x[columnName], numBins=numBins), axis=1)
    return tDf

def binPositiveNegative(val):
    if isNaN(val):
        return None

    if val > 0:
        return 1
    else:
        return 0

def binPolarityPosNeg(df, columnName):
    tqdm.pandas()
    tDf = df.copy()
    tDf[columnName + '_posneg'] = tDf.progress_apply(
        lambda x: binPositiveNegative(x[columnName]), axis=1)
    return tDf

def splitSpacySentences(df, columnName, showProgress=False, progress=500):
  nlp = spacy.load('en_core_web_sm')
  nlp.add_pipe('spacytextblob')

  split1=[]
  split2=[]
  split3=[]
  split4=[]
  split5=[]

  totalRecords = len(df)
  for i, row in tqdm(df.iterrows(), desc="Splitting sentences by polarity"):
      #progress notification
      if i % progress == 0 and showProgress:
          print(str(i) + " " + str("{:.1%}".format(i/totalRecords)) + " records processed for " + str(columnName))

      #is our sentence ok to process
      if (row[columnName] and len(str(row[columnName])) < 1000000):
          doc = nlp(str(row[columnName]))
          assert doc.has_annotation("SENT_START")

      #process sentences in document
      for sent in doc.sents:
          sentDoc = nlp(str(sent.text))
          #print(sent.text + ' (pol:' + str(sentDoc._.polarity) + ', subj:' + str(sentDoc._.subjectivity) + ')')
          polBin = binSpacyPolarity(sentDoc._.polarity, 5)
          if polBin == 1:
            split1.append(sent.text)
          elif polBin == 2:
            split2.append(sent.text)
          elif polBin == 3:
            split3.append(sent.text)
          elif polBin == 4:
            split4.append(sent.text)
          elif polBin == 5:
            split5.append(sent.text)
          else:
            print("Error: spacy sentence split found sentiment out of range")

      df.at[i, columnName + "_tb_star1"] = " ".join(split1)
      df.at[i, columnName + "_tb_star2"] = " ".join(split2)
      df.at[i, columnName + "_tb_star3"] = " ".join(split3)
      df.at[i, columnName + "_tb_star4"] = " ".join(split4)
      df.at[i, columnName + "_tb_star5"] = " ".join(split5)

def calcFlairSentiment(doc, classifier):
  if len(doc) == 0:
    return
  
  sentence = Sentence(doc)

  classifier.predict(sentence)

  value = sentence.labels[0].to_dict()['value']
  if value == 'POSITIVE':
      return sentence.to_dict()['labels'][0]['confidence']
  else:
      return -(sentence.to_dict()['labels'][0]['confidence'])


def flairSentimentEncode(df, columnName):
    tqdm.pandas()
    classifierName = 'en-sentiment'
    print("Loading FLAIR text classifier: " + classifierName)
    classifier = TextClassifier.load(classifierName)
    print("FLAIR text classifier has been loaded")
    print("Generating FLAIR sentiments")
    df[columnName + '_flairSent'] = df.progress_apply(lambda x: calcFlairSentiment(x[columnName], classifier), axis=1)
    print("FLAIR sentiments completed")

def columnEncode(data,
                 columnName,
                 transformerType,
                 colSuffix):
    tqdm.pandas()

    print("Loading sentence transformer: " + transformerType)
    model_enc = SentenceTransformer(transformerType)
    print(f'{transformerType} sentence transformer has been loaded')
    print('Generating encodings')
    data[columnName + colSuffix] = data.progress_apply(lambda x: model_enc.encode(x[columnName]), axis=1)
    print(f'{transformerType} encodings completed')


def getBertEncodeFrame(df, bertColumn, uniqueColumn, otherColumns=None, colPrefix='c'):
    addCol = [uniqueColumn]
    addCol = addCol + otherColumns

    numpy_data = np.array(df[bertColumn].to_list())
    numpy_index = df[uniqueColumn].to_list()

    dfExp = pd.DataFrame(data=numpy_data, index=numpy_index)
    dfExp.reset_index(inplace=True)
    dfExp.rename(columns={'index': uniqueColumn}, inplace=True)
    for colname in dfExp.columns:
        if colname != uniqueColumn:
            dfExp.rename(columns={colname: colPrefix + str(colname)}, inplace=True)

    if len(otherColumns) > 0:
        dfOth = df[df.columns.intersection(addCol)]
        dfRet = pd.merge(dfExp, dfOth, how='inner', on=uniqueColumn)
        return dfRet
    else:
        return dfExp


def plotConfusionMatrix(conf_matrix, axis_labels, titleSuffix, cmap='mako',plotsize=5):
    ax = sns.heatmap(conf_matrix,annot=True, fmt='d', cmap=cmap,xticklabels=axis_labels, yticklabels=axis_labels)

    if plotsize == 5:
        sns.set(rc={'figure.figsize': (20, 8)})
    elif plotsize == 4:
        sns.set(rc={'figure.figsize': (15, 8)})
    elif plotsize == 3:
        sns.set(rc={'figure.figsize': (10, 8)})
    elif plotsize == 2:
        sns.set(rc={'figure.figsize': (8, 8)})
    elif plotsize == 1:
        sns.set(rc={'figure.figsize': (4, 8)})
    else:  # Should be size 1
        # should only be one but catch it and default to size 1
        sns.set(rc={'figure.figsize': (4, 4)})

    plt.title(f'Confusion Matrix: {titleSuffix}', fontsize = 20) # title with fontsize 20
    plt.xlabel('Predicted', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Actual', fontsize = 15) # y-axis label with fontsize 15
    plt.show()


def showTestReport(df, colNameActual, colNamePredict, axisLabels, chartTitle):
  results = metrics.classification_report(pd.to_numeric(df[colNameActual]).to_list(),
                                          df[colNamePredict].to_list(),
                                          zero_division=0)
  print(results)

  cm = confusion_matrix(np.array(pd.to_numeric(df[colNameActual])).reshape(-1, 1),
                        np.array(pd.to_numeric(df[colNamePredict])).reshape(-1, 1)
                      )
  plotConfusionMatrix(cm, axisLabels, chartTitle)


def createBertModel(df, bertColumn, uniqueColumn, targetColumn, classifier, featureFilter=None):
    dfBert = getBertEncodeFrame(df=df,
                                bertColumn=bertColumn,
                                uniqueColumn=uniqueColumn,
                                otherColumns=[targetColumn]
                               )

    #If no filter specified declare it empty
    if featureFilter==None:
        featureFilter = []

    #Get Y value from dataframe
    Y = np.array(dfBert[targetColumn])

    #Drop unneeded columns for model training
    dfBert.drop([uniqueColumn, targetColumn], axis=1, inplace=True)
    if len(featureFilter) > 0:
        display('featureFilter length > 0')
        dfBert.drop(dfBert.columns[featureFilter],axis=1, inplace=True)
    display(dfBert.shape)
    # Get X Value from dataframe
    X = dfBert.to_numpy()

    # split data into train and test sets
    seed = 7
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # fit model no training data
    model = classifier
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)

    # make a dataframe for the results
    tDf = pd.DataFrame(data=y_test, columns=['y_test'])
    tDf['y_pred'] = y_pred.tolist()

    plotModelFeatureImportance(model)

    return model, tDf

#plots importance of features in given model
def plotModelFeatureImportance(model,
                               startValue=0.0001,
                               increment=0.0001,
                               upperValue=0.01,
                               returnAbove=0.002):
    xAxisLabel = 'xAxisVal'
    recCountLabel = 'recCount'
    valueLabel = 'value'
    dx = startValue

    # calc rounding value
    d = decimal.Decimal(str(startValue))
    roundValue = d.as_tuple().exponent * -1

    # list of values for dataframe and comparison

    #Create a dataframe with feature importances


    impDf = pd.DataFrame(data=model.feature_importances_, columns=[valueLabel])
    impDf.reset_index(inplace=True)
    impDf.rename(columns={'index': 'feature'}, inplace=True)

    xAxisVal = [0]

    while dx <= upperValue:
        # add to the list of xAxisValues
        xAxisVal.append(dx)

        dx += increment
        # round value included due to errors in FP addition
        dx = round(dx, roundValue)

    # turn list into dataframe
    tDf = pd.DataFrame(xAxisVal, columns=[xAxisLabel])

    # Add in column for number of features <= that value
    tDf[recCountLabel] = tDf.apply(lambda x:
                                    len(impDf.loc[impDf[valueLabel] >= x[xAxisLabel]]),
                                    axis=1
                                    )
    tDf.plot(x=xAxisLabel, y=recCountLabel)

    #return a list to be used to filter features
    #for next model run
    tDf2 = impDf.loc[impDf[valueLabel] < returnAbove].copy()

    return tDf2

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=[0.1, 0.2, 0.5, 1.0]
    ):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        verbose=4
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

