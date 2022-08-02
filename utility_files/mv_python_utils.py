# -*- coding: utf-8 -*-

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import numpy as np
import math
from yellowbrick.target import ClassBalance


# Start dataframe functions
def exploreDataframe(data, numRecords=1):
    print("dataframe shape: " + str(data.shape))
    print("\ndataframe info: ")
    print(data.info())
    print("\nTop " + str(numRecords) + " in dataframe")
    display(data.head(numRecords))
    print("\nBottom " + str(numRecords) + " in dataframe")
    display(data.tail(numRecords))

def showUniqueColVals(series):
    i = 0
    for l in np.unique(series):
        i += 1
        print(l)
    print("\n--->Unique values: " + str(i))


# End dataframe functions


# Start generic ML functions

# Create a summary frame for us to append results
def initSummaryFrame():
    column_names = ["analysisStage",
                    "columnName",
                    "totalDocuments",
                    "totalNulls",
                    "totalCategories",
                    "upperPerc",
                    "lowerPerc",
                    "percTotalDocuments",
                    "percTotalCategories"
                    ]
    fnDf = pd.DataFrame(columns=column_names)
    return fnDf


# Add columns for upper and lower boundaries
def addUpperAndLowerPercColumns(data,
                                upperPerc,
                                lowerPerc,
                                upperPercColName='upperPerc',
                                lowerPercColName='lowerPerc'):
    fnDf = data
    fnDf[upperPercColName] = upperPerc
    fnDf[lowerPercColName] = lowerPerc
    return fnDf


# Add results data to summary frame
def appendResultsData(resultsFrame,
                      analysisStage,
                      analysisColName,
                      totalDocuments,
                      totalNulls,
                      totalCategories,
                      upperPerc,
                      lowerPerc,
                      upperPercColName='upperPerc',
                      lowerPercColName='lowerPerc',
                      displayAppendedFrame=False):
    if len(resultsFrame) == 0:
        # this is the first row (should be original)
        # add the perc total documents/columns
        percTotalDocuments = 1.0
        percTotalCategories = 1.0
    else:
        allCategories = resultsFrame['totalCategories'].values[0]
        allDocuments = resultsFrame['totalDocuments'].values[0]

        percTotalDocuments = round(totalDocuments / allDocuments, 2)
        percTotalCategories = round(totalCategories / allCategories, 2)

    new_row = {'analysisStage': analysisStage,
               'columnName': analysisColName,
               'totalDocuments': totalDocuments,
               'totalNulls': totalNulls,
               'totalCategories': totalCategories,
               upperPercColName: upperPerc,
               lowerPercColName: lowerPerc,
               'percTotalDocuments': percTotalDocuments,
               'percTotalCategories': percTotalCategories
               }
    # append row to the dataframe
    resultsFrame = resultsFrame.append(new_row, ignore_index=True)
    if displayAppendedFrame:
        print('Results appended. New frame:')
        display(resultsFrame)

    return resultsFrame


def buildReportingFrame(df,
                        analysisColName,
                        summaryColName,
                        cumulativeColName,
                        colOrderName,
                        percTotalColName,
                        percCumulativeColName,
                        ):
    tDf = df.copy()

    tDf[analysisColName] = tDf[analysisColName].astype('string')

    tDf = df.groupby([analysisColName]).size().to_frame(summaryColName)
    tDf = tDf.sort_values(by=summaryColName, ascending=False)
    tDf[cumulativeColName] = tDf[summaryColName].cumsum()

    totalRows = tDf[summaryColName].sum()

    tDf[percTotalColName] = round(tDf[summaryColName] / totalRows, 2)
    tDf[percCumulativeColName] = round(tDf[cumulativeColName] / totalRows, 2)
    tDf[colOrderName] = np.arange(len(tDf)) + 1

    tDf.reset_index(inplace=True)
    tDf.index = np.arange(1, len(tDf) + 1)

    tDf = tDf[
        [colOrderName, analysisColName, summaryColName, cumulativeColName, percTotalColName, percCumulativeColName]]

    return tDf


def showColumnSummary(df,
                      analysisColName
                      ):
    tDf = df.copy()
    totalDocuments = len(tDf)
    totalNulls = tDf[analysisColName].isnull().sum()
    totalNAs = tDf[analysisColName].isna().sum()
    totalCategories = tDf[analysisColName].nunique()

    print(f'Dataframe shape {str(df.shape)}')
    print(f'Analysis column: {analysisColName}')
    print(f'Distinct values (incl. null): {str(totalCategories)}')
    print(f'Number of na   values: {str(totalNAs)}')
    print(f'Number of null values: {str(totalNulls)}')
    print(f'Total documents in corpus: {str(totalDocuments)}', end='\n\n')


def analyzeReportingFrame(df,
                          summaryColName,
                          analysisColName,
                          resultsFrame,
                          analysisName="original",
                          upperPerc=1.0,  # use default for initial run
                          lowerPerc=0.0,  # use default for initial run
                          showRows=2):
    tDf = df.copy()

    totalDocuments = tDf[summaryColName].sum()
    totalCategories = tDf[analysisColName].nunique()
    totalNulls = tDf[analysisColName].isnull().sum()

    print(f'Top {str(showRows)} for summary of column {analysisColName}')
    display(tDf[:showRows])
    print("", end='\n\n')

    print(f'Last {str(showRows)} for summary of column {analysisColName}')
    display(tDf.tail(showRows))
    print("", end='\n\n')

    resultsFrame = appendResultsData(resultsFrame,
                                     analysisName,
                                     analysisColName,
                                     totalDocuments,
                                     totalNulls,
                                     totalCategories,
                                     upperPerc=upperPerc,
                                     lowerPerc=lowerPerc
                                     )
    return resultsFrame


def getFocusedDf(data,
                 upperPerc,
                 lowerPerc):
    fnDf = addUpperAndLowerPercColumns(data=data,
                                       upperPerc=upperPerc,
                                       lowerPerc=lowerPerc
                                       )

    fnDf['upperInclude'] = np.where(fnDf['percCumulative'] <= fnDf['upperPerc'], 1, 0)
    fnDf['lowerInclude'] = np.where(fnDf['percTotal'] >= fnDf['lowerPerc'], 1, 0)
    fnDf = fnDf[(fnDf['upperInclude'] == 1) & (fnDf['lowerInclude'] == 1)]

    return fnDf


def plotColumnAnalysis(df, xColName,
                       percTotalColName,
                       percCumulativeColName,
                       analysisColName,
                       summaryColName,
                       upperPerc,
                       lowerPerc,
                       upperPercColName='upperPerc',
                       lowerPercColName='lowerPerc'):
    sns.set(rc={'figure.figsize': (20, 8)})
    fnDf = addUpperAndLowerPercColumns(data=df,
                                       upperPerc=upperPerc,
                                       lowerPerc=lowerPerc
                                       )

    fnDf2 = fnDf[[xColName, percCumulativeColName, percTotalColName, upperPercColName, lowerPercColName]]

    print(f'Lineplot showing {analysisColName} distribution')
    fig = sns.lineplot(x=xColName, y='value', hue='variable', data=pd.melt(fnDf2, [xColName]))
    fig.set(xlabel='Item rank', ylabel='Percent of all documents in corpus')
    fig.set_title(f'Lineplot summary for: {analysisColName}')
    fig.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()
    print("\n\n")

    print(f'Barplot showing {analysisColName} distribution')
    if len(df) < 100:
        plt.xticks(rotation='vertical')
        fig = sns.barplot(x=analysisColName, y=summaryColName, data=fnDf)
        fig.set(xlabel=f'Items in summary column: {analysisColName}', ylabel='Document Count')
        fig.set_title(f'Barplot summary for: {analysisColName}')
        plt.show()
        print("\n\n")
    else:
        print(f'--->Dataset too large for barchart visibility: {str(len(df))}')
        print('\n\n')


def plotSummaryFrame(dataFrame):
    plotOne = ['percTotalDocuments', 'percTotalCategories']
    plotTwo = ['totalDocuments', 'totalNulls', 'totalCategories']

    tDf = pd.melt(dataFrame, id_vars=['analysisStage'], value_vars=plotOne)
    sns.barplot(x='variable', y='value', hue='analysisStage', data=tDf)
    plt.show()

    print('\n\n')

    tDf = pd.melt(dataFrame, id_vars=['analysisStage'], value_vars=plotTwo)
    sns.barplot(x='variable', y='value', hue='analysisStage', data=tDf)
    plt.show()


def columnExplore(dataFrame,
                  analysisColName,
                  upperPerc=0.9,
                  lowerPerc=0.01,
                  summaryColName='docCount',
                  cumulativeColName='cumulativeCount',
                  colOrderName='order',
                  percTotalColName='percTotal',
                  percCumulativeColName='percCumulative'):
    resultsFrame = initSummaryFrame()

    print("Beginning analysis on 'Main' frame...")

    showColumnSummary(df=dataFrame,
                      analysisColName=analysisColName)

    fnDf = buildReportingFrame(df=dataFrame,
                               analysisColName=analysisColName,
                               summaryColName=summaryColName,
                               cumulativeColName=cumulativeColName,
                               colOrderName=colOrderName,
                               percTotalColName=percTotalColName,
                               percCumulativeColName=percCumulativeColName)

    resultsFrame = analyzeReportingFrame(df=fnDf,
                                         resultsFrame=resultsFrame,
                                         summaryColName=summaryColName,
                                         analysisColName=analysisColName)

    plotColumnAnalysis(df=fnDf,
                       xColName=colOrderName,
                       analysisColName=analysisColName,
                       summaryColName=summaryColName,
                       upperPerc=upperPerc,
                       lowerPerc=lowerPerc,
                       percTotalColName=percTotalColName,
                       percCumulativeColName=percCumulativeColName)

    print("Beginning analysis on 'Focused' frame......")
    focusDf = getFocusedDf(data=fnDf,
                           upperPerc=upperPerc,
                           lowerPerc=lowerPerc)

    resultsFrame = analyzeReportingFrame(df=focusDf,
                                         resultsFrame=resultsFrame,
                                         analysisName="Trimmed",
                                         upperPerc=0.9,
                                         lowerPerc=0.01,
                                         summaryColName=summaryColName,
                                         analysisColName=analysisColName)

    plotColumnAnalysis(df=focusDf,
                       xColName=colOrderName,
                       analysisColName=analysisColName,
                       summaryColName=summaryColName,
                       upperPerc=upperPerc,
                       lowerPerc=lowerPerc,
                       percTotalColName=percTotalColName,
                       percCumulativeColName=percCumulativeColName)

    display(resultsFrame)
    plotSummaryFrame(resultsFrame)

    return resultsFrame


def setPlotSize(plotsize):
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


def examineColumnNumeric(df,
                         colName,
                         binsize=1000,
                         verbose=False,
                         zoom=False,
                         minZoomLevel=0,
                         maxZoomLevel=0,
                         plotsize=1,
                         numRecords=10,
                         forceDecimal=False,
                         decimalPlaces=3):

    binColName = f'bin_at_{str(binsize)}'
    binnedCountName = 'binnedCount'

    # Parameter checking
    if (binsize <= 0):
        print(f'binsize of {str(binsize)} given. Must be > 0 and evenly divisible by 10 or = 1')
        return

    #Do I really care what the binsize is? May be limiting like this
    #if (binsize % 10 != 0) and (binsize != 1):
    #    print(f'binsize of {str(binsize)} given. Must be evenly divisible by 10 or = 1')
    #    return

    if zoom:
        if maxZoomLevel < minZoomLevel:
            print(f'maxZoomLevel given as {str(maxZoomLevel)} which must ' +
                  f'be >= minZoomLevel given as {str(minZoomLevel)}')
            return

        #Do I really need this?
        #if (maxZoomLevel % 10 != 0) or (minZoomLevel % 10 != 0):
        #    print(f'both maxZoomLevel given as {str(maxZoomLevel)} ' +
        #          f'and minZoomLevel given as {str(minZoomLevel)} ' +
        #          f'must be evenly divisible by 10')
        #    return

    #Set size of displayed plot
    if plotsize < 1 or plotsize > 5:
        print(f'plotsize given as {str(plotsize)} must be between 1 and 5')
        return
    else:
        setPlotSize(int(plotsize))

    # Make a copy of the incoming frame as we will be manipulating it
    tDf = df.copy()

    #Null values make it flunk out.
    numNullValues = tDf[colName].isnull().sum()
    if numNullValues > 0:
        print(f'Warning: {numNullValues} null values detected in column. Removing for analysis')
        tDf = tDf.dropna(subset=[colName], axis=0)

    #Groupby and summarize dataframe
    if forceDecimal:
        tDf[binColName] = [int(val / binsize) for val in tDf[colName]]
        tDf[binColName + '2'] = [int(val / binsize) * binsize for val in tDf[colName]]
    else: #Original version using only integers
        tDf = round(tDf[[colName]], 0).astype(int)
        tDf[binColName] = [int(math.trunc(val / binsize) * binsize) for val in tDf[colName]]

    tDf = tDf.groupby(binColName).size().to_frame(binnedCountName).sort_values([binColName], ascending=False)
    tDf.reset_index(inplace=True)

    #Zoom to applicable level
    if zoom:
        tDf = tDf.loc[(tDf[binColName] >= minZoomLevel) & (tDf[binColName] <= maxZoomLevel)]
        tDf.reset_index(drop=True, inplace=True)

    #Show me how it went
    plt.xticks(rotation='vertical')
    fig = sns.barplot(x=binColName, y=binnedCountName, data=tDf, palette="crest")
    fig.set_xlabel(f'Summary column: {colName} (binned at {str(binsize)})', fontsize=15)
    fig.xaxis.labelpad=20
    fig.set_ylabel('Document Count', fontsize=15)

    if zoom:
        titleTail = '\nZoom factor [{0}:{1}]'.format(minZoomLevel, maxZoomLevel)
    else:
        titleTail = ''
    fig.set_title(f'Data dispersion summary for: {colName} (binned at {str(binsize)}){titleTail}', fontsize=20)

    plt.show()

    if verbose:
        exploreDataframe(tDf,numRecords=numRecords)



