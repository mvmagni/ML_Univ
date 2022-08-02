
from yellowbrick.target import ClassBalance
import pandas as pd
from sklearn.model_selection import train_test_split

def trainTestSplit(dataFrame,

                   train_size=0.8,
                   random_state=765,
                   stratifyColumn=None,
                   shuffle=True):

    origDataSize = len(dataFrame)
    indent = '---> '
    if stratifyColumn is None:
        train, test = train_test_split(dataFrame,

                                       train_size=train_size,
                                       random_state=random_state,
                                       shuffle=shuffle
                                       )
    else:
        train, test = train_test_split(dataFrame,
                                       train_size=train_size,

                                       random_state=random_state,
                                       stratify=dataFrame[[stratifyColumn]],
                                       shuffle=shuffle
                                       )

    print(f'Completed train/test split (train_size = {train_size}):')
    print(f'{indent}Original data size: {origDataSize}')
    print(f'{indent}Training data size: {len(train)}')
    print(f'{indent}Testing data size: {len(test)}')
    if stratifyColumn is None:
        print(f'{indent}Not stratified on any column')
    else:
        print(f'{indent}Stratified on column: {stratifyColumn}')

    return train, test


def classBalanceUndersample(dataFrame,
                            columnName,
                            sampleSize=None,
                            alreadyBalanced=False):

    #Display the initial state
    tDf = dataFrame.copy()
    displayClassBalance(data=tDf,
                        columnName=columnName)

    if alreadyBalanced:
        print("Classes already balanced")
        return

    # Not balanced, need to get some info to get size to balance to
    ttlColName = 'ttlCol'

    #If no size specified then calculate based on smallest class
    if sampleSize is None:
        # Find the sample size by finding which group/class is smallest
        tDfSize = tDf.groupby([columnName]).size().to_frame(ttlColName).sort_values(by=ttlColName)
        tDfSize.reset_index(inplace=True)
        sample_size = pd.to_numeric(tDfSize[ttlColName][0])
        sample_class = tDfSize[columnName][0]
        print(f'Undersampling data to match min class: {str(sample_class)} of size: {sample_size}')
    else:
    #Sample size given so use that to balance
        sample_size=sampleSize

    # Do the sampling
    tDf = tDf.groupby(columnName, group_keys=False).apply(lambda x: x.sample(sample_size))
    tDf.reset_index(drop=True, inplace=True)

    displayClassBalance(data=tDf,
                        columnName=columnName,
                        verbose=True)

    # Return the balance dataset
    return tDf


def displayClassBalance(data,
                        columnName,
                        verbose=False,
                        showRecords=5):
    ttlColName = 'ttlCol'

    visualizer = ClassBalance()
    visualizer.fit(data[columnName])  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure

    if verbose:
        tDfSize = data.groupby([columnName]).size().to_frame(ttlColName).sort_values(by=ttlColName).copy()
        tDfSize.reset_index(inplace=True)
        display(tDfSize.head(showRecords))
