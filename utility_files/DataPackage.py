import DataPackageSupport as dps


class DataPackage:
    __version = 0.1

    def __init__(self,
                 origData,
                 uniqueColumn,
                 targetColumn
                 ):
        self.uniqueColumn = uniqueColumn
        self.targetColumn = targetColumn
        self.__setOrigData(origData)


    def __setOrigData(self, origData):
        self.origData = origData
        self.isOrigDataLoaded = True

        #Get and set features listing, removing unique and target columns
        self.dataFeatures = list(origData.columns)
        # Remove unique and target columnm
        self.dataFeatures.remove(self.uniqueColumn)
        self.dataFeatures.remove(self.targetColumn)

        # A new dataframe means we need to reset our work
        self.__resetWork()

    # if the data gets changed then we need to
    # invalidate all the results/work done previously
    def __resetWork(self):
        self.isBalanced = False

        self.__clearTrainTestData()

    def __setTrainData(self, trainData):
        self.trainData = trainData
        self.isTrainDataLoaded = True

    def getTrainData(self):
        return self.trainData

    def __setTestData(self, testData):
        self.testData = testData
        self.isTestDataLoaded = True

    def getTestData(self):
        return self.testData

    def __clearOrigData(self):
        self.origData = None
        self.isOrigDataLoaded = False

    def __clearTrainTestData(self):
        self.isTrainTestSplit = False
        self.isTrainDataLoaded = False
        self.trainData = None

        self.isTestDataLoaded = False
        self.testData = None

    def splitTrainTest(self,
                       stratifyColumn=None,
                       train_size=0.8,
                       random_state=765,
                       shuffle=True
                       ):

        if stratifyColumn is None:
            stratifyColumn = self.targetColumn

        train, test = dps.trainTestSplit(dataFrame=self.getOrigData(),
                                         train_size=train_size,
                                         random_state=random_state,
                                         stratifyColumn=stratifyColumn,
                                         shuffle=shuffle)

        self.__setTrainData(train)
        self.__setTestData(test)
        self.isTrainTestSplit = True
        self.__clearOrigData()

    def getOrigData(self):
        if self.isOrigDataLoaded == False:
            display("Original data frame is not loaded")
        return self.origData

    def display(self):
        emptySpace = '    '
        indent = emptySpace + '---> '

        print(f'{emptySpace}DataPackage summary:')
        print(f'{emptySpace}Attributes:')
        print(f'{indent}uniqueColumn: {self.uniqueColumn}')
        print(f'{indent}targetColumn: {self.targetColumn}')

        print(f'{emptySpace}Process:')
        print(f'{indent}isBalanced: {self.isBalanced}')
        print(f'{indent}isTrainTestSplit: {self.isTrainTestSplit}')

        print(f'{emptySpace}Data:')
        print(f'{indent}isOrigDataLoaded: {self.isOrigDataLoaded}')
        print(f'{indent}isTrainDataLoaded: {self.isTrainDataLoaded}')
        print(f'{indent}isTestDataLoaded: {self.isTrainDataLoaded}')

    def displayClassBalance(self, columnName=None, verbose=False, showRecords=5):
        if columnName is None:
            columnName = self.targetColumn

        dps.displayClassBalance(data=self.getOrigData(),
                                columnName=columnName,
                                showRecords=showRecords,
                                verbose=verbose)

    def classBalanceUndersample(self,
                                sampleSize=None,
                                columnName=None):


        if columnName is None:
            columnName = self.targetColumn

        # Needs to be balanced
        dfBalanced = dps.classBalanceUndersample(dataFrame=self.getOrigData(),
                                                 columnName=columnName,
                                                 sampleSize=sampleSize)

        if not self.isBalanced:
            self.__setOrigData(dfBalanced)
            self.isBalanced = True

    def showClassBalance(self,
                         columnName=None):
        if columnName is None:
            columnName = self.targetColumn

        # Needs to be balanced
        dfBalanced = dps.classBalanceUndersample(dataFrame=self.getOrigData(),
                                                 columnName=columnName,
                                                 alreadyBalanced=True)


    def getXTrainData(self, 
                      finalFeatures=None):
        
        if finalFeatures is None:
            useFeatures = self.dataFeatures
        else:
            useFeatures = finalFeatures 
        
        
        return  self.getTrainData()[useFeatures]    
    
    
    def getXTestData(self,
                    finalFeatures=None):
        
        if finalFeatures is None:
            useFeatures = self.dataFeatures
        else:
            useFeatures = finalFeatures 
        
        return  self.getTestData()[useFeatures]    
    
    
    def getYTrainData(self):
        return self.getTrainData()[self.targetColumn]
    
    
    def getYTestData(self):
        return self.getTestData()[self.targetColumn]