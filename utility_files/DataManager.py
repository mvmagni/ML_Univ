class DataManager:
    __version = 0.1
    
    def __init__(self,
                 xData,
                 yData):
        self.xData = xData
        self.yData = yData
    
    def summary(self):
        print(f'This is a placeholder for the DataManager summary')
        print(f'')