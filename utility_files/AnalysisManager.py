import pickle
import gzip
import os
import LimeSupport as lms
import ShapSupport as ss

class AnalysisManager:
    __version = 0.1
    
    def __init__(self,
                 filename,
                 data_manager,
                 model_manager):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.set_filename (filename)
    
    def summary(self):
        self.data_manager.summary()
        self.model_manager.summary()
    
    def set_filename(self, filename):
        self.filename = filename
    
    def get_filename(self):
        return self.filename
    
    def save(self,
             path='.'):
        if (path=='.'):
            print(f'Saving to directory {os.getcwd()}')
        
        full_filename = path + '/' + self.filename + '.gz'
        with open(full_filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


    def show_lime_global(self):
        lms.showLimeGlobalImportance(XTrain=self.data_manager.xData,
                                     YTrain=self.data_manager.yData)
    
    def show_shap_summary(self,
                          plot_type='bar'):
        for modelStore in self.model_manager.model_list:
            ss.show_shap_summary(modelStore=modelStore,
                                 xData=self.data_manager.xData,
                                 plot_type=plot_type)
    
    def calc_shap_values(self,
                         GPU=False,
                         override=False,
                         debug=False):
        num_missing_values = 0
        
        print (f'idx: shap_value: description')
        for count,mdl in enumerate(self.model_manager.model_list):
            idx = '{0: >3}'.format(count)
            #shap_val_loaded = '{0: <10}'.format(mdl.has_shap_values)
            print (f'{idx}: {str(mdl.has_shap_values).ljust(10)}: {mdl.description}')
            if not mdl.has_shap_values:
                num_missing_values += 1
        
        if num_missing_values > 0:
            print (f'There are {num_missing_values} shap_values missing.')
            print (f'')
        
            for modelStore in self.model_manager.model_list:
                # If no shap values then calculate
                # If override is True then calculate
                if modelStore.has_shap_values is False or override is True:
                    ss.calc_shap_value(modelStore=modelStore,
                                       xData=self.data_manager.xData,
                                       GPU=GPU,
                                       debug=debug)
    
    def show_shap_bar(self):
        for modelStore in self.model_manager.model_list:
            ss.show_shap_bar(modelStore=modelStore)
    
    def show_shap_beeswarm(self):
        for modelStore in self.model_manager.model_list:
            ss.show_shap_beeswarm(modelStore=modelStore)
    
    def show_shap_waterfall(self,
                           value_index=0):
        for modelStore in self.model_manager.model_list:
            ss.show_shap_waterfall(modelStore=modelStore,
                                   value_index=value_index)
                    
    
    