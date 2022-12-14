class ModelManager:
    __version = 0.1

    def __init__(self,
                 model,
                 description):
        
        self.model_list = []
        self.add_model(model, description)
    
    # Add model to list
    def add_model(self, model, description):
        self.model_list.append (ModelStore(model, description))
    
    
    # Displays a list of all stored models
    def list_models(self):
        print (f'idx: Description')
        for count,mdl in enumerate(self.model_list):
            idx = '{0: >3}'.format(count)
            print (f'{idx}: {mdl.description}')
        
    
    # Remove model from list using model index from "list_models"
    def remove_model(self, model_index):
        if (model_index < len(self.model_list)):
            del(self.model_list[model_index])
            print(f'Model with index {model_index} removed')
            print(f'New model list:')
            self.list_models()
        else:
            print(f'Model index does not exist.')
            print(f'Model list:')
            self.list_models()
    
    # Displays a short summary of class information
    def summary(self):
        # How many models listed?
        print(f'Managing {len(self.model_list)} models')
        print(f'')
       
        # Iterate models with model summaries
        for count,mdl in enumerate(self.model_list):
            idx = '{0: >2}'.format(count)
            print (f'Model index: {idx}')
            print (f'Description: {mdl.description}')
            print (f'shap_values calculated: {mdl.has_shap_values}')
            print (f'Model details:')
            print (mdl.model)
            print (f'')
        
        
class ModelStore:
    __version = 0.1
    
    def __init__(self,
                 model,
                 description):
    
        self.model = model
        self.description = description
        
        
        self.has_shap_values = False
        self.shap_values = None
    
    def set_shap_values(self,
                        shap_values):
        self.has_shap_values = True
        self.shap_values = shap_values
    
    def get_shap_values(self):
        return self.shap_values

