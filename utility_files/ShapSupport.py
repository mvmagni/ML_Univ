import shap
import matplotlib.pyplot as plt

# SHAP summary plot
def show_shap_summary(modelStore,
                      xData,
                      plot_type='bar'):
    
    if not modelStore.has_shap_values:
        print(f'No shap values calcuated for {modelStore.description}')
        return
        
    plt.title(f'Model: {modelStore.description}')
    shap.summary_plot(modelStore.shap_values,
                      xData,
                      plot_type=plot_type)  
    plt.show()
    plt.clf()

def show_shap_bar(modelStore):
    if not modelStore.has_shap_values:
        print(f'No shap values calcuated for {modelStore.description}')
        return
        
    plt.title(f'Model: {modelStore.description}')
    shap.plots.bar(modelStore.shap_values)  
    plt.show()
    plt.clf()

def show_shap_beeswarm(modelStore):
    if not modelStore.has_shap_values:
        print(f'No shap values calcuated for {modelStore.description}')
        return
        
    plt.title(f'SHAP beeswarm: {modelStore.description}')
    shap.plots.beeswarm(modelStore.shap_values)  
    plt.show()
    plt.clf()

def show_shap_waterfall(modelStore,
                        value_index=0):
    if not modelStore.has_shap_values:
        print(f'No shap values calcuated for {modelStore.description}')
        return
        
    plt.title(f'SHAP Waterfall [index:{value_index}]: {modelStore.description}')
    shap.plots.waterfall(modelStore.shap_values[value_index])  
    plt.show()
    plt.clf()

def calc_shap_value(modelStore,
                    xData,
                    GPU=False,
                    debug=False):
    print (f'Calculating shap_values for {modelStore.description}')
    if GPU:
        #explainer = shap.explainers.GPUTree(modelStore.model, xData)
        #shap_values = explainer(xData)
        #explainer = shap.Explainer(modelStore.model, xData)
        #shap_values = explainer(xData)
        print(f'STOP: Do not use GPU=True yet')
    else:
        if debug:
            print(f'DEBUG: non-gpu path')
        explainer = shap.Explainer(modelStore.model)
        shap_values = explainer(xData)
    
    modelStore.set_shap_values(shap_values=shap_values)
    if debug:
        print(f'DEBUG: shap_value type: {type(shap_values)}')
        print(f'DEBUG: explainer type: {type(explainer)}')
        print(f'DEBUG: modelStore.model:')
        print(modelStore.model)
        
