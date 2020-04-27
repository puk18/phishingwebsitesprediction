"""
Spyder Editor
This is a temporary script file.

@author: gurvindersingh
"""
import pandas as pd
import numpy as np
from pprint import pprint
import scipy.stats as sps

class RandomForest():

    #Calculate the entropy of a dataset.
    def entropy(column):
      
        elements,counts = np.unique(column,return_counts = True)
        sum= np.sum(counts)
        entropy = 0
        for i in range(len(elements)):
            entropy=entropy+np.sum((-counts[i]/sum)*np.log2(counts[i]/sum))
        return entropy
    
    #Calculates Info gain of a dataset,
    def InfoGain(data,feature_name):
      
        total_entropy = RandomForest.entropy(data["Result"])
        vals,counts= np.unique(data[feature_name],return_counts=True)
        sum= np.sum(counts)
        particular_entropy=0
        for i in range(len(vals)):
            particular_entropy =particular_entropy+np.sum((counts[i]/sum)*RandomForest.entropy(data.where(data[feature_name]==vals[i]).dropna()["Result"]))    
        information_gain = total_entropy - particular_entropy
        return information_gain
        
    
    def ID3(data,originaldata,features,parent_node = None):
     
         if len(np.unique(data["Result"])) <= 1:
            value= np.unique(data["Result"])[0]
            return value
        
         elif len(data)==0:
            return np.unique(originaldata["Result"])[np.argmax(np.unique(originaldata["Result"],return_counts=True)[1])]
        
         elif len(features) ==0:
            return parent_node
          
         else:
            parent_node = np.unique(data["Result"])[np.argmax(np.unique(data["Result"],return_counts=True)[1])]
            features = np.random.choice(features,size=np.int(np.sqrt(len(features))),replace=False)
            item_values = [RandomForest.InfoGain(data,feature) for feature in features] 
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]
            tree = {best_feature:{}}
            features = [i for i in features if i != best_feature]
            for value in np.unique(data[best_feature]):
                divided_dataset = data.where(data[best_feature] == value).dropna()
                recursivetree = RandomForest.ID3(divided_dataset,dataset,features,parent_node)
                tree[best_feature][value] = recursivetree        
            return(tree)    
            
    def predict(dictonary,tree):
    
        for key in list(dictonary.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][dictonary[key]] 
                except:
                    return 'p'
                result = tree[key][dictonary[key]]
                if isinstance(result,dict):
                    return RandomForest.predict(dictonary,result)
                else:
                    return result
    
         
    def split(dataset):
        training_data = dataset.iloc[:7738].reset_index(drop=True)# 70% training set We drop the index respectively relabel the index
        #starting form 0, because we do not want to run into errors regarding the row labels / indexes
        testing_data = dataset.iloc[7738:].reset_index(drop=True) # 30% testing set
        return training_data,testing_data
    
    
    def RandomForest_Train(dataset,number_of_Trees):
      
        random_forest_sub_tree = []
        for i in range(number_of_Trees):
            bootstrap_sample = dataset.sample(frac=1,replace=True)
            
            bootstrap_training_data = RandomForest.split(bootstrap_sample)[0]
            bootstrap_testing_data = RandomForest.split(bootstrap_sample)[1] 
            
            random_forest_sub_tree.append(RandomForest.ID3(bootstrap_training_data,bootstrap_training_data,bootstrap_training_data.drop(labels=['Result'],axis=1).columns))
        return random_forest_sub_tree
    
    
             
    def RandomForest_Predict(query,random_forest):
        predictions = []
        for tree in random_forest:
            predictions.append(RandomForest.predict(query,tree))
        return sps.mode(predictions)[0][0]
    
    
    
    def RandomForest_Test(data,random_forest):
        data['predictions'] = None
        for i in range(len(data)):
            query = data.iloc[i,:].drop('Result').to_dict()
            data.loc[i,'predictions'] = RandomForest.RandomForest_Predict(query,random_forest)
        print('The prediction accuracy is: ',sum(data['predictions'] == data['Result'])/len(data)*100,'%')


    
dataset = pd.read_csv('updatedDataSet.csv',header=None)
dataset = dataset.sample(frac=1)
dataset.columns = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol',
                             'double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length',
                             'Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL',
                             'Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic',
                             'Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report','Result']
        
training_data = RandomForest.split(dataset)[0]
testing_data = RandomForest.split(dataset)[1]

random_forest = RandomForest.RandomForest_Train(dataset,50) 

query = testing_data.iloc[0,:].drop('Result').to_dict()
query_target = testing_data.iloc[0,0]
print('Result: ',query_target)
prediction = RandomForest.RandomForest_Predict(query,random_forest)
print('prediction: ',prediction)

       
        
RandomForest.RandomForest_Test(testing_data,random_forest)

    
