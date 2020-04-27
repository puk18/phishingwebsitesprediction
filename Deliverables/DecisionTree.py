"""
Spyder Editor
This is a temporary script file.
@author: ayusharya
"""
import pandas as pd
import numpy as np
from pprint import pprint



class DecisionTree():
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
      
        total_entropy = DecisionTree.entropy(data["Result"])
        vals,counts= np.unique(data[feature_name],return_counts=True)
        sum= np.sum(counts)
        particular_entropy=0
        for i in range(len(vals)):
            particular_entropy =particular_entropy+np.sum((counts[i]/sum)*DecisionTree.entropy(data.where(data[feature_name]==vals[i]).dropna()["Result"]))    
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
            features_values = [DecisionTree.InfoGain(data,feature) for feature in features] #select best feature
            best_feature = features[np.argmax(features_values)]
            tree = {best_feature:{}}#create tree with starting from best feature
            features = [i for i in features if i != best_feature] #remove best feature from feature array
            #Grow tree
            for value in np.unique(data[best_feature]):
                #value = value
                divided_dataset = data.where(data[best_feature] == value).dropna()
                recursivetree = DecisionTree.ID3(divided_dataset,dataset,features,parent_node)
                tree[best_feature][value] = recursivetree  #add that node to tree   
            return(tree)    
            
    #query is form of dictonary        
    def predict(dictonary,tree):
    
        for key in list(dictonary.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][dictonary[key]] 
                except:
                    return 1
                result = tree[key][dictonary[key]]
                if isinstance(result,dict):
                    return DecisionTree.predict(dictonary,result)
                else:
                    return result
        
    def split(dataset):
        training_data = dataset.iloc[:7738].reset_index(drop=True)# 70% training set We drop the index respectively relabel the index
        #starting form 0, because we do not want to run into errors regarding the row labels / indexes
        testing_data = dataset.iloc[7738:].reset_index(drop=True) # 30% testing set
        return training_data,testing_data
    
    
    
    def test(data,tree):
    
        #convert it to a dictionary by removing target feature records
        dictonary = data.iloc[:,:-1].to_dict(orient = "records")
        predicted = pd.DataFrame(columns=["predicted"]) 
        
        #Calculate the prediction accuracy
        for i in range(len(data)):
            predicted.loc[i,"predicted"] = DecisionTree.predict(dictonary[i],tree) 
        value= (np.sum(predicted["predicted"] == data["Result"])/len(data))    
        print('The prediction accuracy is: ',value*100,'%')
        
    
    
#Import the dataset and define the feature as well as the target datasets / columns#
filename = 'updatedDataSet.csv'
dataset = pd.read_csv(filename,
                      names=['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report','Result',])
    

    
training_data = DecisionTree.split(dataset)[0]
testing_data = DecisionTree.split(dataset)[1]   
    
    
tree = DecisionTree.ID3(training_data,training_data,training_data.columns[:-1])
DecisionTree.test(testing_data,tree)
