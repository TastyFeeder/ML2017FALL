
import sys
from  collections import OrderedDict
import numpy as np
import random
#import matplotlib.pyplot as plt
import math

HR = 9

def main():
    if len(sys.argv) < 2 :
        print("[ERVROR]argv wrong")
        exit(1)

#path from argv
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

# get raw Train data and raw test date
    raw_Train_Data,raw_Train_List = read_train_data(train_path)
#    print(raw_Train_Data["2014/1/1"]["O3"])
    raw_Test_Data,raw_Test_List = read_test_data(test_path)

#scaling train data
#    the_range,scaling_data = feature_scaling(raw_Train_Data,raw_Train_List)
#    print(scaling_data["2014/1/1"]["O3"],the_range)
#    exit(1)
#    print(the_range)
#    exit(1)
    the_range = []

# enocde test to scaling data
#    scaling_test_data = test_encode(raw_Test_Data, the_range)

# get feature list and rmove some of them
    FeatureList = raw_Train_List[:]
#    print (FeatureList)
#    exit(1)
    #blacklist = ["AMB_TEMP","CH4","WIND_DIREC","WIND_SPEED","WS_HR","RAINFALL","RH","THC","WD_HR","CO"]
#['AMB_TEMP','CH4','C0','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
    #blacklist = ['AMB_TEMP','CH4','C0','NMHC','NO','NO2','NOx','O3','PM10','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']

    blacklist = ['AMB_TEMP','CH4','NMHC','NO','NO2','RH','THC']
    for i in blacklist:
        FeatureList.remove(i)
    #FeatureList = ['PM2.5']
#    print(FeatureList)
    #exit(1)
#transform train and test data to np array
    Train_Data,Train_Ans,New_list = get_train_data(raw_Train_Data, FeatureList)
    #Train_Data,Train_Ans,New_list = get_train_data(scaling_data, FeatureList)
#    print(Train_Data.shape)
    Test_Data = get_test_data(raw_Test_Data, New_list)
    #Test_Data = get_test_data(scaling_test_data, FeatureList)
    #Test_Data = get_test_data(scaling_test_data, New_list)
#    print(Test_Data)
#    exit(1)
# the para of gradient descent
    Lamda = 0.
    LR = 4e-10
    MAX_it = 100000
    
# gradient descent
    #w,b = gradient_descent(Train_Data, Train_Ans, Lamda,LR, MAX_it,New_list, the_range)
#    print(Test_Data)

# write model
    #write_model2csv(w,b)
# write csv
    #write_test_csv(w, b, Test_Data, the_range, output_path)
    
#    n_w,n_b = read_model("7.0modol.csv")
#    write_test_csv(n_w, n_b, Test_Data, the_range, output_path)
    n_w,n_b = read_model("last.csv")
    write_test_csv(n_w, n_b, Test_Data, the_range, output_path)
    
    
    return



def feature_scaling(data,FeatureList):
    # {Feature:(mean:sigma)}
    feature_range = {}
    scaling_data = data.copy()
    for f in FeatureList:
        feature_range.update({f:[0,0]})
    for key,val in data.items():
        for f,l in val.items():
            for ele in l:
                feature_range[f][0] += ele
    for f in FeatureList:
        feature_range[f][0] /=5652#scaling_data.shape[0]
    for key,val in scaling_data.items():
        for f,l in val.items():
            for ele in l:
                feature_range[f][1]+= (ele-feature_range[f][0])**2
    for f in FeatureList:
        feature_range[f][1] /=5652#scaling_data.shape[0]
        feature_range[f][1] = math.sqrt(feature_range[f][1])
    for key,val in scaling_data.items():
        for f,l in val.items():
            for ele in range(len(l)):
                scaling_data[key][f][ele] = (scaling_data[key][f][ele] - feature_range[f][0]) / feature_range[f][1]
    return feature_range,scaling_data
            
def test_encode(data, scaling_range):
    # {Feature:(max:min)}
    scaling_data = data.copy()
    for key,val in scaling_data.items():
        for f,l in val.items():
            for ele in range(len(l)):
                scaling_data[key][f][ele] = (scaling_data[key][f][ele] - scaling_range[f][0]) / scaling_range[f][1] 
    return scaling_data

    
def write_test_csv(w,b,test_data,scaling_range, path):
    Result = w * test_data +b
    Result = Result.sum(axis=1)
    fout = open(path,"w")
    fout.write("id,value")
    for e in range(len(Result)):
        #fout.write("\nid_%d,%f"%(e,val_decode(scaling_range, Result[e])))
        fout.write("\nid_%d,%f"%(e,Result[e]))


    
def read_model(path):
    fin = open(path,'r')
    data = fin.read()
    line = data.split('\n')
    b = np.array(float(line[0])) 
    w = np.array([float(x.strip('\'')) for x in line[1][:-1].split(',')])
    print(b,w)
    return w,b

def write_model2csv(w,b):
    fout = open("model.csv","w")
    for i in b.tolist():
        fout.write(str(i)+"\n")
    for i in w.tolist():
        fout.write(str(i)+",")

def val_decode(scaling_range, data):
    return data * scaling_range["PM2.5"][1] + scaling_range["PM2.5"][0]
    

def gradient_descent(Train_Data,Train_Ans,Lamda,LR,max_it,FeatureList, scaling_range):
    w = np.random.uniform(-0.01, 0.01, len(FeatureList) * HR)
    b = np.random.uniform(-0.01, 0.01, 1)
#    print(w, '\n', w.shape, '\n', b, '\n', b.shape)
#    exit(1)
    N = float(Train_Data.shape[0])
    # print(np.min(w))
    old_error = float("inf")
    loss_count = 0
    for i in range(max_it):
        predict = np.dot(Train_Data, w) + b
        # predict(print.shape)
        error = Train_Ans - predict
        # print(error, error.shape)
        #loss_t = val_decode(scaling_range, np.sum(error))
        #loss =   np.sum(loss_t**2)/N +  Lamda * np.sum(w**2)
        loss =   np.sum(error**2)/N +  Lamda * np.sum(w**2)
        
        # print('loss', loss)
        RMSE = np.sqrt(np.sum(error**2) / N)
        #RMSE = np.sqrt(np.sum(loss_t**2) / N)

        # print('RMSE', RMSE)
        # print(error, '\n', error ** 2, '\n', error.shape)

        # calculate w_grad, b_grad
        w_grad = -1 * np.dot(error, Train_Data) + Lamda * w
        # print(w_grad.shape)
        b_grad = -1 * np.sum(error)

        # update w, b
        w -= LR * w_grad
        b -= LR * b_grad
        # print(w, '\n', b)
        the_error = np.sum(error)
        if abs(old_error) > abs(the_error):
            if (abs(old_error - the_error)/ abs(old_error)) < 0.01:
                loss_count+=1
        else :
            LR/=2
        if loss_count >10:
            LR*=1.1
            loss_count = 0
        
        print("it",i,"loss==>",loss,"RMSE====>",RMSE,"LR ===>",LR)
    return w, b
    

def get_test_data(mydata, FeatureList ):
    test_data = []
    for i in range(240):
        day1 = mydata["id_%d"%(i)]
        one = []
        for t in FeatureList:
            if t in day1:
                one+=day1[t][9-HR:]
            else:
                parse = t
                parse = parse.split('-')
                #print(t,parse)
                a = np.array(day1[parse[0]])
                b = np.array(day1[parse[1]])
                c = a*b
                one += c.tolist()
                #print(t,parse,a,b,c)
        test_data.append(one)
    return np.array(test_data)

def add_two_mul_feature(mixlist, one, day1, off, day2, new_list):
    new_feature = str(mixlist[0])+"-"+str(mixlist[1])
    if new_feature not in new_list:
        new_list.append(new_feature)
    if day2 == None:
        a = np.array(day1[mixlist[0]][off:off+HR])
        b = np.array(day1[mixlist[1]][off:off+HR])
        c = a*b
        one+=c.tolist()
    else:
        tmp = day1[mixlist[0]][(24-HR)+off:]
        a = np.array(tmp + day2[mixlist[0]][:off])
        tmp = day1[mixlist[1]][(24-HR)+off:]
        b = np.array(tmp + day2[mixlist[1]][:off])
        c = a*b
        one+=c.tolist()
    
# 9 hr feature and i ans as one traindata
def get_train_data(mydata, FeatureList ):
    train_data = []
    train_ans = []
    new_list = FeatureList.copy()
    for m in range(12):
        for d in range(20):
            day1 = mydata["2014/%d/%d"%(m+1,d+1)]
            for off in range(24-HR):
                one = []
                for t in FeatureList:
                    one+=day1[t][off:off+HR]
# add some special feature here (bad soluation)
#                add_two_mul_feature(["PM2.5","O3"],one,day1,off,None,new_list) 
#                add_two_mul_feature(["PM2.5","PM10"],one,day1,off,None,new_list) 
#                add_two_mul_feature(["PM10","O3"],one,day1,off,None,new_list) 
                train_data.append(one)
                train_ans.append(day1["PM2.5"][off+HR])
            if d != 19:
                day2 = mydata["2014/%d/%d"%(m+1,d+2)]
                for off in range(HR):
                    one = []
                    for t in FeatureList:
                        tmp = day1[t][(24-HR)+off:]
                        tmp = tmp + day2[t][:off]
                        one+=tmp
# add some special feature here (bad soluation)
#                    add_two_mul_feature(["PM2.5","O3"],one,day1,off,day2,new_list)
#                    add_two_mul_feature(["PM2.5","PM10"],one,day1,off,day2,new_list)
#                    add_two_mul_feature(["PM10","O3"],one,day1,off,day2,new_list)
                    train_data.append(one)
                    train_ans.append(day2["PM2.5"][off])
    return np.array(train_data),np.array(train_ans),new_list

def read_test_data(file_path):
    fin = open(file_path,'r', encoding = 'big5')
    data = OrderedDict()
    count = 0
    item = []
    while True:
        count +=1
        line = fin.readline().strip('\n')
        if not line : break
        line = line.split(',')
        line = [w.replace('NR','0.0') for w in line]
        if line[0] not in data:
            data.update({line[0]:{}})
            data[line[0]].update({line[1]:[float(x) for x in line[2:]]})
        else:
            if line[1] not in data[line[0]]:
                data[line[0]].update({line[1]:[float(x) for x in line[2:]]})
        if line[1] not in item:
            item.append(line[1])
    fin.close()
    return data,item


def read_train_data(file_path):
    fin = open(file_path,'r', encoding = 'big5')
    first = fin.readline()
# save as orderdict key is date val is dict of item val list
    data = OrderedDict()
    count = 0
    item = []
    while True:
        count +=1
        line = fin.readline().strip('\n')
        if not line : break
        line = line.split(',')
        line = [w.replace('NR','0') for w in line]
        if line[0] not in data:
            data.update({line[0]:{}})
            data[line[0]].update({line[2]:[float(x) for x in line[3:]]})
        else:
            if line[2] not in data[line[0]]:
                data[line[0]].update({line[2]:[float(x) for x in line[3:]]})
        if line[2] not in item:
            item.append(line[2])
    fin.close()
    return data,item

if __name__ == "__main__":
    #read_model("model.csv")
    main()
