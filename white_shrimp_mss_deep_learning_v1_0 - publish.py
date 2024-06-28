"""
white shrimp - MSS deep learning
v1.0c 2024-06-27  
Open Source @ https://github.com/LBK888/MSS
License: Apache 2.0

蝦蝦多光譜深度學習 自動合併數據 
直接讀取google drive裡的excel檔  

會取得資料夾的所有檔案，對檔案進行資料合併，所以個檔案的sheet name必須一致  

匯入訓練資料格式 data label:  
[input data]*n欄, 空白1欄 , [labels]*m欄   
例如:
光譜1,光譜2,光譜3, ,冷凍h,冷藏h,冷凍次數,解凍次數  

其他設定請在第一段設定區設定:  
root_path = '' 主資料夾


MAPE: Computes the mean absolute percentage error between y_true & y_pred.
loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)

"""
import json
import os
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers,callbacks

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#**** 設定區  ****#
#### ANN structure ####
ANN_Epoch=1200
Train_repeats=8   #整個重複幾次? 用來取平均值

# ANN Model structures
ANN_upLayers=[56,128,64,48,24,8]    #node numbers, input/output nodes are not included
ANN_upDrops=[0,0,0,0,0,0]           #Dropout, 0=該層不使用，0.2~0.5=dropout %

ANN_lowLayers=[128,256,512,256,128,36,24]   #node numbers, input/output nodes are not included
ANN_lowDrops=[0,0.4,0.4,0.5,0.4,0,0]        #Dropout, 0=該層不使用，0.2~0.5=dropout %


#### Comparison axis ####
Compa_Axis=1  #資料組合方式
# 0是數據變多筆，例如200隻蝦跟150隻蝦數據合併，變成350筆訓練資料
# 1是input的項目變多，例如眼28個input與身體10個合併，輸入變成38個nodes

#### 資料夾、檔名名稱 ####
root_path = r'Z:\MSSvsFTcyc\Train_Fields'  #change dir "Shrimp" to your project folder

have_test_data=True       #Test data format has to be xlsx w/ correct sheet names. 
test_data_path=r'Z:\禎禎\MSSvsFTcyc\Test_Fields\LZ_SC.xlsx' #do not include answers

#sanity check
if len(ANN_upLayers)!=len(ANN_upDrops) or len(ANN_lowLayers)!=len(ANN_lowDrops):
  raise SystemExit('Dropout的層數設定與layer不一致')

#是否要進行正規化
to_normalize=True


### functions ###
class SaveBestModel(callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()


def check_processable_xls(path):
    '''
    Go through all folders, and return a list of path of processable videos
    '''
    xls_paths=[]

    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(('.xls','.xlsx')):
                    video_path = os.path.join(root, file)
                    xls_paths.append(video_path)

    elif os.path.isfile(path) and path.endswith(('.xls','.xlsx')):
        xls_paths.append(path)

    else:
        print("Error: Invalid input of file or folder path!")
        SystemExit("Error", "No folder selected. Exiting...")
        exit()

    return xls_paths



def Save_Trained_Fig(history,iX,iY):

    # list all data in history
    fig = plt.figure(figsize=(8, 16))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    print(history.history.keys())
    # summarize history for accuracy
    ax1.plot(history.history['mape'])
    ax1.plot(history.history['val_mape'])
    ax1.set_title('model mape')
    ax1.set(xlabel='epoch',ylabel='mape')
    ax1.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set(xlabel='epoch',ylabel='loss')
    ax2.legend(['train', 'test'], loc='upper left')

    #fig.show()
    fig.savefig(root_path+f'history_x{iX}_y{iY}.pdf', format='pdf')
    plt.close(fig)








##### 取得excel資料  #####
  #找到所有可以讀取的檔案 進行合併
xls_paths=check_processable_xls(root_path)


merged_dfs={}

for xls in xls_paths:
  df = pd.DataFrame()
  data_all = pd.read_excel(xls, sheet_name=None,index_col=None, header=None)
  sheet = pd.ExcelFile(xls)

  print(f'> in {xls}:')
  for s_name in sheet.sheet_names:
    Empty_Col=data_all.get(s_name).columns[data_all.get(s_name).isna().any()].tolist()[0]
    DataN,Total_Col=data_all.get(s_name).shape
    print('Find '+s_name+': data=('+str(Empty_Col)+','+str(DataN)+'); label=('+str(Total_Col-Empty_Col-1)+','+str(DataN)+')')

    #### merging data ####
    if s_name in merged_dfs:
      merged_dfs[s_name]=pd.concat([merged_dfs[s_name],data_all.get(s_name)], axis=0)
    else:
      #new sheet
      merged_dfs[s_name]=data_all.get(s_name)


#save merged file
with pd.ExcelWriter('merged.xlsx') as writer:
  for i, (sName, dataf)  in enumerate(merged_dfs.items()):
    dataf.to_excel(writer, sheet_name=sName, header=False, index=False)




##### read merged file #####
data_all = pd.read_excel('merged.xlsx', sheet_name=None,index_col=None, header=None)
sheet = pd.ExcelFile('merged.xlsx')
ScanN=len(sheet.sheet_names)



##### CHECK SHEET DATA #####
    #合併時需要檢查
prev_DataN=-1
prev_Total_Col=-1
for s_name in sheet.sheet_names:
    DataN,Total_Col=data_all.get(s_name).shape
    if prev_DataN!=DataN and prev_DataN!=-1 and Compa_Axis==1:
        raise SystemExit('水平合併，各sheet間的資料筆數需要相同。檢查sheet名稱包括大小寫是否相同')
    elif prev_Total_Col!=Total_Col and  prev_Total_Col!=-1 and Compa_Axis==0:
        raise SystemExit('垂直合併，各sheet間的欄位需要相同')
    
    prev_DataN=DataN
    prev_Total_Col=Total_Col

print(f'合併檔驗證PASS，{DataN}筆資料 與 {Total_Col}個欄位')


##### Prepare TEST DATA #####
if have_test_data:
    if '.xls' in test_data_path:
        test_data_all = pd.read_excel(test_data_path, sheet_name=None,index_col=None, header=None)
        test_data_sheet = pd.ExcelFile(test_data_path)
    elif '.csv' in test_data_path:
        have_test_data=False
        pass    #not supported so far    

##### SCAN LOOP #####
#prepare variables
means_min_loss=np.zeros((ScanN,ScanN))
stds_min_loss=np.zeros((ScanN,ScanN))
means_min_val_loss=np.zeros((ScanN,ScanN))
stds_min_val_loss=np.zeros((ScanN,ScanN))
means_min_mape=np.zeros((ScanN,ScanN))
stds_min_mape=np.zeros((ScanN,ScanN))
means_max_val_acc=np.zeros((ScanN,ScanN))
stds_max_val_acc=np.zeros((ScanN,ScanN))

means_avg_val_acc=np.zeros((ScanN,ScanN))
stds_avg_val_acc=np.zeros((ScanN,ScanN))

preds=[]
normal_para=[]

for y_idx in range(ScanN):
  for x_idx in range(ScanN):

    #### 1. merging data ####
    xEmpty_Col=data_all.get(sheet.sheet_names[x_idx]).columns[data_all.get(sheet.sheet_names[x_idx]).isna().any()].tolist()[0]

    if y_idx!=x_idx:
      yEmpty_Col=data_all.get(sheet.sheet_names[y_idx]).columns[data_all.get(sheet.sheet_names[y_idx]).isna().any()].tolist()[0]
      data_x=data_all.get(sheet.sheet_names[x_idx]).iloc[:,:xEmpty_Col].to_numpy()
      data_y=data_all.get(sheet.sheet_names[y_idx]).iloc[:,:yEmpty_Col].to_numpy()
      data=np.append(data_x,data_y, axis=Compa_Axis)

      #merge test data if needed  
      if have_test_data and sheet.sheet_names[x_idx] in test_data_sheet.sheet_names and sheet.sheet_names[y_idx] in test_data_sheet.sheet_names:
        test_data_x=test_data_all.get(sheet.sheet_names[x_idx]).to_numpy()
        test_data_y=test_data_all.get(sheet.sheet_names[y_idx]).to_numpy()
        test_data=np.append(test_data_x,test_data_y, axis=Compa_Axis)

          
      if Compa_Axis==0:
        #trainN merge to trainN (訓練筆數merge，label也要增加)
        label_x=data_all.get(sheet.sheet_names[x_idx]).iloc[:,xEmpty_Col+1:].to_numpy()
        label_y=data_all.get(sheet.sheet_names[y_idx]).iloc[:,yEmpty_Col+1:].to_numpy()
        label=np.append(label_x,label_y, axis=Compa_Axis)
      else:
        #橫向增加data,只需要再拿一次label
        label=data_all.get(sheet.sheet_names[x_idx]).iloc[:,xEmpty_Col+1:].to_numpy()
    else:
      Empty_Col=data_all.get(sheet.sheet_names[x_idx]).columns[data_all.get(sheet.sheet_names[x_idx]).isna().any()].tolist()[0]
      data=data_all.get(sheet.sheet_names[x_idx]).iloc[:,:xEmpty_Col].to_numpy()
      label=data_all.get(sheet.sheet_names[x_idx]).iloc[:,xEmpty_Col+1:].to_numpy()

      if have_test_data and sheet.sheet_names[x_idx] in test_data_sheet.sheet_names:
        test_data=test_data_all.get(sheet.sheet_names[x_idx]).to_numpy()

    #update Ns
    trainN,inputN=data.shape
    labelN,outputN=label.shape

    #sanity check
    if trainN!=labelN:
      raise SystemExit('輸入資料與Label資料 筆數不相同')




    #### 2. 正規化 ####
    mean_data=0
    std_data=1
    mean_label=0
    std_label=1

    if to_normalize:
        # 正規化，全體共用一範圍
        mean_data = data.mean() # 平均數
        std_data = data.std()   # 標準差
        # 個別範圍
        #mean_data = np.mean(data, axis=0)
        #std_data = np.std(data, axis=0)
        data -= mean_data
        data /= std_data

        #label正規化
        mean_label = label.mean() # 平均數
        std_label = label.std()   # 標準差
        #mean_label = np.mean(label, axis=0)
        #std_label = np.std(label, axis=0)
        label -= mean_label
        label /= std_label

        #test 正規化
        if have_test_data and sheet.sheet_names[x_idx] in test_data_sheet.sheet_names and sheet.sheet_names[y_idx] in test_data_sheet.sheet_names:
            test_data -= mean_data
            test_data /= std_data

    min_lossA=[]
    min_val_lossA=[]
    min_mapeA=[]
    max_val_accA=[]
    avg_val_accA=[]
    save_best_model = SaveBestModel()

    for train_loop_idx in range(Train_repeats):

        #### 3. data shuffle: ####
        random_seed=None
        np.random.seed(random_seed)
        shuffle_index = np.arange(len(data))
        np.random.shuffle(shuffle_index)
        data = data[shuffle_index]
        label = label[shuffle_index]

        data=data.astype(float)
        label=label.astype(float)


        #### 4. 資料預處理 ####
        # 取資料中的 85% 當作訓練集
        split_num = int(len(data)*0.85)
        train_data=data[:split_num]
        train_label=label[:split_num]


        #### 訓練集、驗證集、測試集的資料形狀 ####
        # 訓練集
        print(train_data.shape)
        print(train_label.shape)

        # 驗證集 (parameter scan 不做測試集)
        validation_data = data[split_num:]
        validation_label = label[split_num:]

        print(validation_data.shape)
        print(validation_label.shape)

        # 測試集
        #test_data = data[-5:]
        #print(test_data.shape)
        #test_label = label[-5:]


        #### Training ####
        # 建立神經網路架構
        model = Sequential()

        if y_idx>x_idx: #heat map下層的ANN架構
          model.add(layers.Dense(ANN_lowLayers[0],
                               kernel_initializer = 'random_normal',
                               activation = 'relu', # 增加一個密集層, 使用ReLU激活函數, 輸入層有1個輸入特徵
                              input_shape=(inputN,)))
          for layer_i in range(1,len(ANN_lowLayers)):
            model.add(layers.Dense(ANN_lowLayers[layer_i],activation = 'relu'))
            if ANN_lowDrops[layer_i]>0 and ANN_lowDrops[layer_i]<1:
              model.add(layers.Dropout(ANN_lowDrops[layer_i]))
        else: #heat map 上層以及對角線用的ANN架構
          model.add(layers.Dense(ANN_upLayers[0],
                               kernel_initializer = 'random_normal',
                               activation = 'relu', # 增加一個密集層, 使用ReLU激活函數, 輸入層有1個輸入特徵
                              input_shape=(inputN,)))
          for layer_i in range(1,len(ANN_upLayers)):
            model.add(layers.Dense(ANN_upLayers[layer_i],activation = 'relu'))
            if ANN_upDrops[layer_i]>0 and ANN_upDrops[layer_i]<1:
              model.add(layers.Dropout(ANN_upDrops[layer_i]))

        model.add(layers.Dense(outputN))  #輸出層固定
        model.summary()    # 顯示模型資訊

        # 編譯及訓練模型
        model.compile(optimizer='adam',loss='mse',metrics=['mape','mae'])  # or 'acc'
        
        history = model.fit(train_data,train_label,            # 訓練集
                                    validation_data=(validation_data,validation_label),  # 驗證集
                                    epochs=ANN_Epoch,verbose=0,callbacks=[save_best_model])   # 訓練週期
        model.set_weights(save_best_model.best_weights)
        
        min_lossA.append(min(history.history['loss']))
        min_val_lossA.append(min(history.history['val_loss']))
        min_mapeA.append(min(history.history['mape']))
        max_val_accA.append(min(history.history['val_mape']))

        avg_val_accA.append(np.array(history.history['val_mape'][:-15]).mean())

    #### END of Training ####
    Save_Trained_Fig(history,x_idx,y_idx)
    model.save(root_path+f'model_x{x_idx}_y{y_idx}.keras')
    # save normalization param
    normal_para.append([ x_idx,y_idx,sheet.sheet_names[x_idx],sheet.sheet_names[y_idx],mean_data,std_data,mean_label,std_label])
    #columns=['x', 'y','x_name','y_name','mean_data','std_data','mean_label','std_label']

    #TEST MODEL by custom imput file
    if have_test_data and sheet.sheet_names[x_idx] in test_data_sheet.sheet_names and sheet.sheet_names[y_idx] in test_data_sheet.sheet_names:
        pred=model.predict(test_data)       
        pred *= std_label
        pred += mean_label
        print(pred)
        preds.append(pred)

    #### Cal mean & std results ####
    min_lossA=np.array(min_lossA)
    min_val_lossA=np.array(min_val_lossA)
    min_mapeA=np.array(min_mapeA)
    max_val_accA=np.array(max_val_accA)
    avg_val_accA=np.array(avg_val_accA)

    means_min_loss[x_idx,y_idx]=min_lossA.mean()
    stds_min_loss[x_idx,y_idx]=min_lossA.std()
    means_min_val_loss[x_idx,y_idx]=min_val_lossA.mean()
    stds_min_val_loss[x_idx,y_idx]=min_val_lossA.std()
    means_min_mape[x_idx,y_idx]=min_mapeA.mean()
    stds_min_mape[x_idx,y_idx]=min_mapeA.std()
    means_max_val_acc[x_idx,y_idx]=max_val_accA.mean()
    stds_max_val_acc[x_idx,y_idx]=max_val_accA.std()
    
    means_avg_val_acc[x_idx,y_idx]=avg_val_accA.mean()
    stds_avg_val_acc[x_idx,y_idx]=avg_val_accA.std()

    if y_idx>x_idx:
      lowLayerN=len(model.layers)
      lowParaN=model.count_params()
    else:
      upLayerN=len(model.layers)
      upParaN=model.count_params()


##### END SCAN LOOP #####




##### 繪圖 ##### 
if ScanN==1:
  # 1 x 1
  lowLayerN=upLayerN
  lowParaN=upParaN

fig = plt.figure(figsize=(ScanN*3, ScanN*6))
ax1 = fig.add_subplot(521)
ax2 = fig.add_subplot(522)
ax3 = fig.add_subplot(523)
ax4 = fig.add_subplot(524)
ax5 = fig.add_subplot(525)
ax6 = fig.add_subplot(526)
ax7 = fig.add_subplot(527)
ax8 = fig.add_subplot(528)
ax9 = fig.add_subplot(529)
ax10 = fig.add_subplot(5,2,10)

fig.suptitle('Deep Learning Parameter Scan\nEpoch='+
             str(ANN_Epoch)+', repeat='+str(Train_repeats)+'\n'+
             'upper: '+str(upLayerN)+' layers & '+str(upParaN)+' parameters\n'+
             'lower: '+str(lowLayerN)+' layers & '+str(lowParaN)+' parameters')


data_titles=sheet.sheet_names
df1 = pd.DataFrame(means_min_loss, columns=data_titles,index=data_titles)
sns.heatmap(df1, annot=True, fmt=".3f", linewidths=.5, ax=ax1, cmap='RdBu')
ax1.set(xlabel="", ylabel="")
ax1.xaxis.tick_top()
ax1.set_title('Mean of minimal loss')

df2 = pd.DataFrame(stds_min_loss, columns=data_titles,index=data_titles)
sns.heatmap(df2, annot=True, fmt=".3f", linewidths=.5, ax=ax2, cmap='RdBu')
ax2.set(xlabel="", ylabel="")
ax2.xaxis.tick_top()
ax2.set_title('SD. of minimal loss')

df3 = pd.DataFrame(means_min_val_loss, columns=data_titles,index=data_titles)
sns.heatmap(df3, annot=True, fmt=".3f", linewidths=.5, ax=ax3, cmap='RdBu')
ax3.set(xlabel="", ylabel="")
ax3.xaxis.tick_top()
ax3.set_title('Mean of minimal val_loss')

df4 = pd.DataFrame(stds_min_val_loss, columns=data_titles,index=data_titles)
sns.heatmap(df4, annot=True, fmt=".3f", linewidths=.5, ax=ax4, cmap='RdBu')
ax4.set(xlabel="", ylabel="")
ax4.xaxis.tick_top()
ax4.set_title('SD. of minimal val_loss')

df5 = pd.DataFrame(means_min_mape, columns=data_titles,index=data_titles)
sns.heatmap(df5, annot=True, fmt=".2f", linewidths=.5, ax=ax5, cmap='RdBu')  # cmap='seismic', higher = red
ax5.set(xlabel="", ylabel="")
ax5.xaxis.tick_top()
ax5.set_title('Mean of minimal mape')

df6 = pd.DataFrame(stds_min_mape, columns=data_titles,index=data_titles)
sns.heatmap(df6, annot=True, fmt=".2f", linewidths=.5, ax=ax6, cmap='RdBu')
ax6.set(xlabel="", ylabel="")
ax6.xaxis.tick_top()
ax6.set_title('SD of minimal mape')

df7 = pd.DataFrame(means_max_val_acc, columns=data_titles,index=data_titles)
sns.heatmap(df7, annot=True, fmt=".2f", linewidths=.5, ax=ax7, cmap='RdBu')
ax7.set(xlabel="", ylabel="")
ax7.xaxis.tick_top()
ax7.set_title('Mean of minimal val_mape')

df8 = pd.DataFrame(stds_max_val_acc, columns=data_titles,index=data_titles)
sns.heatmap(df8, annot=True, fmt=".2f", linewidths=.5, ax=ax8, cmap='RdBu')
ax8.set(xlabel="", ylabel="")
ax8.xaxis.tick_top()
ax8.set_title('SD of minimal val_mape')


df9 = pd.DataFrame(means_avg_val_acc, columns=data_titles,index=data_titles)
sns.heatmap(df9, annot=True, fmt=".2f", linewidths=.5, ax=ax9, cmap='RdBu')
ax9.set(xlabel="", ylabel="")
ax9.xaxis.tick_top()
ax9.set_title('Mean of avg(-10) val_mape')

df10 = pd.DataFrame(stds_avg_val_acc, columns=data_titles,index=data_titles)
sns.heatmap(df10, annot=True, fmt=".2f", linewidths=.5, ax=ax10, cmap='RdBu')
ax10.set(xlabel="", ylabel="")
ax10.xaxis.tick_top()
ax10.set_title('SD of avg(-10) val_mape')

sheetName="merged"

fig.show()
fig.savefig(root_path+sheetName+'.pdf', format='pdf')
plt.close(fig)

with pd.ExcelWriter(root_path+sheetName+'_output.xlsx') as writer:
  df1.to_excel(writer, sheet_name='means_min_loss')
  df2.to_excel(writer, sheet_name='stds_min_loss')
  df3.to_excel(writer, sheet_name='means_min_val_loss')
  df4.to_excel(writer, sheet_name='stds_min_val_loss')
  df5.to_excel(writer, sheet_name='means_max_mape')
  df6.to_excel(writer, sheet_name='stds_max_mape')
  df7.to_excel(writer, sheet_name='means_max_val_mape')
  df8.to_excel(writer, sheet_name='stds_max_val_mape')
  df9.to_excel(writer, sheet_name='means_avg_val_mape')
  df10.to_excel(writer, sheet_name='stds_avg_val_mape')


##### 儲存各項參數結果 #####
normal_para_df = pd.DataFrame(normal_para,columns=['x', 'y','x_name','y_name','mean_data','std_data','mean_label','std_label'])
normal_para_df.to_excel(root_path+sheetName+'_normal_para.xlsx', index=False)


##### 儲存測試集結果 #####
if have_test_data:
    with pd.ExcelWriter(root_path+sheetName+'_preds.xlsx') as writer:
        for i,pred in enumerate(preds):
            df_pred = pd.DataFrame(pred)
            df_pred.to_excel(writer, sheet_name=f'pred_{i}')

    
