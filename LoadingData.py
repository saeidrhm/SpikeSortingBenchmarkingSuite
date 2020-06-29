import numpy as np

def LoadingSimDataRaw(TemplateNamePre,TemplateNamePost,NoCells,NoChannels):
    SpikeCellsList = []
    for CellNum in range(0,NoCells):
        SpikeCellsList.append(np.load(TemplateNamePre+str(CellNum)+TemplateNamePost))
    SpikeChannelDataList = []
    SpikeChannelLabelList = []
    for ChanNum in range(0,NoChannels):
        tempListData = []
        tempListLabel = []
        for cell in range(0,NoCells): 
            tempListData.append(SpikeCellsList[cell][:,ChanNum,:])
            tempListLabel.append(cell*np.ones((SpikeCellsList[cell][:,ChanNum,:]).shape[0]))
        SpikeChannelDataList.append(np.concatenate(tempListData))
        SpikeChannelLabelList.append(np.concatenate(tempListLabel))
    return SpikeChannelDataList,SpikeChannelLabelList

def LoadingSimDataOverlappingSuppressed(TemplateNamePre,TemplateNamePostOS,NoCells,NoChannels):
    SpikeCellsList = []
    for CellNum in range(0,NoCells):
        SpikeCellsList.append(np.load(TemplateNamePre+str(CellNum)+TemplateNamePostOS))
    SpikeChannelDataList = []
    SpikeChannelLabelList = []
    for ChanNum in range(0,NoChannels):
        tempListData = []
        tempListLabel = []
        for cell in range(0,NoCells): 
            tempListData.append(SpikeCellsList[cell][ChanNum][:,:])
            tempListLabel.append(cell*np.ones((SpikeCellsList[cell][ChanNum][:,:]).shape[0]))
        SpikeChannelDataList.append(np.concatenate(tempListData))
        SpikeChannelLabelList.append(np.concatenate(tempListLabel))
    return SpikeChannelDataList,SpikeChannelLabelList

import pickle
def SaveResult(filename,data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
