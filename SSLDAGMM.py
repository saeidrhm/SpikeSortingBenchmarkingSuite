import os
import scipy.io as sio

MatlabBinAdr = "/share/binary/Matlab16/bin/matlab"
Matlabglnxa64Adr = "/share/binary/Matlab16/bin/Matlab_R2016a_glnxa64.lic"

##GMMLDA
def ClLDAFeatureExtractionSingleChan(Data,Parametrs):
    sio.savemat('spikes.mat', {'spikes':Data}) ##write the data as .mat file
    file = open("LDAGMMdefvar.m","w")
    file.write("load('spikes.mat');")
    file.write("sortMethodInput = '"+str(Parametrs['LDACL_sortMethodInput'])+"';")
    file.write("numClusRangeInput = "+str(Parametrs['LDACL_numClusRangeInput'])+";")
    file.write("DimInput = "+str(Parametrs['LDACL_DimInput'])+";")
    file.write("maxIterInput = "+str(Parametrs['LDACL_maxIterInput'])+";")
    file.write("minSampleSizeInput = "+str(Parametrs['LDACL_minSampleSizeInput'])+";")
    file.write("SampIterInput = "+str(Parametrs['LDACL_SampIterInput'])+";")
    file.write("minClusSizeInput = "+str(Parametrs['LDACL_minClusSizeInput'])+";")
    file.write("earlyStopInput = "+str(Parametrs['LDACL_earlyStopInput'])+";")
    file.write("doPlotInput = "+str(Parametrs['LDACL_doPlotInput'])+";")
    file.write("extensionfilenameInput = '"+str(Parametrs['LDACL_extensionfilenameInput'])+"';")
    file.close()
    bashCommand = str(MatlabBinAdr) + " -c "+str(Matlabglnxa64Adr) +" -r "  + '"' +"run('LDAGMMdefvar.m'); run('ClLDACaller_v2.m'); exit();" + '"'

    ResFlag = os.system(bashCommand)
    ##Read results
    LDACLFeatures = sio.loadmat('ClLDA_lastY_'+str(Parametrs['LDACL_extensionfilenameInput'])+'.mat')
    return(LDACLFeatures['lastY'])

def ClLDAFeatureExtraction(Data,Parametrs):
    RetList = []
    for CurrChan in range(0,len(Data)):
        RetList.append(ClLDAFeatureExtractionSingleChan(Data[CurrChan][:,:],Parametrs))
    return(RetList)

