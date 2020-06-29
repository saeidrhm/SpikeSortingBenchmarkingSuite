RecordingDuration = 10 
NumberofExcitNeurons = 3
NumberofInhibNeurons = 6
SamplingRate = 32000
SpikeTrainSeed = 123
TemplateSeed = 1000
RecordingsSeed = 200
GlobalSeed = 0
ProbName = 'tetrode-mea-l'
NumofSpikeRangeinSec = (5,100)
Nobs = 50
BetweenSameCellFiring = 2##ref_per = 2m
#NExc= 2 # number of excitatory cells
#NInh= 1 # number of inhibitory cells
FExc= 20 # average firing rate of excitatory cells in Hz
FInh= 20 # average firing rate of inhibitory cells in Hz
StExc= 1 # firing rate standard deviation of excitatory cells in Hz
StInh= 1 # firing rate standard deviation of inhibitory cells in Hz
NoiseLevel= 20 # noise standard deviation in uV
NoiseMode= 'uncorrelated' # [uncorrelated | distance-correlated | far-neurons]
BetweenSameChanFiringGap = 10 ## 
overlap_window_size_coef = 5.0 ##overlapping window size in ms
DuplicateSuppression = False ## trying to select distinct cell templates
LimSelTemp = {
      "L5_BP_bAC217_1": 1,##inh
      "L5_BTC_bAC217_1": 1,##inh
      "L5_ChC_cACint209_1": 1,##inh
      "L5_DBC_bAC217_1": 0,##inh
      "L5_LBC_bAC217_1": 1,##inh
      "L5_MC_bAC217_1": 0,##inh
      "L5_NBC_bAC217_1": 2,##inh
      "L5_NGC_bNAC219_1": 0,##inh
      "L5_SBC_bNAC219_1": 0,##inh
      "L5_STPC_cADpyr232_1": 1,##exc
      "L5_TTPC1_cADpyr232_1": 0,##exc
      "L5_TTPC2_cADpyr232_1": 1,##exc
      "L5_UTPC_cADpyr232_1": 1##exc
    }## number of each templates should be selected
CellTempSelByName = False ## enable selecting templates by name

seed_value= GlobalSeed
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)


import MEArec as mr
import yaml
from pprint import pprint

def SimulationTemplates():
    default_info, mearec_home = mr.get_default_config()
    # define cell_models folder
    cell_folder = default_info['cell_models_folder']
    template_params = mr.get_default_templates_params()
    template_params['target_spikes'] = NumofSpikeRangeinSec
    template_params['probe'] = ProbName
    template_params['n'] = Nobs
    # the templates are not saved, but the intracellular simulations are saved in 'templates_folder'
    tempgen = mr.gen_templates(cell_models_folder=cell_folder, params=template_params)
    mr.save_template_generator(tempgen, filename='test_templates.h5')


##Should be in parallel
def SpikeOverlapSuppression(recgen):
    NoCell = len(recgen.spiketrains)
    NoChan = recgen.spiketrains[0].waveforms.shape[1]
    RecgenSpiketrains = [ [[] for _ in range(NoChan) ] for _ in range(NoCell)]
    RecgenSpiketrainsDelIdx = [ [] for _ in range(NoCell)]
    for currcell in range(0,NoCell):
        RecgenSpiketrainsDelIdx[currcell] = np.where(recgen.spiketrains[currcell].annotations['overlap']!='NO')
    ##copy waveforms to RecgenSpiketrains
    for currcell in range(0,NoCell):
        for currchannel in range(0,NoChan):
            RecgenSpiketrains[currcell][currchannel] = recgen.spiketrains[currcell].waveforms[:,currchannel,:]
    for currcell in range(0,NoCell):
        for currchannel in range(0,NoChan):
            if len(RecgenSpiketrainsDelIdx[currcell])>0 :
                RecgenSpiketrains[currcell][currchannel] = np.delete(RecgenSpiketrains[currcell][currchannel],np.unique(np.concatenate(RecgenSpiketrainsDelIdx[currcell])),axis=0)
    return(RecgenSpiketrains,RecgenSpiketrainsDelIdx)

def SimulationRecording():
    recordings_params = mr.get_default_recordings_params()
    recordings_params['spiketrains']['duration'] = RecordingDuration
    recordings_params['spiketrains']['n_exc'] = NumberofExcitNeurons
    recordings_params['spiketrains']['n_inh'] = NumberofInhibNeurons
    recordings_params['spiketrains']['f_exc'] = FExc
    recordings_params['spiketrains']['f_inh'] = FInh
    recordings_params['spiketrains']['st_exc'] = StExc
    recordings_params['spiketrains']['st_inh'] = StInh
    recordings_params['spiketrains']['ref_per'] = BetweenSameCellFiring
    recordings_params['recordings']['seed'] = RecordingsSeed
    recordings_params['spiketrains']['seed'] = SpikeTrainSeed
    recordings_params['templates']['seed'] = TemplateSeed
    recordings_params['recordings']['noise_level'] = NoiseLevel
    recordings_params['recordings']['extract_waveforms'] = True
    recordings_params['recordings']['overlap'] = True
    recordings_params['recordings']['overlap_window_size_coef'] = overlap_window_size_coef
    recordings_params['recordings']['DuplicateSuppression'] = DuplicateSuppression
    recordings_params['recordings']['LimSelTemp'] = LimSelTemp
    recordings_params['recordings']['CellTempSelByName'] = CellTempSelByName
    recordings_params['templates']['overlap_threshold'] = 0.0 
    recgen = mr.gen_recordings(templates='test_templates.h5', params=recordings_params)
    for currcell in range(0,(NumberofInhibNeurons+NumberofExcitNeurons)):
        outfile = "spike_trains_cell_"+ str(currcell)+"_NoInh_"+str(NumberofInhibNeurons)+"_NoExt_"+str(NumberofExcitNeurons)+"_noise_"+str(NoiseLevel)+"_SS_"+str(SpikeTrainSeed)+"_TS_"+str(TemplateSeed)+"_RS_"+str(RecordingsSeed)+"_GS_"+str(GlobalSeed)+"_Dur_"+str(RecordingDuration)+"_cell_based_raw"+".npy"
        np.save(outfile,recgen.spiketrains[currcell].waveforms)
    RecgenSpiketrains,RecgenSpiketrainsDelIdx = SpikeOverlapSuppression(recgen)
    for currcell in range(0,(NumberofInhibNeurons+NumberofExcitNeurons)):
        outfile = "spike_trains_cell_"+ str(currcell)+"_NoInh_"+str(NumberofInhibNeurons)+"_NoExt_"+str(NumberofExcitNeurons)+"_noise_"+str(NoiseLevel)+"_SS_"+str(SpikeTrainSeed)+"_TS_"+str(TemplateSeed)+"_RS_"+str(RecordingsSeed)+"_GS_"+str(GlobalSeed)+"_Dur_"+str(RecordingDuration)+"_cell_based_overlapping_suppressed"+".npy"
        np.save(outfile,RecgenSpiketrains[currcell])
    outfile = outfile = "spike_trains_cell"+"_NoInh_"+str(NumberofInhibNeurons)+"_NoExt_"+str(NumberofExcitNeurons)+"_noise_"+str(NoiseLevel)+"_SS_"+str(SpikeTrainSeed)+"_TS_"+str(TemplateSeed)+"_RS_"+str(RecordingsSeed)+"_GS_"+str(GlobalSeed)+"_Dur_"+str(RecordingDuration)+"_cell_based_overlapping_IDX_list"+".npy"
    np.save(outfile,RecgenSpiketrainsDelIdx)
    

SimulationTemplates()
for noise in [5]:
    NoiseLevel = noise
    SimulationRecording()

