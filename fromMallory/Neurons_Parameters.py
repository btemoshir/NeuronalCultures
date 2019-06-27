# Parameters for Adaptative Exponential Integrate and Fire neurons
# simulated with NEST 


#Intrinsically Bursting with a<0
#ISI_inburst noise INdependant ; IBI noise dependant 
#postburst sAHP + linear increase and exp increase before burst of Vm 
di_IB = {
    'w':-50.,
    'V_m': -60.,
    'E_L': -70.0,
    'I_e': 38.8,
    'g_L': 9.,
    'C_m': 400.0,
    'a': -6.5,       
    'V_th': -50.0,
    'V_reset': -47.8,  
    'Delta_T': 2.,
    'b': 10.,
    'tau_w': 400.0,	    
    'V_peak': 0.0,
    'tau_syn_ex' : 0.2,
    't_ref' : 2.,
}

# Regular Spiking neurons without adaptation at low freq ; adaptation will be effective at high freq
# Low variability with noise : ISI noise INdependant

di_RS = {
    'w': 37.,
    'V_m': -51.,
    'E_L': -70.0,
    'I_e': 262.,
    'g_L': 12.01,
    'C_m': 200.0,
    'a': 2.,       
    'V_th': -50.0,
    'V_reset': -48.,  
    'Delta_T': 1.8,
    'b': 70.0,
    'tau_w': 300.0,
    'V_peak': 30.0,
    'tau_syn_ex' : 0.2,
       't_ref' : 2.,
}

# Input dependant / Noise driven neurons
# Note that at cst small freq (4 Hz), the weight impact a lot the hist of isi
# However, at low weight, the freq does not impact the width but only the mean
# => weight impact more the dynamic (more diverse)

#Input dependant / Noise driven neurons
di_ND = {
       'I_e' : 25.,
       'E_L': -64.2,
       'V_m': -64.2,
       'g_L': 10.,
       'C_m': 250.,
       'a': -1.5,
       'V_th': -55.,
       'V_reset': -59.,
       'Delta_T': 5.5,
       'b': 50.,
       'tau_w': 500.,
       'V_peak': 20.,
       'tau_syn_ex' : 0.2,
       't_ref' : 2.,
}

