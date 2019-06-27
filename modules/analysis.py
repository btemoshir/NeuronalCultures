#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import nngt
import nest
import nngt.simulation as ns


def getBursts(net, spikes, senders, num_neurons, vms_time, vms, sim_time, interburst_time=200, plot_overlay=False, plot_bursts=False, plot_rasters=False):
    
    """
    Function which outputs the time and id of spikes of each burst
    
    Inputs:
    - net: the nngt network, used for extractions of the positions for plotting
    - spikes:  The spike times of teh neurons from the whole recording
    - senders: The id of the spikes at the corresponding spike times. Should be of the same length as that of spikes
    - num_neurons: The number of neurons
    - vms_time: The time points of the membrane recordings
    - vms: The values of the membrane recordings
    - sim_time: The total simulation time
    - interburst_time: The time between two bursts (change this parameter to get better bursts)
    - plot_overlay: If to overlay bursts on membrane potential recordings
    - plot_bursts: If to plot the bursts with a spatial time dependent colour coding scheme
    - plot_rasters: To plot the rasters of each of the bursts separately
    
    Outputs:
    - burst_times   = list of spike times at each burst, each element of the list is an array of the spike times.
    - burst_neurons = list of ids of the spiking neurons, each element of the list is an array of the spike ids.
    
    """
    
    N = num_neurons
    neurons_pot = np.zeros((N,sim_time-1))
    neurons_pot_time = vms_time[::N]
    
    for i in range(N):
        neurons_pot[i] = vms[i::N]
    
    neurons_pot_mean = np.mean(neurons_pot,axis=0)
    y = np.arange(len(neurons_pot_time))
    interp = np.interp(y,xp = neurons_pot_time, fp = -neurons_pot_mean)
    peaks = sc.signal.find_peaks(interp,distance=interburst_time)
    burst_centres = y[peaks[0]]
    
    burst_times   = []
    burst_neurons = []
    
    for i in range(len(burst_centres)):
        if i > 0:
            indices = np.where(( burst_centres[i] - interburst_time/2 < spikes) & (spikes < burst_centres[i] + interburst_time/2))
        else:
            indices = np.where(( 0 < spikes) & (spikes < burst_centres[i] + interburst_time/2))
        
        burst_times.append(spikes[indices])
        burst_neurons.append(senders[indices])
    
    if plot_overlay:
        edge_start = np.zeros(len(burst_times))
        edge_end    = np.zeros(len(burst_times))
        for i in range(len(burst_times)):
            edge_start[i] = min(burst_times[i])
            edge_end[i]   = max(burst_times[i])
        
        plt.figure(figsize=(10,5))
        plt.plot(vms_time,vms,'.')
        #plt.plot(neurons_pot_time,neurons_pot_mean,'.')
        plt.scatter(burst_centres,np.zeros(len(burst_centres))+max(vms),marker='.',c='black')
        plt.bar(edge_start, np.zeros(len(burst_centres))+min(vms), align='edge', width=(edge_end-edge_start), alpha=0.5, color='green')
        
        plt.show()
        
    if plot_bursts:
        num_subplots = np.ceil(len(burst_times)**0.5)
        plt.figure(figsize=(30,25))

        for i in range(len(burst_times)):
            unique_neurons,unique_indices = np.unique(burst_neurons[i],return_index=True)
            plt.subplot(num_subplots,num_subplots,i+1)
            plt.scatter((net.get_positions(unique_neurons-1))[:,0],(net.get_positions(unique_neurons-1)[:,1]), c = burst_times[i][unique_indices], cmap='hot',marker='o')
            plt.colorbar()
            
        plt.show()
        
    if plot_rasters:
        num_subplots = np.ceil(len(burst_times)**0.5)
        plt.figure(figsize=(20,20))
        
        for i in range(len(burst_times)):
            plt.subplot(num_subplots,num_subplots,i+1)
            plt.plot(burst_times[i],burst_neurons[i],'.')
        
        plt.show()
        
    return burst_times,burst_neurons

def getBursts_phase(net, phase_time, phase, spikes, senders, vms_time, vms, min_fire = 1., threshold = 0.8, threshold_low = False, plot_overlay=False, plot_bursts=False, plot_rasters=False, plot_overlay_raster=False):
    
    """
    Function which computes the bursts from the mean phase of the neurons. A burst is where the mean phase is gretater than a threshold value.
    
    Inputs:
    - net: the nngt network, used for extractions of the positions for plotting
    - spikes:  The spike times of teh neurons from the whole recording
    - senders: The id of the spikes at the corresponding spike times. Should be of the same length as that of spikes
    - vms_time: The time points of the membrane recordings
    - vms: The values of the membrane recordings
    - max_delay : the maximal synaptic delay in the network, we use this to check the length of these burst, one burst can only be separated by 2 times this delay 
    - threshold: the threshold an integer between 0. to 1. which determines how many neurons as a fraction of total neurons are reqd for a burst.    
    - plot_overlay: If to overlay bursts on membrane potential recordings
    - plot_bursts: If to plot the bursts with a spatial time dependent colour coding scheme
    - plot_rasters: To plot the rasters of each of the bursts separately
    - plot_overlay_rasters: To plot the overlay on spike rasters
    
    Outputs:
    - burst_times   = list of spike times at each burst, each element of the list is an array of the spike times.
    - burst_neurons = list of ids of the spiking neurons, each element of the list is an array of the spike ids.
    
    TODO: Implement Lower Threshold properly. has problems.
    
    """
    len_phase     = np.shape(phase)[1]
    burst_times   = []
    burst_neurons = []
    mean_phase    = np.mean(phase,axis=0)    
    N             = net.node_nb()    

    
    if threshold_low:
        idx1   = np.argwhere((np.diff(np.sign(mean_phase - 2*np.pi*threshold)))>0).flatten()
        idx2   = np.argwhere((np.diff(np.sign(mean_phase - 2*np.pi*threshold_low)))<0).flatten()
        slope1 = (np.diff(mean_phase,n=2))[idx1]
        slope2 = (np.diff(mean_phase,n=2))[idx2]
        
        idx1 = idx1[slope1>0]
        idx2 = idx2[slope2<0]
        
        if len(idx1)>1:
            if len(idx2)>0:
                if idx2[0]<idx1[0]:
                    np.delete(idx2,0)
                    np.delete(slope2,0)
                
        if len(idx1)>1:
            if len(idx2)>0:
                if idx2[-1]<idx1[-1]:
                    np.delete(idx1,-1)
                    np.delete(slope1,-1)
        
        #if len(idx1)>len(idx2):
        #    for i in range(len(idx2)):
        #        if phase_time[idx2[i]]<phase_time[idx1[i]]:
        #            idx1 = np.delete(idx1,i)
        #else:
        #    for i in range(len(idx1)):
        #        if phase_time[idx2[i]]<phase_time[idx1[i]]:
        #            idx2 = np.delete(idx2,i)
        
        #print(len(idx1),len(idx2))
        
        plt.figure(figsize=(20,5))
        plt.plot(time_phase,np.mean(phase,axis=0),'.')
        plt.vlines(phase_time[idx1],0,2*np.pi,colors='red')
        plt.vlines(phase_time[idx2],0,2*np.pi,colors='green')
            
        for i in range(0,len(idx1)):
            x = (spikes > phase_time[idx1[i]]) & (spikes < phase_time[idx2[i]])
            if len(np.argwhere(x)) > min_fire*N:
                burst_times.append(spikes[x])
                burst_neurons.append(senders[x])
        
        
    else:
        idx   = np.argwhere(np.diff(np.sign(mean_phase - 2*np.pi*threshold))).flatten()
        slope = (np.diff(mean_phase))[idx]
        
        if slope[0]<0:
            np.delete(idx,0)
            np.delete(slope,-1)

        if slope[-1]>0:
            np.delete(idx,-1)
            np.delete(slope,-1)

        for i in range(0,len(idx)-1,2):
            x = (spikes > phase_time[idx[i]]) & (spikes < phase_time[idx[i+1]])
            if len(np.argwhere(x)) > min_fire*N:
                burst_times.append(spikes[x])
                burst_neurons.append(senders[x])
        
    
    if burst_times:     
        if plot_overlay:
            edge_start = np.zeros(len(burst_times))
            edge_end    = np.zeros(len(burst_times))
            
            for i in range(len(burst_times)):
                edge_start[i] = min(burst_times[i])
                edge_end[i]   = max(burst_times[i])
            plt.figure(figsize=(10,5))
            plt.plot(vms_time,vms,'.')
            plt.bar(edge_start, np.zeros(len(burst_times))+min(vms), align='edge', width=(edge_end-edge_start), alpha=0.5, color='green')
            plt.show()
            
        if plot_overlay_raster:
            edge_start = np.zeros(len(burst_times))
            edge_end    = np.zeros(len(burst_times))
            
            for i in range(len(burst_times)):
                edge_start[i] = min(burst_times[i])
                edge_end[i]   = max(burst_times[i])
            plt.figure(figsize=(10,5))
            plt.plot(spikes,senders,'.')
            plt.bar(edge_start, np.zeros(len(burst_times)) + 1.2*N, align='edge', width=(edge_end-edge_start), alpha=0.5, color='green')
            plt.show()

        if plot_bursts:
            num_subplots = np.ceil(len(burst_times)**0.5)
            plt.figure(figsize=(30,25))

            for i in range(len(burst_times)):
                unique_neurons,unique_indices = np.unique(burst_neurons[i],return_index=True)
                plt.subplot(num_subplots,num_subplots,i+1)
                plt.scatter((net.get_positions(unique_neurons-1))[:,0],(net.get_positions(unique_neurons-1)[:,1]), c = burst_times[i][unique_indices], cmap='hot',marker='o')
                plt.colorbar()
            plt.show()

        if plot_rasters:
            num_subplots = np.ceil(len(burst_times)**0.5)
            plt.figure(figsize=(20,20))
            
            for i in range(len(burst_times)):
                plt.subplot(num_subplots,num_subplots,i+1)
                plt.plot(burst_times[i],burst_neurons[i],'.')
            plt.show()
    else:
        print('No Bursts Found')
        
    return burst_times,burst_neurons   


def phaseNeurons(vms,vms_time,num_neurons):
    
    '''
    Calculates the phase of the neurons as a function of time using the membrane potential.
    
    Input:
    - vms         = the vms reading provided by nest, collectively for all neurons
    - vms_time    = the vms time reacding provided by nest
    - num_neurons = the number of neurons in the network
    
    Outputs:
    - phase: the phase of the neurons as a function of time. Size = num_neurons*time
    
    Algorithm:
    
    Phase is set to 0 when the vms is min and to 2*pi when it is maximum. Then it is linearly interpolated with the potential value of the membrane.
    
    '''
    
    N = num_neurons
    sim_time = len(vms[::N])
    
    zero_phase       = np.zeros(num_neurons)
    twopi_phase      = np.zeros(num_neurons)
    neurons_pot      = np.zeros((N,sim_time))
    neurons_pot_time = np.zeros((N,sim_time))
    phase            = np.zeros((N,sim_time))
    
    for i in range(N):
        zero_phase[i]       = np.min(vms[i::N])
        twopi_phase[i]      = np.max(vms[i::N])
        neurons_pot[i]      = vms[i::N]
        neurons_pot_time[i] = vms_time[i::N]
        
        y        = np.arange(len(neurons_pot_time[i]))
        interp   = np.interp(y,xp = neurons_pot_time[i], fp = neurons_pot[i])
        z        = 0. + 2*np.pi*(interp - zero_phase[i])/(twopi_phase[i] - zero_phase[i])
        phase[i] = np.interp(neurons_pot_time[i], xp = y, fp = z )
         
    return phase


def phaseNeurons_firing(senders, spikes, num_neurons, sim_time, interval = 0.05):
    
    '''
    Calculates the phase of the neurons as a function of time using the spiking times of the neurons.
    
    Input:
    - senders     = the neuron index of the firing neurons output my NEST recorder 
    - spikes      = the spiking time of the neurons output my NEST recorder 
    - num_neurons = the number of neurons in the network
    - sim_time    = the total simulation time in ms
    - interval    = the time inetrval size at which we calculate the phase
    
    Outputs:
    - time : the time at which the phase is calculated
    - phase: the phase of the neurons as a function of time. Size = num_neurons*time
    
    Algorithm:
    
    Phase is set to 2pi when a neuron fires after which it is set to 0. The phase rises linearly in time from 0 to 2pi till the next firing event.
    
    '''
    
    
    N = num_neurons
    neuron_times = []
    time  = np.arange(0,sim_time,interval)
    phase = np.zeros([N,len(time)])
    
    if interval is None:
        x = (np.sort(abs(np.sort(spikes) - np.roll(np.sort(spikes),1))))
        interval = x[np.nonzero(x)][0]
    
    for i in range(N):
        neuron_times.append(spikes[np.where(senders == i+1)])
        phase_2pi  = np.zeros(len(neuron_times[i]),dtype=int)
        
        for j in range(len(neuron_times[i])):
            phase_2pi[j] = (np.abs(time - neuron_times[i][j])).argmin()
            if j == 0:
                phase[i,0:phase_2pi[j]] = 2*np.pi*(time[0:phase_2pi[j]] - time[0])/(time[phase_2pi[j]] - time[0])
            else:
                phase[i,phase_2pi[j-1]:phase_2pi[j]] = 2*np.pi*(time[phase_2pi[j-1]:phase_2pi[j]] - time[phase_2pi[j-1] + 1]) /(time[phase_2pi[j]] - time[phase_2pi[j-1]+1])  
        
        if (len(time) > phase_2pi[-1]+2):
            if time[-1] > time[phase_2pi[-1]+1]:
                phase[i,phase_2pi[-1]:] = 2*np.pi*(time[phase_2pi[-1]:] - time[phase_2pi[-1]+1])/(time[-1] - time[phase_2pi[-1]+1])
        
    return time,phase
    
    
def kuramotoOP(phase,num_neurons):
    
    '''
    Calculates the Kuramoto Order Paramter of the system of neurons:
    
    Input:
    
    - phase        = the phase for all the neurons. Size = num_neurons*time
    - num_neurons  = number of neurons  
    
    Output:
    - a = r
    - b = phi
    
    where r*exp(-i\phi) is the Kuramoto OP
    '''
    phase = np.zeros(np.shape(phase)) + 0. + phase*1j
    z = np.sum(np.exp(phase),axis=0)
    z = z/num_neurons
    a = np.abs(z)
    b = np.angle(z)
    
    return a,b
    
    
def getBursts_threshold(net, spikes, senders, vms_time, vms, max_delay = 30., threshold = 0.2, plot_overlay=False, plot_bursts=False, plot_rasters=False, plot_overlay_raster=False):
    
    """
    Function which computes the bursts from neuronal spiking data. Each burst is only called a burst if atleast one neuron fires more than once in a burst period.
    
    Inputs:
    - net: the nngt network, used for extractions of the positions for plotting
    - spikes:  The spike times of teh neurons from the whole recording
    - senders: The id of the spikes at the corresponding spike times. Should be of the same length as that of spikes
    - vms_time: The time points of the membrane recordings
    - vms: The values of the membrane recordings
    - max_delay : the maximal synaptic delay in the network, we use this to check the length of these burst, one burst can only be separated by 2 times this delay 
    - threshold: the threshold an integer between 0. to 1. which determines how many neurons as a fraction of total neurons are reqd for a burst.    
    - plot_overlay: If to overlay bursts on membrane potential recordings
    - plot_bursts: If to plot the bursts with a spatial time dependent colour coding scheme
    - plot_rasters: To plot the rasters of each of the bursts separately
    - plot_overlay_rasters: To plot the overlay on spike rasters
    
    Outputs:
    - burst_times   = list of spike times at each burst, each element of the list is an array of the spike times.
    - burst_neurons = list of ids of the spiking neurons, each element of the list is an array of the spike ids.
    
    
    TODO: ADD SUPPORT FOR UPPER THRESHOLD AND FOR LOWER THRESHOLD TO FINISH A BURST.
    
    """
    
    burst_times   = []
    burst_neurons = []
    spike_times   = []
    N       = net.node_nb()
    t       = []
    n       = []
    i       = 1
    senders = senders[np.argsort(spikes)]
    spikes  = np.sort(spikes)
    decider = False
    
    while i < len(spikes):
        
        while np.size(np.where((spikes[i]-0.1<spikes) & (spikes<(spikes[i] + 2*max_delay) ))) > N*threshold:
            t.append(spikes[i])
            n.append(senders[i])
            i += 1
            if len(t) > N:
                decider = True
        
        if decider:
            burst_times.append(np.asarray(t))
            burst_neurons.append(np.asarray(n))
            del t,n
            t = []
            n = []
        
        decider = False
        i += 1
        del t,n
        t = []
        n = []
        
    #if burst_times:
    #    del burst_neurons[0], burst_times[0]
    
    if burst_times:     
        if plot_overlay:
            edge_start = np.zeros(len(burst_times))
            edge_end    = np.zeros(len(burst_times))
            
            for i in range(len(burst_times)):
                edge_start[i] = min(burst_times[i])
                edge_end[i]   = max(burst_times[i])
            plt.figure(figsize=(10,5))
            plt.plot(vms_time,vms,'.')
            plt.bar(edge_start, np.zeros(len(burst_times))+min(vms), align='edge', width=(edge_end-edge_start), alpha=0.5, color='green')
            plt.show()
            
        if plot_overlay_raster:
            edge_start = np.zeros(len(burst_times))
            edge_end    = np.zeros(len(burst_times))
            
            for i in range(len(burst_times)):
                edge_start[i] = min(burst_times[i])
                edge_end[i]   = max(burst_times[i])
            plt.figure(figsize=(10,5))
            plt.plot(spikes,senders,'.')
            plt.bar(edge_start, np.zeros(len(burst_times)) + 1.2*N, align='edge', width=(edge_end-edge_start), alpha=0.5, color='green')
            plt.show()

        if plot_bursts:
            num_subplots = np.ceil(len(burst_times)**0.5)
            plt.figure(figsize=(30,25))

            for i in range(len(burst_times)):
                unique_neurons,unique_indices = np.unique(burst_neurons[i],return_index=True)
                plt.subplot(num_subplots,num_subplots,i+1)
                plt.scatter((net.get_positions(unique_neurons-1))[:,0],(net.get_positions(unique_neurons-1)[:,1]), c = burst_times[i][unique_indices], cmap='hot',marker='o')
                plt.colorbar()
            plt.show()

        if plot_rasters:
            num_subplots = np.ceil(len(burst_times)**0.5)
            plt.figure(figsize=(20,20))
            
            for i in range(len(burst_times)):
                plt.subplot(num_subplots,num_subplots,i+1)
                plt.plot(burst_times[i],burst_neurons[i],'.')
            plt.show()
    else:
        print('No Bursts Found')
        
    return burst_times,burst_neurons    


def removeTransients_spikes(spikes, senders, cut_time_begin=5000., cut_time_end=False, cut_begin=True, cut_end=False):
    
    """
    Function to remove transients at the beginning and/or the end but just for the spikes, not for the other properties.   
    
    Input:
    
    - spikes:  The spike times of teh neurons from the whole recording
    - senders: The id of the spikes at the corresponding spike times. Should be of the same length as that of spikes
    - cut_time_begin: The time till which the function cuts the recordings at the beginning.
    - cut_time_end: The time after which the function cuts the recordings at the end.
    - cut_begin: If to cut the beginning.
    - cut_end: If to cut the end.
    
    Output:
    
    - spikes: spikes after cut
    - senders: senders after cut
    
    Note: The output spikes by NEST are not usually ordered. This function does NOT order it either.
    """
    
    
    
    if cut_time_begin > max(spikes):
        raise Exception('Initial time to cut is greater than total simulation time.')
           
    if cut_begin:
        senders  = senders[spikes > cut_time_begin]
        spikes   = spikes[spikes > cut_time_begin] - cut_time_begin

    if cut_end:
        senders = senders[spikes < cut_time_end]
        spikes  = spikes[spikes < cut_time_end]

    return spikes,senders
    

def removeTransients_vms(vms_time, vms, w_adap=False, cut_time_begin=5000., cut_time_end=False, cut_begin=True, cut_end=False, wcut=False):
    
    """
    Function to remove transients at the beginning and/or the end of the vms voltage and the adaptation current.
    Function also optinally return the conditions after the initial cut for the adaptation current.
    
    Input:
    
    - vms_time:       The times of the recording at which VMS is recorded.
    - vms:            The vms from NEST recording of the potential for all neurons.
    - w_adap:         The adaptation current from NEST recording for all neurons.
    - cut_time_begin: The time till which the function cuts the recordings at the beginning.
    - cut_time_end:   The time after which the function cuts the recordings at the end.
    - cut_begin:      If to cut the beginning.
    - cut_end:        If to cut the end.
    - wcut :          If to cut and return the adaptation current, w.
    
    Output:
    
    - spikes:  spikes after cut
    - senders: senders after cut
    - w_adap:  adaptation current after cut
    
    Note: The output spikes by NEST are not usually ordered. This function does NOT order it either.
    
    """
    
    if cut_begin:
        vms      = vms[vms_time > cut_time_begin]
        vms_time = vms_time[vms_time > cut_time_begin] - cut_time_begin
        if wcut:
            w_adap = w_adap[vms_time > cut_time_begin]
            
    if cut_end:
        vms      = vms[vms_time < cut_time_end]
        vms_time = vms_time[vms_time < cut_time_end]
        if wcut:
            w_adap = w_adap[vms_time < cut_time_end]
    
    if wcut:
        return vms_time, vms, w_adap
    
    else:
        return vms_time, vms



def burst_properties(net, burst_times, burst_neurons, spikes, senders, sim_time, interval=0.1):
    
    """
    Inputs:
    
    - net:           the NNGT network
    - burst_times:   the burst times from a burst detection function
    - burst_neurons: the burst neurons from a burst detection function
    - spikes :       spikes from NEST
    - senders:       spiking neuron index from NEST
    - sim_time:      simulation time
    - interval:      the time interval used to calculate phase for the phaseNeurons_firing function    
    
    Outputs:
    
    - propertiesDS:  a class which contains the following burst_properties: IBI, burst_duration, spikes_per_burst, burster proportion, phase_time, phase, kOP_r, kOP_theta, burst_times, burst_neurons
    
    """
    
    N                 = net.node_nb()
    num_bursts        = len(burst_neurons)
    burst_duration    = np.zeros(num_bursts)
    spikes_per_burst  = np.zeros(num_bursts)
    burster_proportion= np.zeros(num_bursts)
    burst_velocity    = np.zeros(num_bursts)
    
    if num_bursts>0:
        IBI       = np.zeros(num_bursts-1)
    else:
        IBI       = np.zeros(num_bursts)
    
    for i in range(num_bursts):
        burst_duration[i]     = max(burst_times[i]) - min(burst_times[i])
        spikes_per_burst[i]   = float(len(burst_times[i]))
        burster_proportion[i] = float(len(np.unique(burst_neurons[i])))
        burst_velocity[i]     = spikes_per_burst[i]/burst_duration[i]
        
        if i<num_bursts-1:
            IBI[i] = min(burst_times[i]) - max(burst_times[i-1])
            
    time,phase_spike = phaseNeurons_firing(senders=senders, spikes=spikes, num_neurons=N, sim_time=sim_time,interval=interval)
    r,theta          = kuramotoOP(phase_spike,N)
    
    class propertiesDS:
        
        def __init__(self, bt, bn, IBI, burst_duration, spikes_per_burst, burster_proportion, r, theta, time, phase_spike):
            if bt:
                #self.IBI                     = IBI
                self.IBI_mean                = np.mean(IBI)

                #self.burst_duration          = burst_duration
                self.burst_duration_mean     = np.mean(burst_duration) 

                #self.spikes_per_burst        = spikes_per_burst
                self.spikes_per_burst_mean    = np.mean(spikes_per_burst)

                #self.burster_proportion      = burster_proportion
                self.burster_proportion_mean = np.mean(burster_proportion)

                self.phase_time         = time
                self.phase              = phase_spike
                self.phase_mean         = np.mean(phase_spike,axis=0)

                self.kOP_r              = r
                self.kOP_theta          = theta
                self.kOP_mean           = np.mean(r)

                self.burst_times        = bt
                self.burst_neurons      = bn
                
            else:
                #self.IBI                     = 0.
                self.IBI_mean                = 0.

                #self.burst_duration          = 0.
                self.burst_duration_mean     = 0.

                #self.spikes_per_burst        = 0.
                self.spikes_per_burst_mean   = 0.

                #self.burster_proportion      = 0.
                self.burster_proportion_mean = 0.

                #self.phase_time         = time
                #self.phase              = phase_spike
                self.phase_mean         = np.mean(phase_spike,axis=0)

                #self.kOP_r              = r
                #self.kOP_theta          = theta
                self.kOP_mean           = np.mean(r)

                #self.burst_times        = bt
                #self.burst_neurons      = bn
                
    
    return propertiesDS(burst_times,burst_neurons,IBI,burst_duration,spikes_per_burst,burster_proportion, r, theta, time, phase_spike)



def percolation_netw_statistics(net, method_removal='uniform', spacing=0.002, lspac=250, starting=0.8, params=False, \
                                plot_ssc=True, plot_gc=True, activity=False, plot_activity=False):
    
    """ 
    Function which removes the edges of the network progressively, calculates structural properties, and optionally activity while removing the edges.
    
    Inputs:
    - net:           The NNGT network
    - method_removal:The edge removal method - See percolation Edges function of percolation
    - spacing:       The fraction of edges to progressively remove
    - lspac:         The maximum number of times teh function will remove edges
    - starting:      Starting fraction for removing edges. This many edges are removed the first time
    - params:        The model parametrs to calculate distributions of initial V and w
    - plot_scc:      If to plot the number of strongly connected components
    - plot_gc:       If to plot the size of the giant component
    - activity:      If to calculate activity or not
    - plot_activity: If to plot the activity properties
    
    Outputs:
    - frac:   an array of the fraction of original edges that are removed
    - scc:    an array of number of strongly connected components at each frac
    - deg:    an array of mean in-degree of the network at each frac
    - ending: the number of iterations in removing the edges. Max value = lspac
    - gc:     an array of size of giant components at each frac
    - BP:     the burst properties class at each frac (only returned if activity is True)
    
    """
    
    
    
    from modules import percolation as per
    from modules import simulation as nsim
    import networkx as nx
    from modules import analysis
    import multiprocessing
    
    
    #Setting some parametrs manually:
    sim_time              = 25000.
    noise_rate            = 100.
    noise_weight_fraction = 0.5
    Vm_IC                 = ('gaussian',-50.,30.)
    w_IC                  = ('gaussian',params['w'],params['b'])
    cut_time_begin        = 15000.
    threshold             = 0.4
    min_fire              = 0.05
    interval              = 0.1
    
    num_omp    = multiprocessing.cpu_count()
    scc        = np.ones((lspac))
    frac       = np.zeros((lspac))
    deg        = np.zeros((lspac))
    gc         = np.zeros((lspac))
    num_connec = len((np.asarray(net.edges())))
    BP         = []
    ending     = lspac-1
    N          = net.node_nb()
    frac[0]    = starting
    
    if method_removal == 'burst-initiators':
        
        senders,spikes,vms,vms_time,w = nsim.Simulation(net, num_omp= num_omp, \
                                        sim_time = sim_time, noise_rate = noise_rate, noise_weight_fraction = noise_weight_fraction,\
                                       noise = 'Minis', save_spk = "data/spk", show_activity = False, return_activity = True,\
                                       animation = False, anim_name = "data/anim.mp4",Vm_IC=Vm_IC,w_IC=w_IC)
        
        spikes,senders = removeTransients_spikes(spikes, senders, cut_time_begin = cut_time_begin, cut_time_end=False, cut_begin=True, cut_end=False)
        
        vms_time, vms  = removeTransients_vms(vms_time, vms, w_adap=False, cut_time_begin = cut_time_begin, cut_time_end=False, cut_begin=True, cut_end=False, wcut=False)
        
        time_phase, phase = phaseNeurons_firing(senders, spikes, num_neurons=N, sim_time = sim_time -cut_time_begin, interval = interval)
        
        burst_times,burst_neurons = getBursts_phase(net, time_phase, phase, spikes, senders, vms_time, vms, min_fire = min_fire, threshold = threshold, plot_overlay=False, plot_bursts=False, plot_rasters=False, plot_overlay_raster=False)
        
        per.percolationEdges(net, num_remove=int(frac[0]*num_connec), method_removal=method_removal, burst_times=burst_times, burst_neurons=burst_neurons)
        
    else:
        per.percolationEdges(net, num_remove=int(frac[0]*num_connec), method_removal=method_removal)
    
    deg[0]  = np.mean(net.degree_list(deg_type='in'))
    scc[0]  = nngt.analysis.num_scc(net)
    frac[0] = 1. - np.float64(len((np.asarray(net.edges()))))/num_connec
    gc[0]   = (max(nx.strongly_connected_component_subgraphs(net.to_directed()), key=len)).number_of_nodes()
    
    if activity:
        senders,spikes,vms,vms_time,w = nsim.Simulation(net, num_omp= num_omp, \
                                        sim_time = sim_time, noise_rate = noise_rate, noise_weight_fraction = noise_weight_fraction,\
                                       noise = 'Minis', save_spk = "data/spk", show_activity = False, return_activity = True,\
                                       animation = False, anim_name = "data/anim.mp4",Vm_IC=Vm_IC,w_IC=w_IC)
        
        #V = np.zeros(N)
        #W = np.zeros(N)
        #for i in range(N):
            #V[i] = vms[::N][-1]
           # W[i] = w[::N][-1]
            
        
        #senders,spikes,vms,vms_time,_ = nsim.Simulation(net, num_omp= num_omp, \
        #                                sim_time = 25000., noise_rate = 100., noise_weight_fraction = 0.5,\
        #                               noise = 'Minis', save_spk = "data/spk", show_activity = False, return_activity = True,\
        #                               animation = False, anim_name = "data/anim.mp4", Vm_IC=V, w_IC=W)
        
        spikes,senders = removeTransients_spikes(spikes, senders, cut_time_begin=cut_time_begin, cut_time_end=False, cut_begin=True, cut_end=False)
        
        vms_time, vms  = removeTransients_vms(vms_time, vms, w_adap=False, cut_time_begin=cut_time_begin, cut_time_end=False, cut_begin=True, cut_end=False, wcut=False)
        
        time_phase, phase = phaseNeurons_firing(senders, spikes, num_neurons=N, sim_time = sim_time - cut_time_begin , interval = interval)
        
        burst_times,burst_neurons = getBursts_phase(net, time_phase, phase, spikes, senders, vms_time, vms,\
                                                             min_fire = min_fire, threshold = threshold, plot_overlay=False,\
                                                             plot_bursts=False, plot_rasters=False, plot_overlay_raster=False)
        
        
        BP.append(analysis.burst_properties(net, burst_times, burst_neurons, spikes, senders, sim_time = sim_time - cut_time_begin ,interval = interval))
        
        
    for i in range(1,lspac):

        if len(net.edges()) > 1:
            
            if method_removal == 'burst-initiators':
                per.percolationEdges(net,num_remove=int(spacing*num_connec),method_removal=method_removal, burst_times=burst_times, burst_neurons=burst_neurons)
                
            else:            
                per.percolationEdges(net,num_remove=int(spacing*num_connec),method_removal=method_removal)
            
            if len(net.edges()) > 1:
                frac[i] =  1. - np.float64(len((np.asarray(net.edges()))))/num_connec
            else:
                frac[i] = 1.
                
            deg[i] = np.mean(net.degree_list(deg_type='in'))
            scc[i] = nngt.analysis.num_scc(net)
            gc[i]  = (max(nx.connected_component_subgraphs(net.to_undirected()), key=len)).number_of_nodes()
        
            if activity:
                senders,spikes,vms,vms_time,_ = nsim.Simulation(net, num_omp= num_omp, \
                                                sim_time = sim_time, noise_rate = noise_rate, noise_weight_fraction = noise_weight_fraction,\
                                               noise = 'Minis', save_spk = "data/spk", show_activity = False, return_activity = True,\
                                               animation = False, anim_name = "data/anim.mp4", Vm_IC = Vm_IC ,w_IC=w_IC)
             
                spikes,senders = removeTransients_spikes(spikes, senders, cut_time_begin = cut_time_begin, cut_time_end=False, cut_begin=True, cut_end=False)
        
                vms_time, vms  = removeTransients_vms(vms_time, vms, w_adap=False, cut_time_begin = cut_time_begin, cut_time_end=False, cut_begin=True, cut_end=False, wcut=False)

                time_phase, phase = phaseNeurons_firing(senders, spikes, num_neurons=N, sim_time = sim_time-cut_time_begin , interval = interval)
        
                burst_times,burst_neurons = getBursts_phase(net, time_phase, phase, spikes, senders, vms_time, vms,\
                                                             min_fire = min_fire, threshold = threshold, plot_overlay=False,\
                                                             plot_bursts=False, plot_rasters=False, plot_overlay_raster=False)

                BP.append(analysis.burst_properties(net, burst_times, burst_neurons, spikes, senders, sim_time = sim_time - cut_time_begin, interval = interval))   
                
        else:
            ending = i
            break        
    
    
    if plot_ssc:
        if plot_gc:
            fig, ax = plt.subplots(2,1)
            ax1 = ax[0]
        else:
            fig, ax1 = plt.subplots()
        fig.set_size_inches(15,10)
        ax2 = ax1.twinx()
        ax1.plot(frac[0:ending], scc[0:ending], 'g.')
        ax2.plot(frac[0:ending], deg[0:ending], 'b.')
        ax1.set_xlabel('Fraction of original edges removed')
        ax1.set_ylabel('Number of Strongly connected components', color='g')
        ax2.set_ylabel('Average in-Degree', color='b')

    if plot_gc:
        if plot_ssc:
            ax3 = ax[1] 
        else:
            fig, ax3 = plt.subplots()
            fig.set_size_inches(15,10)
      
        ax4 = ax3.twinx()
        ax3.plot(frac[0:ending], gc[0:ending], 'g.')
        ax4.plot(frac[0:ending], deg[0:ending], 'b.')
        ax3.set_xlabel('Fraction of original edges removed')
        ax3.set_ylabel('Size of the Giant Component', color='g')
        ax4.set_ylabel('Average in-Degree', color='b')
        fig.suptitle('Removing connections with method = ' + method_removal)
        
        plt.show()
    
    if plot_activity:
        fig, ax1 = plt.subplots()
        fig.set_size_inches(10,5)
        ax2 = ax1.twinx()
        ax1.plot(frac[0:ending], kOP1[0:ending], 'g.')
        ax2.plot(frac[0:ending], deg[0:ending], 'b.')
        ax1.set_xlabel('Fraction of original edges removed')
        ax1.set_ylabel('Mean Kuramoto O.P', color='g')
        ax2.set_ylabel('Average in-Degree', color='b')
        
        plt.show()
        
    if activity:
        return frac, scc, deg, ending, gc, BP
    else:
        return frac, scc, deg, ending, gc
    
    
def plot_activity_properties(frac,scc,deg,ending,gc,BP):
        
        """
        Function to plot activity as we remove the network. Only to be used with percolation_netw_statistics function to plot Kuramoto Order Parameter anmd other Burst properties.
        
        """
        
        kOP                = np.zeros(len(BP))
        IBI                = np.zeros(len(BP))
        burst_dur          = np.zeros(len(BP))
        spikes_per_burst   = np.zeros(len(BP))
        burster_proportion = np.zeros(len(BP))
        
        
        for i in range(len(BP)):
            kOP[i]                = BP[i].kOP_mean
            IBI[i]                = BP[i].IBI_mean
            burst_dur[i]          = BP[i].burst_duration_mean
            spikes_per_burst[i]   = BP[i].spikes_per_burst_mean
            burster_proportion[i] = BP[i].burster_proportion_mean
        
        fig, ax1 = plt.subplots()
        fig.set_size_inches(10,5)
        ax2 = ax1.twinx()
        ax1.plot(frac[0:ending], kOP, 'g.')
        ax2.plot(frac[0:ending], deg[0:ending], 'b.')
        ax1.set_xlabel('Fraction of original edges removed')
        ax1.set_ylabel('Mean Kuramoto O.P', color='g')
        ax2.set_ylabel('Average in-Degree', color='b')
        
        plt.show()
    
    
        fig, ax1 = plt.subplots()
        fig.set_size_inches(10,5)
        ax2 = ax1.twinx()
        ax1.plot(frac[0:ending], IBI, 'g.')
        ax2.plot(frac[0:ending], deg[0:ending], 'b.')
        ax1.set_xlabel('Fraction of original edges removed')
        ax1.set_ylabel('IBI', color='g')
        ax2.set_ylabel('Average in-Degree', color='b')
        
        plt.show()
        
        fig, ax1 = plt.subplots()
        fig.set_size_inches(10,5)
        ax2 = ax1.twinx()
        ax1.plot(frac[0:ending], burst_dur, 'g.')
        ax2.plot(frac[0:ending], deg[0:ending], 'b.')
        ax1.set_xlabel('Fraction of original edges removed')
        ax1.set_ylabel('Burst Duration', color='g')
        ax2.set_ylabel('Average in-Degree', color='b')
        
        plt.show()
        
        fig, ax1 = plt.subplots()
        fig.set_size_inches(10,5)
        ax2 = ax1.twinx()
        ax1.plot(frac[0:ending], spikes_per_burst, 'g.')
        ax2.plot(frac[0:ending], deg[0:ending], 'b.')
        ax1.set_xlabel('Fraction of original edges removed')
        ax1.set_ylabel('Spikes per Burst', color='g')
        ax2.set_ylabel('Average in-Degree', color='b')
        
        plt.show()
        
        fig, ax1 = plt.subplots()
        fig.set_size_inches(10,5)
        ax2 = ax1.twinx()
        ax1.plot(frac[0:ending], burster_proportion, 'g.')
        ax2.plot(frac[0:ending], deg[0:ending], 'b.')
        ax1.set_xlabel('Fraction of original edges removed')
        ax1.set_ylabel('Proportion of bursters', color='g')
        ax2.set_ylabel('Average in-Degree', color='b')
        
        plt.show()