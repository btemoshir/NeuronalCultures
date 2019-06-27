#!/usr/bin/env python3
#-*- coding:utf-8 -*-

'''
Neuronal Network Generation of aeif_psc_alpha neurons uniformly distributed in a
circular culture with an exponential distance rule connectivity with a
delay and a constant synaptic weight.
'''

#Simulator
import nngt
import nngt.simulation as ns
import nest 

#Maths
import matplotlib.pyplot as plt
import numpy as np

def Make_Network(net_file, num_omp, culture_radius, num_neurons, 
avg_deg, lambda_edr, delay_slope, delay_offset, syn_weight, 
neuron_params = None, plot_degrees = False, plot_delays = False):
    '''
    Create a NNGT spatial network
    
    Params : 
    --------
    
    - net_file       : string,
            Name of the file to save the network, with extension
    - num_omp        : int,
            Number of threads to suse for NNGT
    - culture_radius :  float,
            Radius of the culture in micrometers
    - num_neurons    : int,
            Total number of neurons
    - neuron_params  : dict of dict , default None, default params of NEST
            Dictionnary containing the parameters for aeif_psc_alpha
            neurons, all values not inside this dict will stay at default 
            value (NEST)
            First keys contain the names (string) of the groups of neurons,
            second keys are the parameters as a dict. 
    - avg_deg        : int,
            Average degree of neurons
    - lambda_edr     : float,
            Characteristic scale for the exponential distance rule in micrometers
    - delay_slope    : float,
            Slope of the linear calculated delay as a function of the distance
            in milliseconds per micrometers
    - delay_offset   : float, 
            Offset of the linear calculated delay in milliseconds
    - syn_weight     : float, or dict 
            Synaptic weight for all connections, if dict must be parametrize by
            synaptic weigh distribution
    - plot_degrees   : bool, Default False
            Wether to plot the degrees distribution or not
    - plot_delays   : bool, Default False
            Wether to plot the delays distribution or not        
    
    Return :
    ------
            :class: '~NNGT.SpatialNetwork'
            
    Note : NEST Kernel is reset before and after the simulation, set to 
    default in the end

    '''
    # Set the parameters of NNGT
    nngt.set_config('omp', num_omp)
    rdmint        = np.random.random_integers(1, high = 1000)
    rdmseeds      = np.random.random_integers(1, high = 1000, size = num_omp)
    nngt.seed(msd = rdmint, seeds = rdmseeds)

    # Initialize the culture
    # Create the culture's environment
    culture = nngt.geometry.Shape.disk(culture_radius)
    
    # Create the neuronal groups
    if neuron_params is None:
        group_names = ['aeif_psc_alpha']
        groups      = [nngt.NeuralGroup(num_neurons, neuron_model='aeif_psc_alpha')]
        pop         = nngt.NeuralPop.from_groups(groups, names = group_names)
    else:
        group_names = neuron_params.keys()
        nb_n        = [neuron_params[name]['nb_neurons'] for name in group_names]
        for name in group_names:
            del neuron_params[name]['nb_neurons']
        groups      = [nngt.NeuralGroup(nb_neurons, neuron_model='aeif_psc_alpha', neuron_param=neuron_params[group]) for group,nb_neurons in zip(group_names,nb_n)]
        pop         = nngt.NeuralPop.from_groups(groups, names = group_names)
        

    # Create the Network
    print('Making Network')
    # Distance distribution 
    dr = 'distance_rule'
    p_dr = {'scale': lambda_edr, 'rule': 'exp'}
    
    # Connection delay
    d   = {
        'distribution': 'lin_corr', 'correl_attribute': 'distance',
        'slope': delay_slope, 'offset': delay_offset }
    
    net = nngt.SpatialNetwork(shape=culture, population=pop, delays=d)
    
    nngt.generation.connect_neural_groups(net, group_names, group_names, dr, 
    avg_deg = avg_deg , weights = syn_weight, **p_dr)
    
    
    # Degree distribution
    if plot_degrees == True:
        nngt.plot.degree_distribution(net, ['in', 'out'], show=True)
    
    # Delay distribution
    if plot_delays == True:
        d_dist = net.get_delays()
        bin_d = np.linspace(np.min(d_dist),np.max(d_dist),25)
        plt.hist(d_dist,bins=bin_d,density=True)
        plt.plot(np.mean(d_dist),[0.],'*',label = 'mean')
        plt.plot(np.median(d_dist),[0.],'.',label = 'median')
        plt.ylim(ymin=-0.01)
        plt.legend()
        plt.ylabel('percentage of synaptic delay')
        plt.xlabel('synaptic delay')
        plt.show()
            
    if net_file is not None: 
        net.to_file(net_file)
    return net

def Simulation(net_file, num_omp, sim_time, noise_rate, noise_weight_fraction,
noise, save_spk, I_excitation = None, delay_excitation = None,
noise_on_neurons = None, use_database = False, w_IC = None, Vm_IC = None, show_activity = False,
return_activity = True, animation = False, anim_name = None):
    '''
    Make a nest simulation and save the activity text file as 
    Nest id ~ spike time ~ X posiion ~ Y position
    
    Params : 
    --------
    - net              : string,
            path to a saved :class: '~NNGT.SpatilaNetwork' used for the 
            simulation
    - num_omp          : int,
            Number of threads to suse for NNGT and Nest
    - sim_time         : float,
            Time of the simulation in milliseconds
    - noise_rate       : float,
            Rate of the random noise 
    - noise_weight_fraction :
            weight fraction of the noise : multiplicatif factor used 
            like a synaptic weight
    - noise_on_neurons : 1D array
            list of NEST ids to set the noise only on those. 
    - use_database     : bool, default False
            Wether to the database or nor
    - w_IC             : 3-tuples
            Initial condition for adaptative current w. it must be 
            ('distrib_name', double, double)
    - Vm_IC            : 3-tuples
            Initial condition for adaptative current w. it must be 
            ('distrib_name', double, double)
    - noise            : string
            Type of noise, can be 'Minis', 'Pnoise', 'Periodic' (not noisy)
            Be careful, it will modify the network in Nest with regards to the one 
            in NNGT, see terminal WARNING.
    - I_excitation     : float,
            If noise is 'Periodic', set the current input to make the 
            fictitious neuron that will excitate the culture (375.5pA for
            66ms interval between each occurences) 
    - delay_excitation : list of float of size numer of neurons
            List of the delay to set up the constant input
    - save_spk : string
            Path to sve the activity
    - show_activity : bool,
            Wether to plot and show the raster plot of activity
            
    Return : Nothing
    --------
    '''
    # Get the network and the population names
    net         = nngt.load_from_file(net_file)
    pop         = net.population
    group_names = pop.keys()
    
    # Seed the random number generator and set NNGT and Nest kernels
    nest.ResetKernel()    
    rdmint        = np.random.random_integers(1, high = 1000)
    rdmseeds      = list(np.random.random_integers(1, high = 1000, size = num_omp))
    nest.SetKernelStatus({'local_num_threads' : num_omp, 'rng_seeds' : rdmseeds})
    nngt.set_config({'use_database' : use_database, "multithreading" : True, "omp" : num_omp, })
    nngt.seed(msd = rdmint, seeds = rdmseeds)
    
    # Send network to nest
    gids = net.to_nest()
    
    if noise_on_neurons is None:
        noise_on_neurons = gids
    
    # Create the excitation 
    if noise == 'Minis':
        in_deg    = np.average(net.get_degrees('in'))
        base_rate = noise_rate / in_deg
        ns.set_minis(net, base_rate, weight = noise_weight_fraction, gids = noise_on_neurons)
        
    elif noise == 'Pnoise':
        pgs    = nest.Create("poisson_generator", params = {'rate' : noise_rate})
        co_w   = np.mean(net.get_weights())
        nest.Connect(pgs, noise_on_neurons, syn_spec = {'weight' : noise_weight_fraction*co_w})
    
    elif noise == 'Periodic':
        neuron_cst_I = nest.Create('iaf_psc_alpha', params={'I_e' : I_excitation})
        co_w   = np.mean(net.get_weights())
        for i, nids in enumerate(noise_on_neurons):
            nest.Connect(neuron_cst_I, [nids], syn_spec = {'weight' : noise_weight_fraction*co_w, 'delay' : delay_excitation[i]})
    
    if w_IC is not None:
        ns.randomize_neural_states(net, instructions = {'w' : w_IC})
    if Vm_IC is not None:
        ns.randomize_neural_states(net, instructions = {'V_m' : Vm_IC})
    
    #Database
    if use_database == True:
        nngt.database.log_simulation_start(net, "nest-2.14", save_network = True)
        
    # Simulate and Record
    print('Start Simulation')
    recorders, recorded = ns.monitor_groups(group_names, network=net)
    nest.Simulate(sim_time)
    
    if animation:
        anim = nngt.plot.AnimationNetwork(recorders, net, trace = 30.)
        anim.save_movie(anim_name, interval = 20., fps = 15)
    
    times   = nest.GetStatus(recorders[0])[0]['events']['times']
    senders = nest.GetStatus(recorders[0])[0]['events']['senders']
    
    #Database
    if use_database == True:
        nngt.database.log_simulation_end()
        
    if show_activity == True:
        print('ploting activity')
        nngt.simulation.plot_activity(show = True)
    
    #save activity and reset Nest kernel
    ns.save_spikes(save_spk, recorder = recorders, network = net)
    nest.ResetKernel()
    
    print('Simulation done and saved')
    
    if return_activity:
        return senders , times


if __name__ == '__main__':
    save_spk   = 'test' + '.txt'
    net_file   = 'test' + '.el'
    anim_name  = 'test' + '.mp4'


    # Synapse Parameters
    w = 20.
    # Make the network
    net = Make_Network(net_file, num_omp = 12, culture_radius = 1000, num_neurons = 500, avg_deg = 50, lambda_edr = 150.,
    delay_slope = 0.01, delay_offset = 2., syn_weight = w, )
    # Simulate
    senders , spikes = Simulation(net_file, 12, sim_time = 15000, noise_rate = 10., noise_weight_fraction = 0.5, 
    noise = 'Minis', save_spk = save_spk, show_activity = True, return_activity = False,
    animation = False, anim_name = anim_name)
    
### EDIT: MOSHIR

##TODO: Create a function to get activity output as desired
##TODO: Create a function to percolate the network as desired