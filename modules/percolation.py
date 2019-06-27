#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import nngt
import nest
import nngt.simulation as ns


def percolationNodes(net,num_nodes,method_initiation='uniform', method_removal='uniform', method_rewire = None, frac_rewire = 0.5):
    
    """
    Function which percolates the nodes of the network according to certain rules.
    
    Inputs:
     
    - net = network to be percolated
    - num_nodes = number of nodes to be removed
    - frac_rewire = fraction of in connections to the node being removed to be rewired
    
    - method_initiation ='uniform' : randomly choose initialtion
                         'in-degree': highest in-degree initiation
                         'out-degree': highest out-degree initiation
                         'edge_proximity': closest to the edge is removed first
                         'centre_proximity': closest to the centre is removed first
    
    - method_removal =  'uniform': randomly remove nodes equal to num_nodes
                        'in-degree': remove nodes randomly but nodes are removed proportionately to their in-degree
                        'out-degree':same as indegree dependent but with out-degree
                        'connection-uniform': only nodes connected to the initiation are randomly attacked, and continued
                        'connection-in-degree': only nodes connected to the initiation are attacked with probablility of their in-degree, and continued
                        'distance-uniform' : the node removal happens progressively with distance
                        'distance-proportional': the node removal is inversely proportional to the distance from the last removed node
                        'distance-exp': the node removal is with probability that is exp(-distance) 
                        
    - method_rewire =   'None': no rewiring, all in and out connections to a removed node are removed
                        'uniform': rewire a fraction 'frac_rewire' of all the incoming connections to the other neurons unifromly
                        'in-degree': rewire a fraction 'frac_rewire' of all the incoming connections to the other neurons with probability proportional to their in-degree
                        'out-degree': same as above but with out-degree
                        'connected-in': the rewiring probability is proportional to already existing connections between the node deleted and the neuron to which we rewire
                        'connected-out': the rewiring probability is inversly proportional to already existing connections between the node who's out connected was removed and the neuron to which we rewire
                        'spatial-in': the rewiring probability is proportional to the spatial distance between the node deleted and the neuron to which we rewire
                        'spatial-out': the rewiring probability is inversly proportional to the spatial distance between the node who's out connection deleted and the neuron to which we rewire
                        'spatial-exp': the rewiring probability is exp(-distance)
                                        
    Returns:
    
    - In place changes the network
    - nodes_removes = an array of the gids of the removed nodes
    
    """
    
    nodes_removed           = []
    connections_removed     = []
    #connections_added       = []
    position_removed        = []
    connections_removed_in  = []
    
    if method_initiation == 'uniform':
        removed = np.random.choice(net.nodes())
        
    elif method_initiation == 'in-degree':
        removed = np.asarray(net.nodes())[np.argmax(nngt.analysis.node_attributes(net,'in-degree'))]
        
    elif method_initiation == 'out-degree':
        removed = np.asarray(net.nodes())[np.argmax(nngt.analysis.node_attributes(net,'out-degree'))]
        
    elif method_initiation == 'edge_proximity':
        x,y = net.get_positions()[:,0], net.get_positions()[:,1]
        dis = (x**2 + y**2)**0.5
        removed = np.asarray(net.nodes())[np.argmax(dis)]
        
    elif method_initiation == 'centre_proximity':
        x,y = net.get_positions()[:,0], net.get_positions()[:,1]
        dis = (x**2 + y**2)**0.5
        removed = np.asarray(net.nodes())[np.argmin(dis)]
    
    connections_removed.append(np.asarray(net.edges)[np.where(np.asarray((net.edges))[:,0] == removed)])
    connections_removed_in.append(np.asarray(net.edges)[np.where(np.asarray((net.edges))[:,1] == removed)])
    position_removed.append(net.get_positions(removed))
    
    nngt.core.BaseGraph.remove_node(net,removed)
    nodes_removed.append(removed)
    added_conn = rewire_edges(net,connections_removed_in[-1],frac_rewire,method_rewire,position_removed[-1],removed)
    
    if num_nodes > 1:
        
        for i in range(num_nodes-1):

            if method_removal == 'uniform':
                removed = np.random.choice(net.nodes())

            elif method_removal == 'in-degree':
                p_in = nngt.analysis.node_attributes(net,'in-degree')
                p_in  = p_in.astype('float64')
                p_in = p_in/np.sum(p_in)
                removed = np.random.choice(net.nodes(),p=p_in)

            elif method_removal == 'out-degree':
                p_out = nngt.analysis.node_attributes(net,'out-degree')
                p_out = p_out.astype('float64')
                p_out = p_out/np.sum(p_out)
                removed = np.random.choice(net.nodes(),p=p_out)

            elif method_removal == 'connection-uniform':
                removed = np.random.choice(connections_removed[-1][:,1])

            elif method_removal == 'connection-in-degree':
                p_in  = nngt.analysis.node_attributes(net,'in-degree',nodes = (connections_removed[-1][:,1]))
                p_in  = p_in.astype('float64')
                p_in  = p_in/np.sum(p_in)
                removed = np.random.choice((connections_removed[-1][:,1]),p=p_in)

            elif method_removal == 'distance-uniform':
                x,y = net.get_positions()[:,0], net.get_positions()[:,1]
                d = np.sqrt((x- position_removed[-1][0])**2 + (y-position_removed[-1][1])**2)
                p_dis = np.sum(d)/d
                removed = np.random.choice(net.nodes(),p=p_dis)

            elif method_removal == 'distance-exp':
                x,y = net.get_positions()[:,0], net.get_positions()[:,1]
                d = np.sqrt((x- position_removed[-1][0])**2 + (y-position_removed[-1][1])**2)
                p_exp = np.exp(-d)/(np.sum(np.exp(-d)))
                removed = np.random.choice(net.nodes(),p=p_exp)

            connections_removed.append(np.asarray(net.edges)[np.where(np.asarray((net.edges))[:,0] == removed)])
            connections_removed_in.append(np.asarray(net.edges)[np.where(np.asarray((net.edges))[:,1] == removed)])
            position_removed.append(net.get_positions(removed))

            nngt.core.BaseGraph.remove_node(net,removed)
            nodes_removed.append(removed)
            #added_conn = rewire_edges(net,connections_removed_in[-1],frac_rewire,method_rewire,position_removed[-1],removed)
        
        ## TODO: Add support for connections_added to be output which are not removed again
        
    return nodes_removed,#connections_removed,position_removed,connections_removed_in ,connections_added
  
    
    
def rewire_edges(net,removed_edges,frac_rewire,method_rewire,position_removed_last=None,removed=None):
    
    
    """
    Function which creates new connections between nodes based on some rules:
    (to be used with 'percolationNodes()' function)
    
    Inputs:
    
    - net = the nngt spatial network
    - removed_edges = The edges that we have removed, a fraction of which will be rewired
    - frac_rewire = the fraction of the removed edges to be rewired
    - position_removed = the spatial coordinates of the node that was last removed as [x,y], only needed for position based methods
    - removed = the index of the node last removed, only needed for 'connected -in ' method
    
    - method_rewire =   'None': no rewiring, all in and out connections to a removed node are removed
                        'uniform': rewire a fraction 'frac_rewire' of all the incoming connections to the other neurons unifromly
                        'in-degree': rewire a fraction 'frac_rewire' of all the incoming connections to the other neurons with probability proportional to their in-degree
                        'out-degree': same as above but with out-degree
                        'connected-in': the rewiring probability is proportional to already existing connections between the node deleted and the neuron to which we rewire
                        'connected-out': the rewiring probability is inversly proportional to already existing connections between the node who's out connected was removed and the neuron to which we rewire
                        'spatial-in': the rewiring probability is inversely proportional to the spatial distance between the node deleted and the neuron to which we rewire
                        'spatial-out': the rewiring probability is inversly proportional to the spatial distance between the node who's out connection was deleted and the neuron to which we rewire
                        'spatial-exp': the rewiring probability is exp(-distance) where distance is from the nodes who's outgoing connection was removed
    
    Outputs:
    
    - In place changes the network
    - connections_added = an array of tuples of the connections that were added in the rewiring.
    
    """
    
    
    connections_added = []
    
    if method_rewire is None:
        print('No Rewiring')
    
    else:
        size = int(np.floor(frac_rewire*len(removed_edges)))
        node_out = np.random.choice(removed_edges[:,1], size=size)
        
        for i in range(len(node_out)):
        
            if method_rewire == 'uniform':
                node_in = np.random.choice(np.delete((net.nodes()),node_out[i]))
                print(node_in)
                nngt.Graph.add_edge(net,node_out[i],node_in)
            
            elif method_rewire == 'in-degree':
                p_in  = nngt.analysis.node_attributes(net,'in-degree', nodes=np.delete(net.nodes(),node_out[i]))
                p_in  = p_in.astype('float64')
                p_in  = p_in/np.sum(p_in)
                node_in = np.random.choice(np.delete(np.random.choice(net.nodes()),node_out[i]),p=p_in)
                nngt.Graph.add_edge(net,node_out[i],node_in)
                
            elif method_rewire == 'out-degree':
                p_out  = nngt.analysis.node_attributes(net,'out-degree', nodes=np.delete(net.nodes(),node_out[i]))
                p_out  = p_out.astype('float64')
                p_out  = p_out/np.sum(p_out)
                node_in = np.random.choice(np.delete(net.nodes(),node_out[i]),p=p_out)
                nngt.Graph.add_edge(net,node_out[i],node_in)
            
            elif method_rewire == 'connected-in':
                connected_nodes = (np.asarray(net.edges)[np.where(np.asarray((net.edges))[:,0] == removed)])[:,0]
                node_in = np.random.choice(connected_nodes)
                nngt.Graph.add_edge(net,node_out[i],node_in)
                
            elif method_rewire == 'connected-out':
                connected_nodes = (np.asarray(net.edges)[np.where(np.asarray((net.edges))[:,0] == node_out[i])])[:,0]
                node_in = np.random.choice(connected_nodes)
                nngt.Graph.add_edge(net,node_out[i],node_in)
                
            elif method_rewire == 'spatial-in':
                x,y = net.get_positions(nodes=np.delete(net.nodes(),node_out[i]))[:,0], net.get_positions(nodes=np.delete(net.nodes(),node_out[i]))[:,1]
                d = np.sqrt((x- position_removed_last[0])**2 + (y-position_removed_last[1])**2)
                p_dis = np.sum(d)/d
                node_in = np.random.choice(np.delete(net.nodes(),node_out[i]),p=p_dis)
                nngt.Graph.add_edge(net,node_out[i],node_in)
                
            elif method_rewire == 'spatial-out':
                x,y = net.get_positions(nodes=np.delete(net.nodes(),node_out[i]))[:,0], net.get_positions(nodes=np.delete(net.nodes(),node_out[i]))[:,1]
                d = np.sqrt((x- net.get_positions(node_out[i])[0])**2 + (y-net.get_positions(node_out[i])[1])**2)
                p_dis = np.sum(d)/d
                node_in = np.random.choice(np.delete(net.nodes(),node_out[i]),p=p_dis)
                nngt.Graph.add_edge(net,node_out[i],node_in)
                
            elif method_rewire == 'spatial-exp':
                x,y = net.get_positions(nodes=np.delete(net.nodes(),node_out[i]))[:,0], net.get_positions(nodes=np.delete(net.nodes(),node_out[i]))[:,1]
                d = np.sqrt((x- net.get_positions(node_out[i])[0])**2 + (y-net.get_positions(node_out[i])[1])**2)
                p_exp = np.exp(-d)/np.sum(np.exp(-d))
                node_in = np.random.choice(np.delete(net.nodes(),node_out[i]),p=p_exp)
                nngt.Graph.add_edge(net,node_out[i],node_in)
                
            connections_added.append((node_out[i],node_in))
                                     
    return np.asarray(connections_added)


def percolationEdges(net, num_remove = None, frac_remove = False, method_removal='uniform', method_rewiring=None, burst_times= False, burst_neurons = False ):
    
    """
    Function which removes conncetions in the network depending on rules:
    
    Inputs:
    
    - net = the nngt network
    - num_remove = the number of connections to remove 
    - frac_remove = the fraction of total connections to be removed
      (Use either frac_remove or num_remove, Not together)
    
    - method_removal =  'uniform': randomly uniformly remove connections
                        'in-degree': randomly remove in connection from nodes with probability proportional to the node's in-degree
                        'in-degree-inverse': randomly remove in connection from nodes with probability inversely proportional to the node's in-degree
                        'out-degree': randomly remove out connection from nodes with probability proportional to the node's out-degree
                        'out-degree-inverse': randomly remove out connection from nodes with probability inversely proportional to the node's out-degree
                        'burst-initiators': remove out connections of nodes with probability inversely proportional to their spiking time in a burst.
    - burst_times = the times of bursts for all the bursts as a list of list. This is outout by burst detection functions. Only to be used with 'burst_initiators' method_removal.
    - burst_neurons =  the neuron identity of bursts for all the bursts as a list of list. This is outout by burst detection functions. Only to be used with 'burst_initiators' method_removal.
    
    """
    edges_removed = []
    
    edges = np.asarray(net.edges())

    if method_removal == 'uniform':
        
        if frac_remove:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=int(np.floor(frac_remove*len(edges))),replace=False)]
        else:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=num_remove,replace=False)]
    
    elif method_removal == 'in-degree':
        
        p_in = nngt.analysis.node_attributes(net,'in-degree')
        p_in  = p_in.astype('float64')        
        p_edges = np.zeros(len(edges))
        for i in range(len(edges)):
            p_edges[i] = p_in[edges[i][1]]
        p_edges = p_edges/np.sum(p_edges)
        
        if frac_remove:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=int(np.floor(frac_remove*len(edges))),p=p_edges,replace=False)]
        else:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=num_remove,p=p_edges,replace=False)]
        
    elif method_removal == 'out-degree':
        
        p_in = nngt.analysis.node_attributes(net,'out-degree')
        p_in  = p_in.astype('float64')        
        p_edges = np.zeros(len(edges))
        for i in range(len(edges)):
            p_edges[i] = p_in[edges[i][0]]
        p_edges = p_edges/np.sum(p_edges)
        
        if frac_remove:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=int(np.floor(frac_remove*len(edges))),p=p_edges,replace=False)]
        else:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=num_remove,p=p_edges,replace=False)]
            
    elif method_removal == 'in-degree-inverse':
        
        p_in = nngt.analysis.node_attributes(net,'in-degree')
        p_in  = p_in.astype('float64')
        p_edges = np.zeros(len(edges))
        for i in range(len(edges)):
            p_edges[i] = 1./p_in[edges[i][1]]
        p_edges = p_edges/np.sum(p_edges)
        
        if frac_remove:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=int(np.floor(frac_remove*len(edges))),p=p_edges,replace=False)]
        else:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=num_remove,p=p_edges,replace=False)]
            
    elif method_removal == 'out-degree-inverse':
        
        p_in = nngt.analysis.node_attributes(net,'out-degree')
        p_in  = p_in.astype('float64')
        p_edges = np.zeros(len(edges))
        for i in range(len(edges)):
            p_edges[i] = 1./p_in[edges[i][0]]
        p_edges = p_edges/np.sum(p_edges)
        
        if frac_remove:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=int(np.floor(frac_remove*len(edges))),p=p_edges,replace=False)]
        else:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=num_remove,p=p_edges,replace=False)]
        
    elif method_removal == 'burst-initiators':
        N                             = net.node_nb()
        if len(burst_neurons) == 0:
            p_in = np.ones(N)
        else:
            Bidx                          = np.random.choice(np.arange(len(burst_neurons)))
            unique_neurons,unique_indices = np.unique(burst_neurons[Bidx],return_index=True)
            unique_times                  = burst_times[Bidx][unique_indices]
            unique_neurons                = unique_neurons - 1
            mask                          = np.isin(np.arange(N), unique_neurons, invert=True)
            unique_neurons                = np.append(unique_neurons,np.arange(N)[mask])
            unique_times                  = np.append(unique_times,np.arange(N)[mask] + 100000.)
            unique_times                  = unique_times[np.argsort(unique_neurons)]
            unique_neurons                = np.sort(unique_neurons)

            p_in                          = 1./(1+ unique_times-min(unique_times))
            p_edges                       = np.zeros(len(edges))
        
        for i in range(len(edges)):
            p_edges[i] = p_in[edges[i][0]]
        p_edges = p_edges/np.sum(p_edges)
        
        if frac_remove:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=int(np.floor(frac_remove*len(edges))),p=p_edges,replace=False)]
        else:
            ed_rem = edges[np.random.choice(np.arange(len(edges)),size=num_remove,p=p_edges,replace=False)]
                   
    
    nngt.core.BaseGraph.remove_edges_from(net,ed_rem)    
    edges_removed.append(ed_rem)
        
    
    return edges_removed

