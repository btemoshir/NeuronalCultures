#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Network
import nngt
import nest 

#Maths
import numpy as np
from scipy.signal import argrelextrema as localext
from sklearn.cluster import DBSCAN

#plot
import matplotlib.pyplot as plt
import itertools as it

#Shapely
from shapely.geometry import Point, MultiPoint, Polygon, LineString, MultiLineString
from shapely.prepared import prep

'''
Note :
------

The first idea of this python doc is to create functions to analysis 
spatial dynamic of simulated (and in vitro) neuronal culture activity. 

The 'algorithm' goes as follow : 
1. define a phase for all neuron, and for the network as the mean phase
2. The Network phase can define starting and ending time of a burst as a
maximum and a minimum in a specific range. With this we define a burst 
as a minimum number of spiking/desorganized neurons. 
3. the neurons that spiked just before the burst start are sorted out. 
(just before the burst = inbetween 1. the mean between of the starting 
time of the considered burst and the end of the previous one and 2. the 
start of the burst)
  3.1 They are first approximately sorted by a DBSCAN algorithm to dif-
  ferentiate different region of the culture and remove noisy spike.
  (DBSCAN params are the min number of neurons to be a cluster which is 3
  -see 3.2- and the range for looking for neighbors set to a fourth or 
  a fifth of the culture typical lenght)
  3.2 Then for each detected cluster, they are called center of activity
  if and only if there are more spiking neurons than non spiking neurons
  inside the hull formed by the neurons inside the cluster.
  (need at lest 3 neurons in the cluster). otherwise the detected cluster 
  is regarded as noise.
4. The previous hull will be use to define a mathematical base to develop
the phase onto. An expansion factor is taken to increase the funtion area

TO DO :
What if there is no center of activity ? 
what if there are new centers of activity ?
'''

"""Paths and patches"""

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy import asarray, concatenate, ones

class Polygon(object):
    # Adapt Shapely or GeoJSON/geo_interface polygons to a common interface
    def __init__(self, context):
        if isinstance(context, dict):
            self.context = context['coordinates']
        else:
            self.context = context

    @property
    def exterior(self):
        return (getattr(self.context, 'exterior', None)
                or self.context[0])

    @property
    def interiors(self):
        value = getattr(self.context, 'interiors', None)
        if value is None:
            value = self.context[1:]
        return value


def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    if hasattr(polygon, 'geom_type'):  # Shapely
        ptype = polygon.geom_type
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    else:  # GeoJSON
        polygon = getattr(polygon, '__geo_interface__', polygon)
        ptype = polygon["type"]
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon['coordinates']]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    vertices = concatenate([
        concatenate([asarray(t.exterior)[:, :2]] +
                    [asarray(r)[:, :2] for r in t.interiors])
        for t in polygon])
    codes = concatenate([
        concatenate([coding(t.exterior)] +
                    [coding(r) for r in t.interiors]) for t in polygon])

    return Path(vertices, codes)


def PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes):

      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)

    """
    return PathPatch(PolygonPath(polygon), **kwargs)

def normalise_array(array):
    return (array - np.min(array)) / ( np.max(array) - np.min(array))

def convolve_gauss(fr, sigma, dt, crop=5.):
    '''
    Convolve a spiketrain by an gaussian kernel with width `sigma`.

    Parameters
    ---------
    - fr : 1D array
        Firing rate.
    - sigma : double
        Timescale for the gaussian decay.
    - dt : double
        Timestep for the "continuous" time arrays that will be returned.
    - crop : double, optional (default: 5.)
        Crop the gaussian after `crop*sigma`.

    Returns
    -------
    ts, fr : 1D arrays
    '''
    # create the gaussian kernel
    tkernel  = np.arange(-crop*sigma, crop*sigma + dt, dt)
    ekernel  = np.exp(-(tkernel/(2*sigma))**2)
    #ekernel /= np.sum(ekernel)
    # convolve
    fr = np.array(convolve(fr, ekernel, "same"))
    return fr

def All_Neuron_Phase(Activity_Raster, times):
    '''
    Compute the phase at times (1d array) for all neurons
    
            !!! times must not be too long !!!
            
    '''
    phases = np.zeros(shape = (len(Activity_Raster),len(times)))
    for i,r in enumerate(Activity_Raster):
        r = np.array(r)
        isi = np.append(np.diff(r),[np.inf])
        idx = np.digitize(times,r) - 1
        ph = (times - r[idx]) / isi[idx]
        idx = np.where(ph < 0)[0]
        ph[idx] = 0
        phases[i] = ph
        
    return phases

def Single_Neuron_Phase(Activity_Raster, time, Kuramoto = True):
    '''
    Compute the phase of all neurons at 'time'
    
    Params :
    --------
    
    - Activity_Raster : nD array of 1D arrays
                       spike trains of all neurons
    - time            : float
                       time at which we compute the phase
    - Kuramoto        : bool
                        Wether to return the phase in [-2 pi , 0] (True,
                        Default) or in [0,1] (False)
    Return :
    --------
            phase of the neurons phi = (t-t_k)/(t_k-t_k-1) as function of time
    '''
    phi = []
    for r in Activity_Raster:
        r = np.array(r)
        isi = np.append(np.diff(r),[np.inf])
        idx = np.digitize(time,r) - 1
        ph = (time - r[idx]) / isi[idx]
        if ph < 0:
            ph = 0.
        phi.append(ph)
    if Kuramoto:
        return np.array(phi)*2*np.pi
    else:
        return np.array(phi)

def Network_Phase(Activity_Raster, times, smooth = False):
    '''
    Compute the phase of all neurons at 'time'
    
    Params :
    --------
    
    - Activity_Raster : nD array of 1D arrays
                       spike trains of all neurons
    - time            : 1d array
                       times at which we compute the phase
    - smooth        : bool
                        Wether to smooth the phase by gaussian convolution
        return :
        --------
                phase of the neurons phi = (t-t_k)/(t_k-t_k-1) as function of time
    '''
    phases = np.zeros(shape = len(times))
    
    for r in Activity_Raster:
        r = np.array(r)
        isi = np.append(np.diff(r),[np.inf])
        idx = np.digitize(times,r) - 1
        ph = (times - r[idx]) / isi[idx]
        idx = np.where(ph < 0)[0]
        ph[idx] = 0
        phases += ph
    phases /= len(Activity_Raster)
    
    if smooth == True:
        dt = times[1]-times[0]
        phases = convolve_gauss(phases, sigma = dt, dt = dt, crop = 2*dt )
    
    return np.array(phases)

def kuramoto_od(phases):
    '''
    Compute the Kuramoto order parameter
    
    use for one time step
    '''
    j = np.complex(0,1)
    S = sum(np.exp(j*np.array(phases))) / len(phases)
    return np.abs(S) , np.angle(S)

def kuramoto_radius(phases):
    '''
    Compute the Kuramoto order parameter
    
    Parameters :
    ------------
    - phases  : nd array shape = (n_neurons, n_times)
    
    Result : 
    -------- 
    Kuramoto as a function of time
    '''
    j = np.complex(0,1)
    S = np.sum(np.exp(j*np.array(phases)), axis = 0) / phases.shape[0]
    return np.abs(S)

def mean_simps(x, y, t1, t2):
    '''
    Compute the mean of y between x1 and x2 with the simps method from
    scipy to compute the integral
    '''
    from scipy.integrate import simps
    
    id1 = np.where(x > x1)[0][0]
    id2 = np.where(x > x2)[0][0]

    y = y[id1:id2]
    x = x[id1:id2]
    
    return simps(y,x) / (y[-1] - y[1])

def mean_direct(y, x, x1, x2):
    '''
    Compute the mean for evenly spaced x
    '''
    id1 = np.where(x > x1)[0][0]
    id2 = np.where(x > x2)[0][0]

    y = y[id1:id2]
    
    return np.mean(y)
    
def Burst_times(Activity_Raster, time_array, th_high = 0.6, 
                th_low = 0.4, ibi_th = 0.9, plot = False):
    '''
    Compute the starting and ending time of a burst
    
    Params :
    --------
    - Activity_Raster : ND array of 1D arrays
                       spike trains of all neurons
    - time_array      : 1D array
                        times for the analaysis
    - th_high         : float
                        upper threshold use to separate the network phase
    - th_low          : float
                        lower threshold use to separate the network phase
    - ibi_th          : float
                        threshold use to separate the detected local maxima
    - plot            : Bool
                        Wether to plot of not the results
    Return :
    --------
            2D array, starting and ending point of bursts
    '''
    def check_minmax(idx):
        array = ( Net_phi[idx] > th_high )
        ar = np.append([0],array)
        br = np.append(array,[1])
        return ar + br
    
    def is_there_one(array):
        if 0 in array or 2 in array:
            return True
        else:
            return False
    
    #Compute the network phase
    dt = time_array[1] - time_array[0]
    Net_phi = Network_Phase(Activity_Raster, time_array)
    #Net_phi = convolve_gauss(Net_phi, sigma = 5*dt, dt=dt, crop = 5.)
    
    #f,a = plt.subplots()
    #a.plot(time_array,Net_phi)
    
    #Compute argmin and argmax of the network phase
    mask_up = (Net_phi > th_high)
    ph_up = Net_phi[mask_up]
    max_idx = localext(ph_up, np.greater)[0]
    
    mask_dw = (Net_phi < th_low)
    ph_dw = Net_phi[mask_dw]
    min_idx = localext(ph_dw, np.less)[0]
    
    if len(ph_up) == 0 or len(ph_dw) == 0:
        print('No oscillations')
    else:
        #first extrema is a max
        ts = time_array[mask_up][max_idx[0]]
        pop = np.where(time_array[mask_dw][min_idx] < ts)[0]
        min_idx = np.delete(min_idx, pop)
        #last extrema is a min
        ts = time_array[mask_dw][min_idx[-1]]
        pop = np.where(time_array[mask_up][max_idx] > ts)[0]
        max_idx = np.delete(max_idx, pop)

        time_bursts = []
        
        #Clustering extrema
        border_idx = []
        #find borders of clusters of minima and maxima
        for times,idx in [
                           [time_array[mask_up], max_idx], 
                           [time_array[mask_dw], min_idx]
                         ]:
            #if idx[0] == min_idx[0]:
                #a.vlines(times[idx], [0.25], [0.75], 'r')
            imi = np.diff(times[idx])
            th = np.mean(imi)
            w = np.where(imi > th*ibi_th)[0] + 1
            border_idx.append(idx[w])
            #if idx[0] == min_idx[0]:
                #a.vlines(times[idx[w]], [0], [0.5], 'b')
        
        
        idx = np.where(time_array[mask_dw][border_idx[1]] 
                       > time_array[mask_up][border_idx[0][-1]])[0]
        #take only one last min
        if len(idx) > 1:
            border_idx[1] = np.delete(border_idx[1], idx[1:])
        #get rid of the last border if its a maximum
        else:
            idx = np.where(time_array[mask_up][border_idx[0]]
                                    > time_array[mask_dw][border_idx[1][-1]])[0]
            if len(idx) > 0:
                border_idx[0] = np.delete(border_idx[0], idx)
        
        
        idx = np.where(time_array[mask_up][border_idx[0]] 
                       < time_array[mask_dw][border_idx[1][0]])[0]
        #take only one first max
        if len(idx) > 1:
            border_idx[0] = np.delete(border_idx[0], idx[:-1])
        #get rid of first minimum
        else:
            idx = np.where(time_array[mask_dw][border_idx[1]]
                                    < time_array[mask_up][border_idx[0][0]])[0]
            if len(idx) > 0:
                border_idx[0] = np.delete(border_idx[0], idx)
                
        #a.vlines(time_array[mask_dw][border_idx[1]], [0.75], [1.], 'r')
        
        #return only one extrema in each clusters
        for times,phi,idx,func,border in [
                           [time_array[mask_up], ph_up, max_idx, np.argmax, border_idx[0]], 
                           [time_array[mask_dw], ph_dw, min_idx, np.argmin, border_idx[1]] 
                         ]:
            
            loop = [0]
            loop.extend(border)
            colors = [plt.cm.viridis(each) for each in np.linspace(0.,1.,len(loop))]
            
            #if idx[0] == min_idx[0]:
                #a.vlines( times[loop], [0.75], [1.], 'k')
            
            for i in range(1,len(loop)):
                ts = loop[i-1]
                te = loop[i]
                
                mask = (idx < te) & (idx >= ts)
                #keep only the largest maxima and smallest minima
                grped_idx = idx[mask]
                gold_idx = func(phi[grped_idx])
                time_bursts.append(times[grped_idx][gold_idx])
                
                #if idx[0] == min_idx[0] and i == 1:
                    #a.vlines(times[grped_idx], [0.5], [.8], 'k')
                    #a.vlines(times[grped_idx][gold_idx], [0.8], [1.], 'grey')
        plt.show()
        time_bursts.sort()
        idx_bursts = [np.where(time_array == tb )[0][0] for tb in time_bursts]
        added_idx = check_minmax(idx_bursts)

        #sort min max alternating to remove double detected max or double detected min
        PROBLEM = is_there_one(added_idx)
        while PROBLEM:

            idx = np.where(added_idx[1:-1] == 0)[0]
            if len(idx) != 0:
                idx = idx[0]
                if Net_phi[idx_bursts[idx]] >= Net_phi[idx_bursts[idx+1]]:
                    idx_bursts.pop(idx)
                else:
                    idx_bursts.pop(idx+1)

            idx = np.where(added_idx[1:-1] == 2)[0]
            if len(idx) != 0:

                idx = idx[0]
                if Net_phi[idx_bursts[idx]] <= Net_phi[idx_bursts[idx+1]]:
                    idx_bursts.pop(idx)
                else:
                    idx_bursts.pop(idx+1)

            added_idx = check_minmax(idx_bursts)
            if added_idx[0] == 0:
                idx_bursts = idx_bursts[1:]
            added_idx = check_minmax(idx_bursts)
            if added_idx[-1] == 2:
                idx_bursts = idx_bursts[:-1]
            added_idx = check_minmax(idx_bursts)
            PROBLEM = is_there_one(added_idx)

        time_bursts = [time_array[i] for i in idx_bursts]

        if len(time_bursts) != 0:
            N_burst = len(time_bursts) / 2 
            ibi = np.mean(np.diff(time_bursts)[range(1,2*N_burst-1,2)])
            if time_bursts[-1] > time_array[-1]-ibi:
                time_bursts.pop(-1)
                time_bursts.pop(-1)
            time_bursts = [[time_bursts[i],time_bursts[i+1]] 
                                for i in range(0,len(time_bursts)-1,2)]
            idx_bursts = [[idx_bursts[i],idx_bursts[i+1]] 
                                for i in range(0,len(time_bursts)-1,2)]

            if plot == True:
                f,a = plt.subplots()
                a.plot(time_array, Net_phi)
                a.vlines(time_bursts, [0]*len(time_bursts), [1]*len(time_bursts), ['r','g'])
                plt.plot()

        else:
            print('No burst detected')
            if plot == True:
                f,a = plt.subplots()
                a.plot(time_array, Net_phi)
                plt.plot()
    
        return np.array(time_bursts), Net_phi
    
def First_To_Fire(Activity_Raster, time_end, time_start):
    '''
    Find neurons that fire before burst
    
    Params : 
    --------
    - Activity_Raster : nD array of 1D arrays
                       spike trains of all neurons
    - time_end         : float 
                        time of the end of the previous burst
    - time_start       : float
                        time of the start of the burst to find ftf
    return:
    -------
            List of index of neuron that fire
    '''
    ret = []
    
    for i, spikes in enumerate(Activity_Raster):
        idxinf = np.where(spikes > np.mean([time_end, time_start]))[0]
        ftf    = np.where(spikes[idxinf] < time_start)[0]
        if len(ftf) != 0:
            ret.append(i)
    return ret 
    
def Ids_in_cluster(burst_NXY, epsilon, neighbors):
    '''
    DBSCAN Cluster detection function
    
    Parameters : 
    ------------
    - burst_NXY : list (neuron id , X position , Y position)
            sorted in increasing time of spiking 
    - epsilon : float 
            epsilon parameter in DBSCAN : range of clustering for a 
            node
    - neighbors : int
            minimum number of neighbors near a node to be in a cluster. 
    Return     : dict of dict
    ------------
            cluster ids as first keys, and neurons ids in the detected 
            cluster as second keys
    '''
    n_ids = np.array(burst_NXY)[:,0]
    X = np.delete(burst_NXY,0,1)
    
    #DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=neighbors , metric='euclidean', algorithm='ball_tree', leaf_size=5).fit(X)

    # Mask = True if in a detected core
    labels = db.labels_
    unique_labels = set(labels)
    print(unique_labels)
    in_core_mask = np.zeros_like(labels, dtype=bool)
    in_core_mask[db.core_sample_indices_] = True
    
    r_ids = dict()
    for k in unique_labels:
        if k != -1:  # -1 = non-labeled neuron (not in a cluster)
            k_core_mask  = (labels == k)
            cluster_size = len(n_ids[k_core_mask & in_core_mask])
            r_ids[k] = dict(zip(n_ids[k_core_mask & in_core_mask], [None]*cluster_size))
                  
    return r_ids    

def is_center(ftf_positions, all_positions, to_spike_or_not_to_spike):
    '''
    ftf_positions : coordinates of the first to fire neurons to test
    
    all_positions : all neurons coordinates
    
    to_spike_or_not_to_spike : in the same order as all_positions, wether the neuron spiked or not
    
    return complex hull as a shapely geometry if the neurons are firing
    in a small region of space, None otherwise
    '''
    ftf_mp = MultiPoint(ftf_positions)
    ftf_hull = ftf_mp.convex_hull
    
    active = 0
    non_active = 0
    
    for coord, spk in zip(all_positions,to_spike_or_not_to_spike):
        a = Point(coord)
        if a.intersects(ftf_hull):
            if spk:
                active += 1
            else:
                non_active += 1

    if active >= non_active:
        return ftf_hull
    else:
        return None

def angle( coords, tax, tin, culture_radius ):
    '''
    return the theta angle linearly extrapolated from tax and tin
    '''
    ret = tax
    
    ret += (( tin - tax ) / culture_radius) * np.sqrt(coords[0]**2 + coords[1]**2)
    
    return ret * np.pi / 180.
    
def polar_angle(coords):
    '''
    first is x coordinate
    second if y coordinate
    '''
    j = np.complex(0,1)
    ret = np.angle(coords[0] + j * coords[1])
    if ret > 0:
        return ret 
    else:
        return ret + 2*np.pi

def directions(a, t):
    '''
    Compute the polar directions given alpha
    '''
    if a == 0. or a == 2*np.pi:
        return [a + t, a + np.pi/2., a + np.pi - t, a + np.pi,
                a + t + np.pi, a + 3*np.pi/2., a + 2*np.pi - t]
    elif a < np.pi / 2.:
        return [a - t, a, a + t, a + np.pi / 2., a + np.pi - t,
                a + np.pi, a + t + np.pi, a + 3*np.pi/2.]
    elif a == np.pi/2.:
        return [a - t, a, a + t, a + np.pi / 2., a + np.pi - t,
                a + np.pi, a + t + np.pi]
    elif a < np.pi:
        return [a - np.pi/2., a - t, a, a + t, a + np.pi/2., a + np.pi - t,
                a + np.pi, a + t + np.pi]
    elif a == np.pi:
        return [a - np.pi + t, a - np.pi/2., a - t, a, 
                a + t, a + np.pi/2., a + np.pi - t]
    elif a < 3*np.pi/2.:
        print('i')
        return [a - np.pi - t, a - np.pi, a - np.pi + t, 
                a - np.pi/2., a - t, a, a + t, a + np.pi/2.]
    elif a == 3*np.pi/2.:
        return [a - np.pi - t, a - np.pi, a - np.pi + t, 
                a - np.pi/2., a - t, a, a + t]
    elif a < 2*np.pi:
        return [a - 3*np.pi/2., a - np.pi - t, a - np.pi, a - np.pi + t,
               a - np.pi/2., a - t, a, a + t]

def directions_toplot(a):
    '''
    which directions are towards the culture given alpha
    '''
    if a == 0. or a == 2*np.pi:
        return [2,3,4,5]
    elif a < np.pi / 2.:
        return [4,5,6,7]
    elif a == np.pi/2.:
        return [4,5,6,7]
    elif a < np.pi:
        return [5,6,7,0]
    elif a == np.pi:
        return [6,7,0,1]
    elif a < 3*np.pi/2.:
        return [0,1,2,3]
    elif a == 3*np.pi/2.:
        return [0,1,2,3]
    elif a < 2*np.pi:
        return [2,3,4,5]

def is_center(ftf_positions, all_positions, to_spike_or_not_to_spike):
    '''
    ftf_positions : coordinates of the first to fire neurons to test
    
    all_positions : all nueonrs coordinates
    
    to_spike_or_not_to_spike : in the same order as all_positions, wether the neuron spiked or not
    
    return True if the neurons are firing in a small region of space
    '''
    ftf_mp = MultiPoint(ftf_positions)
    ftf_hull = ftf_mp.convex_hull
    
    active = 0
    non_active = 0
    
    for coord, spk in zip(all_positions,to_spike_or_not_to_spike):
        a = Point(coord)
        if a.intersects(ftf_hull):
            if spk:
                active += 1
            else:
                non_active += 1

    if active >= non_active:
        return ftf_hull
    else:
        return None    
        
def span_culture(surfaces, culture):
    '''
    look if the surfaces (array of polygons) span the culture (polygon)
    '''
    r = []
    for s in surfaces:
        if s.area == culture.area:
            r.append(1)
    
    if np.sum(r) == len(surfaces):
        return False
    else:
        return True
    
def Plot_Ring_Analysis(activity, positions, hulls, bmin, 
                     step_buff, ttime, culture_radius, phi_lvl = 50, plot_rings = True, 
                     circular = False,
                     dir_segmentation = True, theta_max = 90, theta_min = 30, 
                     join_clusters = True, plot_neurons = False, kill = 10):
    '''
    REALLY Long function to plot the ring description of the activity.
    It can show the spatial rings, in specified directions and the mean phase
    evolution in time inside the rings.
    
    Params : 
    --------
    activity         : list N x number of spikes
    positions        : 2Nd array ; coordinates of all neurons
    hulls            : 1d array of shapely geometries - hulls of the 
                       first clusters
    bmin             : float ; minumum buffering size
    step_buff        : float ; buffering step 
    ttime            : 1d array ; times at which the phase will be computed
    phi_lvl          : int ; number of step in the phase representation
    circular         : bool ; wether to use circular cluster-hull
    dir_segmentation : bool ; wether to investigate different directions
    theta_max        : float ; maximum angle (rad) 
    theta_min        : float ; minimum angle (rad)
    join_clusters    : bool ; wether to gather all clusters
    plot_neurons     : bool ; wether to plot neurons with their phase as 
                       color at several time step
    kill             : number of time step avoided in 'ttime' to plot 
                       neurons
    
    Return :
    -------- 
    Figures
    '''
    #number of neurons
    num_neurons = len(activity)
    
    #Create the spherical culture
    culture = Point([0.,0.]).buffer(culture_radius)
    #concentric base that span the whole culture, used to build up rings
    spatial_base = []
    #concentric rings that span the whole culture
    rings        = []

    while None in hulls:
        hulls.remove(None)

    if len(hulls) == 0:
        print('no cluster of activity detected')
    #should the hulls be redefined as circular shape centered in the hull centroid
    elif circular == True:
        for i in range(len(hulls)):
            hulls[i] = hulls[i].centroid.buffer(bmin)
            
    #segmentation by direction or not
    if dir_segmentation == False:
        #do we join all clusters
        if join_clusters == True:
            #if there single neuron are detected as cluster, we buff them to a circular shape
            first_base = None
            for h in hulls:
                if h.area == 0.:
                    if first_base == None:
                        first_base = h.buffer(bmin)
                    else:
                        first_base = first_base.union(h.buffer(bmin))
                else:
                    if first_base == None:
                        first_base = h
                    else:
                        first_base = first_base.union(h)

            if first_base.area == 0.:
                spatial_base.append(first_base.buffer(bmin).intersection(culture))
            else:
                spatial_base.append(first_base.intersection(culture))

            #first ring is the hull
            rings = [spatial_base[0]]
            #while the culture is not entirely described by the spatial base we keep on increasing its size
            expension_param = [bmin]
            
            while (culture.area - spatial_base[-1].area) > 0.:
                bmin += step_buff
                spatial_base.append(first_base.buffer(bmin).intersection(culture))
                rings.append(spatial_base[-1].difference(spatial_base[-2]))
                expension_param.append(bmin)

            #caracteristic of the base    
            dim_base = len(rings)

            #should we plot it 
            if plot_rings == True:
                fig, ax = plt.subplots(figsize = (8,8))
                color   = it.cycle(['r','g','b'])
                for r, col in zip(rings, color):
                    #plot  = first_base.buffer(b).intersection(culture)
                    patch = PolygonPatch(r, fc = col, ec = col, alpha = .2)
                    ax.add_patch(patch)
                ax.set_xlim([-culture_radius,culture_radius])
                ax.set_ylim([-culture_radius,culture_radius])
                ax.set_ylabel('y ($\mu$m)')
                ax.set_xlabel('x ($\mu$m)')
                plt.show()

            #in which ring neurons fall
            where_neurons = np.zeros(num_neurons)
            #useless ringNB
            toremove = []
            #usefull ringNB
            kept     = []
            #expansion parameter use for plot the phase
            plot_exp = []
            
            for nb,poly in enumerate(rings):
                prepared_polygon = prep(poly)
                points   = [Point(positions[i]) for i in range(num_neurons)]
                data     = zip(points, range(0,num_neurons))
                N_filter = filter(lambda x: prepared_polygon.intersects(x[0]), data)
                N_filter = [N_filter[i][1] for i in range(len(N_filter))]
                where_neurons[N_filter] = nb

                if nb not in where_neurons:
                    toremove.append(nb)
                else: 
                    kept.append(nb)
                    plot_exp.append(expension_param[nb])

            #effective base dimension, as the usefull number of rings
            edim_base = dim_base - len(toremove)
            #for each rings, 
            maskring = [( where_neurons == ringNB) for ringNB in kept]
            #mean phase in each ring
            alpha_phase = []

            for i,t in enumerate(ttime):
                phases = Single_Neuron_Phase(activity, t, Kuramoto = False)
                color = [plt.cm.viridis(each) for each in phases]
                alpha_phase.append([np.mean(phases[mask]) for mask in maskring]) 
                #plot the phase at each time step 
                if plot_neurons == True and i%kill == 0:
                    fig, ax = plt.subplots(figsize = (7,7))
                    #for r, col in zip(rings, it.cycle(['r','g','b'])):
                       #plot  = first_base.buffer(b).intersection(culture)
                        #patch = PolygonPatch(r, fc = 'w', ec = col, alpha = .2)
                        #ax.set_ylabel('y ($\mu$m)')
                        #ax.set_xlabel('x ($\mu$m)')
                        #ax.add_patch(patch)
                    heatmap = ax.scatter(positions[:,0], positions[:,1], c = color,s = 6)
                    ax.set_aspect('equal')
                    ax.set_ylim([-culture_radius,culture_radius])
                    ax.set_xlim([-culture_radius,culture_radius])
                    ax.set_ylabel('y ($\mu$m)')
                    ax.set_xlabel('x ($\mu$m)')
                    fig.suptitle(str(t))
                    plt.show()

            fig_alpha, ax_alpha = plt.subplots(figsize = (8,8))
            alphas = ax_alpha.contourf(ttime, plot_exp, np.array(alpha_phase).T, 
                         levels = np.linspace(0.,1.,phi_lvl), cmap = 'Greys')
            cb = plt.colorbar(alphas)
            cb.set_label('mean phase')
            ax_alpha.set_ylabel('expension parameter')
            ax_alpha.set_xlabel('time (ms)')
            plt.show()
            
        else:
            
            first_hulls = len(hulls)
            centroids   = np.array([h.centroid.coords[:][0] for h in hulls])
            
            #single point detected as clusters
            for i in range(first_hulls):
                if hulls[i].area == 0.:
                    hulls[i] = hulls[i].buffer(bmin)
                    
            #spatial base and rings for all clusters
            spatial_base.append(hulls)
            rings.append(hulls)
            #the bases for all clusters span the whole culture ? (True == No)
            test = True
            expension_param = [bmin]
            while test:
                bmin  += step_buff
                expension_param.append(bmin)
    
                spb = []
                rg  = []

                for h in hulls:
                    spb.append(h.buffer(bmin).intersection(culture))
                    add = h.buffer(bmin).difference(h.buffer(bmin-step_buff)).intersection(culture)
                    if add.area != 0.:
                        rg.append(h.buffer(bmin).difference(h.buffer(bmin-step_buff)).intersection(culture).buffer(-1.))
                    else:
                        #if one cluster already span the whole culture, its ring is nothing == None
                        rg.append(np.NaN)
                        
                spatial_base.append(spb)
                rings.append(rg)
                test = span_culture(spb, culture)
                
            rings = np.array(rings)

            for cluster in range(first_hulls):
                dim_base = len(rings[:,cluster][np.where(rings[:,cluster] != np.NaN)[0]])

                if plot_rings == True and lcuster == 0:
                    fig, ax = plt.subplots(figsize = (8,8))
                    color   = it.cycle(['r','g','b'])
                    for rs, col in zip(rings[:,cluster], color):
                        #plot  = first_base.buffer(b).intersection(culture)
                        if rs is not None:
                            patch = PolygonPatch(rs, fc = col, ec = col, alpha = .2)
                            ax.add_patch(patch)
                    ax.set_ylabel('y ($\mu$m)')
                    ax.set_xlabel('x ($\mu$m)')
                    ax.set_xlim([-culture_radius,culture_radius])
                    ax.set_ylim([-culture_radius,culture_radius])
                    plt.show()

                #in which ring neurons fall
                where_neurons = np.zeros(num_neurons)
                #useless ringNB
                toremove = []
                #usefull ringNB
                kept     = []
                #expansion parameter use for plot the phase
                plot_exp = []
                
                for nb,poly in enumerate(rings[:,cluster]):
                    if poly is not None:
                        prepared_polygon = prep(poly)
                        points   = [Point(positions[i]) for i in range(num_neurons)]
                        data     = zip(points, range(0,num_neurons))
                        N_filter = filter(lambda x: prepared_polygon.intersects(x[0]), data)
                        N_filter = [N_filter[i][1] for i in range(len(N_filter))]
                        where_neurons[N_filter] = nb

                    if nb not in where_neurons:
                        toremove.append(nb)
                    else: 
                        kept.append(nb)
                        plot_exp.append(expension_param[nb])

                #effective base dimension, as the usefull number of rings
                edim_base = dim_base - len(toremove)
                #for each rings, 
                maskring = [( where_neurons == ringNB) for ringNB in kept]
                #mean phase in each ring
                alpha_phase = []
                
                for i,t in enumerate(ttime):
                    phases = Single_Neuron_Phase(activity, t, Kuramoto = False)
                    color = [plt.cm.viridis(each) for each in phases]
                    alpha_phase.append([np.mean(phases[mask]) for mask in maskring]) 
                    #plot the phase at each time step 
                    if plot_neurons == True and i%kill == 0:
                        fig, ax = plt.subplots(figsize = (7,7))
                        #for r, col in zip(rings, it.cycle(['r','g','b'])):
                           #plot  = first_base.buffer(b).intersection(culture)
                            #patch = PolygonPatch(r, fc = 'w', ec = col, alpha = .2)
                            #ax.set_ylabel('y ($\mu$m)')
                            #ax.set_xlabel('x ($\mu$m)')
                            #ax.add_patch(patch)
                        heatmap = ax.scatter(positions[:,0], positions[:,1], c = color,s = 6)
                        ax.set_aspect('equal')
                        ax.set_ylim([-culture_radius,culture_radius])
                        ax.set_xlim([-culture_radius,culture_radius])
                        ax.set_ylabel('y ($\mu$m)')
                        ax.set_xlabel('x ($\mu$m)')
                        fig.suptitle(str(t))
                        plt.show()

                fig_alpha, ax_alpha = plt.subplots(figsize = (8,8))
                alphas = ax_alpha.contourf(ttime, plot_exp, np.array(alpha_phase).T, 
                             levels = np.linspace(0.,1.,phi_lvl), cmap = 'Greys')
                cb = plt.colorbar(alphas)
                cb.set_label('mean phase')
                ax_alpha.set_ylabel('expension parameter')
                ax_alpha.set_xlabel('time (ms)')
                plt.show()
    
    
    else:
        if join_clusters == True:
            print('you cannot join clusters if you want to analyse directionnality')
            
        first_hulls = len(hulls)
        centroids   = np.array([h.centroid.coords[:][0] for h in hulls])

        for i in range(first_hulls):
            if hulls[i].area == 0.:
                hulls[i] = hulls[i].buffer(bmin)

        #spatial base and rings for all clusters
        spatial_base.append(hulls)
        rings.append(hulls)
        #the bases for all clusters span the whole culture ? (True == No)
        test = True
        expension_param = [bmin]
        
        while test:
            bmin  += step_buff
            expension_param.append(bmin)
            
            spb = []
            rg  = []
            
            for h in hulls:
                spb.append(h.buffer(bmin).intersection(culture))
                add = h.buffer(bmin).difference(h.buffer(bmin-step_buff)).intersection(culture)
                if add.area != 0.:
                    #negative buffer to manage the intersection with the lines after
                    rg.append(h.buffer(bmin).difference(h.buffer(bmin-step_buff)).intersection(culture).buffer(-1.))
                else:
                    rg.append(None)
            spatial_base.append(spb)
            rings.append(rg)
            test = span_culture(spb, culture)

        rings = np.array(rings)
        for cluster in range(first_hulls):
            dim_base = len(rings)
            
            # the line line1 crossing the centroid and the origin is the bisector of two
            # lines making an angle theta with line1 ; theta depends on the centroid distance with the origin
            theta = angle(centroids[cluster], theta_max, theta_min, culture_radius)
            # polar coordinate of the centroids
            alpha = polar_angle(centroids[cluster])
            
            # Theta and alpha define different lines, that define different region of space (directions)
            polar_directions = directions(alpha, theta)
            # slope of the lines
            print(alpha, theta, polar_directions)
            beta = alpha - np.pi - theta
            if beta > np.pi /2.:
                    beta = -np.pi + beta
            
            # centroid / origin
            line1 = LineString([( culture_radius,  culture_radius*centroids[cluster][1] / centroids[cluster][0]),
                                (-culture_radius, -culture_radius*centroids[cluster][1] / centroids[cluster][0])])   
            # centroid / -theta with line1
            line2 = LineString([( culture_radius, ( culture_radius - centroids[cluster][0]) * np.tan(beta) + centroids[cluster][1] ),
                                (-culture_radius, (-culture_radius - centroids[cluster][0]) * np.tan(beta) + centroids[cluster][1] )])
            # centroid / + theta with line2
            line3 = LineString([( culture_radius, centroids[cluster][1] + (centroids[cluster][0] - culture_radius)*centroids[cluster][0] / centroids[cluster][1]),
                                (-culture_radius, centroids[cluster][1] + (centroids[cluster][0] + culture_radius)*centroids[cluster][0] / centroids[cluster][1])])

            beta  = -np.pi + alpha + theta
            # centroid / perpendicular to line1
            line4 = LineString([( culture_radius, ( culture_radius - centroids[cluster][0]) * np.tan(beta)  + centroids[cluster][1] ),
                                (-culture_radius, (-culture_radius - centroids[cluster][0]) * np.tan(beta)  + centroids[cluster][1] )])

            lines = line1.union(line2).union(line3).union(line4).buffer(2.)
            
            # for each rings, there are several directions
            multiring = [[rings[:,cluster][0]]]
            # Buff +1 because unbuffed before
            efr = (rings[:,cluster] != None)
            multiring.extend([list(p.difference(lines).buffer(1.).geoms) for p in rings[:,cluster][efr]])
            print(len(multiring), dim_base)
            #it can be ordered
            #for bbprout in range(dim_base):
                #multiring[bbprout].sort(key = lambda x : polar_angle(np.array(x.centroid.coords[:][0]) - np.array(centroids[cluster])))

            #to which directions, the rings parts belong to (first ring is the hull, direction -1)
            angular_direction = [[-1]]
            for prout in multiring[1:]:
                agle_dir = []
                agle_dir.append([np.digitize(polar_angle(np.array(surface.representative_point().coords[:][0]) - centroids[cluster]), polar_directions) for surface in prout])
                agle_dir = np.array(agle_dir)
                # 8 and 0 are the same region
                m = (agle_dir == 8)
                agle_dir[m] = 0
                angular_direction.extend(agle_dir)

            if plot_rings == True:
                fig, ax = plt.subplots(figsize = (8,8))
                color   = it.cycle(['r','g','b'])
                phapha  = it.cycle([0.1,0.25,0.45,0.65])
                for mr, col in zip(range(len(angular_direction)), color):
                    for ar in range(len(angular_direction[mr])):
                        pha = angular_direction[mr][ar]*0.1+0.2
                        patch = PolygonPatch(multiring[mr][ar], fc = col, ec = col, alpha = pha)
                        ax.add_patch(patch)
                ax.set_xlim([-culture_radius,culture_radius])
                ax.set_ylim([-culture_radius,culture_radius])
                ax.set_ylabel('y ($\mu$m)')
                ax.set_xlabel('x ($\mu$m)')
                plt.show()
            
            #in which ring neurons fall
            where_neurons = np.zeros(num_neurons)
            #useless ringNB
            toremove = []
            #usefull ringNB
            kept     = []
            #expansion parameter use for plot the phase
            plot_exp = []
            
            for nb,poly in enumerate(rings[:,cluster]):
                if poly is not None:
                    prepared_polygon = prep(poly)
                    points   = [Point(positions[i]) for i in range(num_neurons)]
                    data     = zip(points, range(0,num_neurons))
                    N_filter = filter(lambda x: prepared_polygon.intersects(x[0]), data)
                    N_filter = [N_filter[i][1] for i in range(len(N_filter))]
                    where_neurons[N_filter] = nb

                if nb not in where_neurons:
                    toremove.append(nb)
                else:
                    kept.append(nb)
                    plot_exp.append(expension_param[nb])

            #effective base dimension, as the usefull number of rings
            edim_base = dim_base - len(toremove)
            #for each rings, 
            maskring = [( where_neurons == ringNB) for ringNB in kept]
            #mean phase in each ring
            alpha_phase = []
            
            # in which directions are neurons
            neurons_direction = np.array([np.digitize(polar_angle(pepe - centroids[cluster]),
                                        polar_directions) for pepe in positions])

            for ddir in directions_toplot(alpha):
                
                fig, ax = plt.subplots(1,2,figsize = (16,8))
                
                color   = it.cycle(['r','g','b'])
                phapha  = it.cycle([0.1,0.25,0.45,0.65])
                for mr, col in zip(range(len(angular_direction)), color):
                    for ar in range(len(angular_direction[mr])):
                        if angular_direction[mr][ar] == ddir:
                            patch = PolygonPatch(multiring[mr][ar], fc = col, ec = col, alpha = pha)
                            ax[0].add_patch(patch)
                ax[0].set_xlim([-culture_radius,culture_radius])
                ax[0].set_ylabel('y ($\mu$m)')
                ax[0].set_xlabel('x ($\mu$m)')
                ax[0].set_ylim([-culture_radius,culture_radius])
                
                maskdir = [( neurons_direction[maskr] == ddir) for maskr in maskring]

                alpha_phase = []
                for t in ttime:
                    phases = Single_Neuron_Phase(activity, t, Kuramoto = False)
                    color = [plt.cm.viridis(each) for each in phases]
                    alpha_phase.append([np.mean(phases[m1][m2]) for m1, m2 in zip(maskring,maskdir)])

                alphas = ax[1].contourf(ttime, plot_exp, np.array(alpha_phase).T, 
                             levels = np.linspace(0.,1.,100), cmap = 'Greys')
                ax[1].set_ylabel('expension parameter')
                ax[1].set_xlabel('time (ms)')
                cb = plt.colorbar(alphas)
                cb.set_label('mean phase')
                plt.show()


    
if __name__ == '__main__':
    spk_file = '/home/mallory/Documents/These/javier-avril-CR/Simulations2/EDRNetworks/RSneurons/2000EDR_100_lambda700.0_weight10.0.txt'
    # ~ spk_file = '/home/mallory/Documents/These/September2018/RP report/Simu2/N1000_AS2.txt'
    return_list = []
    with open(spk_file, "r") as fileobject:
        for i, line in enumerate(fileobject):
            if not line.startswith('#'):
                lst = line.rstrip('\n').split(' ')
                return_list.append([int(lst[0]),float(lst[1]),float(lst[2]),float(lst[3])])
    NTXY     = np.array(sorted(return_list, key = lambda x:x[1]))
    neurons  = [i for i in set(NTXY[:,0])]
    activity = []
    print(len(neurons))
    senders = NTXY[:,0]
    times   = NTXY[:,1]
    pos     = NTXY[...,2:]
    tmin , tmax = np.min(times) , np.max(times)
    dt = 10.
    num_neurons = len(set(NTXY[:,0]))
    positions   = []

    for nn in neurons:
        nspk = np.where(senders == nn)[0]
        tspk = times[nspk]
        activity.append(tspk)
        positions.append(pos[nspk[0]])
    
    positions = np.array(positions)
    axes      = plt.subplot(111)
    taxes    = axes.twinx()

    tb , ext_idx, t_ext = Burst_times(activity, tmin, tmax, dt)

    '''
    Paramter for ring analysis
    '''
    culture_radius = 2000.
    ux = np.unique(positions[:, 0])
    uy = np.unique(positions[:, 1])
    xx        = np.sort(ux)
    yy        = np.sort(uy)

    # For the heatmaps
    step_ht   = 5.
    xstep     = max(0.5*np.min(np.diff(xx)), step_ht)
    ystep     = max(0.5*np.min(np.diff(yy)), step_ht)

    eps = np.max(xx)/4.
    burst = 2
    
    # culture
    culture = Point([0.,0.]).buffer(culture_radius)

    step_buff    = 75
    ttime = np.linspace(t_ext[burst][0]-150, t_ext[burst][1] + 100, 150)
    bmin = np.min([xstep,ystep])

    theta_max, theta_min = 90 , 20
    
    '''
    Analysis
    '''
    
    ftf     = First_To_Fire(activity, t_ext[burst-1][1], t_ext[burst][0])
    pos_ftf = positions[ftf]
    NXY     = zip(ftf,pos_ftf[:,0],pos_ftf[:,1])
    clt     = Ids_in_cluster(NXY, eps, 3)
    hulls = []
    
    for cc in clt.keys():
        to_p = np.array(clt[cc].keys()).astype(int)
        spiked = np.zeros(num_neurons)
        spiked[ftf] = True
        hulls.append(is_center(positions[to_p], positions, spiked))
    
    Plot_Ring_Analysis(num_neurons, positions, hulls, bmin, step_buff, ttime,
                     phi_lvl = 50, plot_rings = True, circular = False,
                     dir_segmentation = False, theta_max = 90, theta_min = 30, 
                     join_clusters = True , plot_neurons = False, kill = 15)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
