# -*- coding: utf-8 -*-

__author__ = 'Xiaowen Fang'

from scipy.spatial import Delaunay
import numpy as np
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
import math
#import matplotlib.pyplot as plt
from descartes import PolygonPatch
#from matplotlib  import cm
import sys
from multiprocessing import Process, Manager, Array, Pool, Lock
import multiprocessing
from time import clock

#This is also from http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        try:
            return geometry.MultiPoint(list(points)).convex_hull
        except BaseException as e:
            print('F***! They are on the same line!', points)
            print('numpy.float64 is ', points)
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[[i,j]])
    coords = np.array(points)
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        try:    #could have ZeroDivisionError and ValueError
            # Area of triangle by Heron's formula
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            circum_r = a*b*c/(4.0*area)
            # Here's the radius filter.
            if circum_r < 1.0/alpha:
                add_edge(edges, edge_points, coords, ia, ib)
                add_edge(edges, edge_points, coords, ib, ic)
                add_edge(edges, edge_points, coords, ic, ia)
        except ZeroDivisionError as e:
            print("So points are? ", ia, ib, ic)
        except ValueError as e:
            print('a is %f, b is %f, c is %f, s is %f' %(a,b,c,s))
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    #return the union of polygons in this alpha shape
    return cascaded_union(triangles), edge_points


#The main basin finder algorithm
def Basin_Finder(S, c, Omega, delta2, colormap, job_list, basins, jump):
    """
    The main algorithm to find basins (and saddles, not implement yet)

    Parameters
    ----------
    S : set of grid points under certain energy level (c).
    c : highest energy level
    Omega : list of basins and saddles to be stored
    delta2 : step length c decreases
    basins : ???
    """
    
    print('now you are in basin_finder')
    if c < -6368:       #Don't need to jump below certain level. Because there are more valuble basins in low energy level, we want to explore them very carefully.
        jump = False
    S_origin = np.copy(S)   # Just copy the points
    #Jump test
    c = (c - delta2 * 50) if jump else (c - delta2)     # If jump, then decrease the energy level with large step size. Otherwise just decrease with normal step size.
    print('This is C! ', c)
    s_under = clock()   # Timing. You can comment it out.
    S = np.array([p for p in S if p[3] < c])      #get points below energy level c. Could be faster? Optimize this!
    f_under = clock()   #Timing
    print('compting points under c takes : ', f_under - s_under)
    print('# of points after decending energy : ', len(S))
    
    if (len(S) < 100 and c > -6368) or len(S) < 3:    #Just want to end it earlier if less than 100 points in a high energy level
        print('Unimportant basin. I am returning', len(S), c)  #Debug message
        return    
    # construct alpha shape for points under energy level c
    try:
        s_ashape = clock()  #Timing
        bounds, edge_points = alpha_shape(np.delete(S, [2,3], 1), alpha=1/0.15)    #construct alpha shape of points under energy level c. Note: S is [pc1, pc2, energy, index], so we want to delete last two columns to make it [pc1, pc2]
        f_ashape = clock()  #Timing
        print('computing alpha shape takes : ', f_ashape - s_ashape) # print time used
    except BaseException as e:
        print("What's wrong with S : ", S)
        return    
            
    k = 1 if not isinstance(bounds, geometry.multipolygon.MultiPolygon) else len(bounds)    #weather it's a single polygon or multiple polygons in the alpha shape

    #no basin splitting detected
    print('# of basin in this iteration is ', k)   #Debugging message
    if k == 1:  
        print('Still no basin splitting. And continue the previous action!')
        print('# of points left is : ', len(S))
        
        job_list.put((k,S,c,jump))     #no basin splitting. So still follow the jump condition as before
        
        print('We put you in job_list, continue the same action')
        
    #basin splitting detected
    else:
        if jump:    # It jumps here. So we need to revert back
            print('Ohhh! We need to revert back!')
            c += 49 * delta2
            s_under = clock()
            S = [p for p in S_origin if p[3] < c]      #points with energy less than c. Could be faster? Optimize this!
            f_under = clock()
            print('compting points under c takes : ', f_under - s_under)
            print('# points left after reverting back should be greater than before : ', len(S))
            try:    # construct alpha shape
                s_ashape = clock()
                bounds, edge_points = alpha_shape(np.delete(S, [2,3], 1), alpha=1/0.15)    #alpha shape
                f_ashape = clock()
                print('computing alpha shape takes : ', f_ashape - s_ashape)   
            except BaseException as e:
                print("What's wrong with S : ", S)
                return  
            k = 1 if not isinstance(bounds, geometry.multipolygon.MultiPolygon) else len(bounds)    #weather it's a single polygon or multiple polygons in the alpha shape
            if k == 1:
                print('Oh! Did not find basin split after revert back! So we need to decend carefully to find basin')
                print('So # of points is : ', len(S))
                job_list.put((k,S,c,False))  #decending carefully until you find me!
                print('We put you in job_list, decending carefully')
                #patch = PolygonPatch(bounds, fc=colormap.to_rgba(c), ec='#000000', fill=True, zorder=-1)  #Here, map c value to color code using color map
                #basins.append(patch)
            else:   #Ohhh! You find me!
                print('Lucky! you find me right after revert back!')
                old_S = S[:]
                S = []
                # For each detected basin, find the points inside the basin. And do basin_finder on those points
                for i in range(k):
                    points_in_basin = [p.tolist() for p in old_S if geometry.Point([p[0], p[1]]).within(bounds[i])]
                    #print('points in basin are :', points_in_basin)
                    job_list.put_nowait((k,points_in_basin,c, True))    # k is meaningless. Should remove k
                    print('Put you in job_list. You are so lucky! ', i)
                    old_S = [p for p in old_S if p.tolist() not in points_in_basin]     # remove points already put in job_list
                    
                    #print('We put stuff into jobList! VeryGood! So are you empty now? ', job_list.empty())

        else:   #no need to revert
            #get points in different basins. Stored in S   
            print('Find basins and no need to revert back')
            old_S = S[:]
            S = []
            for i in range(k):
                print('!!!! # of K is ', k)
                points_in_basin = [p.tolist() for p in old_S if geometry.Point([p[0], p[1]]).within(bounds[i])]
                job_list.put((k,points_in_basin,c,True))
                old_S = [p for p in old_S if p.tolist() not in points_in_basin]
                    #print('How many points in each basin? ', len(x))
                    #print('We got points in a basin! Very good!')
                    #S.append(points_in_basin)     #append the points in a polygon
                print('Put you in job_list. Finally find you, basin!  ',  i)
                #print('We put stuff into jobList! VeryGood! So are you empty now? ', job_list.empty())
        #get saddles between neighbor basins
        #Could speed up by this way https://gis.stackexchange.com/questions/226085/fast-minimum-distance-between-two-sets-of-polygons/226143#226143

        #Pseudo code below. To be done
    
    #store the basin figure so we can plot it later.
    if k > 1:
        for i in range(k):
            patch = PolygonPatch(bounds[i], fc=colormap.to_rgba(c), ec='#000000', fill=True, zorder=-1)  #Here, map c value to color code using color map
            basins.append(patch)       
            print('We add a patch/basin!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    return

