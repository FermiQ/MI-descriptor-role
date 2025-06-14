# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:36:19 2021

@author: Richard Couperthwaite
"""

import numpy as np
import sys
from pyDOE import lhs
from kmedoids import kMedoids
from scipy.spatial import distance_matrix
from acquisitionFunc import expected_improvement, knowledge_gradient, thompson_sampling, upper_conf_bound, probability_improvement
import pandas as pd
from pickle import load

def k_medoids(sample, num_clusters):
    # clusters the samples into the number of clusters (num_clusters) according 
    # to the K-Medoids clustering algorithm and returns the medoids and the 
    # samples that belong to each cluster
    D = distance_matrix(sample, sample)
    M, C = kMedoids(D, num_clusters)
    return M, C  

def call_model(param):
    # this function is used to call any model given in the dictionary of
    # parameters (param)
    output = param["Model"](param["Input Values"])
    return param, output

def cartesian(*arrays):
    # combines a set of arrays (one per dimension) so that all combinations of
    # all the arrays are in a single matrix with columns for each dimension
    mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
    dim = len(mesh)  # number of dimensions
    elements = mesh[0].size  # number of elements, any index will do
    flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
    reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
    return reshape

def sampleDesignSpace(ndim, nsamples, sampleScheme):
    # This function provides three approaches to sampling of the design space
    # firstly, Latin hypercube sampling (LHS)
    # secondly, a grid based appraoch (Grid)
    # and the final approach allows for custom sampling of specific values
    # in this last approach, any additional samples required are found by 
    # Latin Hypercube sampling
    if sampleScheme == "LHS":
        x = lhs(ndim, nsamples)
    if sampleScheme == "Grid":
        for jjj in range(nsamples-1):
            input_arr = np.linspace(0,1,jjj+1)
            all_arr = []
            for ii in range(ndim):
                all_arr.append(input_arr)
            x = cartesian(*all_arr)
            if x.shape[0] >= nsamples:
                return x
    if sampleScheme == "Custom":
        dfInputs = pd.read_csv("data/possibleInputs.csv", index_col=0)
        if dfInputs.shape[0] > nsamples:
            x = dfInputs.sample(n=nsamples)
        else:
            x_other = pd.DataFrame(lhs(ndim, nsamples-dfInputs.shape[0]),columns=dfInputs.columns)
            x = pd.concat((dfInputs, x_other))         
    return np.array(x)

def apply_constraints(samples, ndim, resolution=[], A=[], b=[], Aeq=[], beq=[], 
                      lb=[], ub=[], func=[], sampleScheme="LHS", opt_sample_size=True,
                      evaluatedPoints=[]):
    # This function handles the sampling of the design space and the application 
    # of the constraints to ensure that any points sampled satisfy the constratints
    sampleSelection = True
    constraints = np.zeros((5))
    if A != []:
        constraints[0] = 1
    if Aeq != []:
        constraints[1] = 1
    if lb != []:
        constraints[2] = 1
    if ub != []:
        constraints[3] = 1
    if func != []:
        if (type(func) == list):
            constraints[4] = len(func)
        else:
            constraints[4] = 1
    lhs_samples = samples
    
    x_largest = []
    largest_set = 0
    
    while sampleSelection:
        try:
            x = sampleDesignSpace(ndim, lhs_samples, sampleScheme)
        except:
            x = sampleDesignSpace(ndim, lhs_samples, "LHS")
        if resolution != []:
            x = np.round(x, decimals=resolution)
        constr_check = np.zeros((x.shape[0], ndim))
        
        # Apply inequality constraints
        if (A != []) and (b != []) and (len(A) == ndim):
            A_tile = np.tile(np.array(A), (x.shape[0],1))
            constr_check += A_tile*x <= b
            constraints[0] = 0

        # Apply equality constraints
        if (Aeq != []) and (beq != []):
            Aeq_tile = np.tile(np.array(Aeq), (x.shape[0],1))
            constr_check += Aeq_tile*x <= beq
            constraints[1] = 0
        
        # Apply Lower and Upper Bounds
        if (lb != []) and (len(lb) == ndim):
            lb_tile = np.tile(np.array(lb).reshape((1,ndim)), (x.shape[0],1))
            constr_check += x < lb_tile
            constraints[2] = 0
        if (ub != []) and (len(ub) == ndim):
            ub_tile = np.tile(np.array(ub).reshape((1,ndim)), (x.shape[0],1))
            constr_check += x > ub_tile
            constraints[3] = 0
        
        constr_check = np.sum(constr_check, axis=1)
        
        # Apply custom function constraints
        if (type(func) == list) and (func != []):
            for ii in range(len(func)):
                try:
                    constr_check += func[ii](x)
                    constraints[4] -= 1
                except:
                    pass
        elif (type(func) != list) and (func != []):
            try:
                constr_check += func(x)    
                constraints[4] = 0
            except:
                pass
        
        # Duplicate Check: if a particular sample has been queried from all models
        # it needs to be removed from the potential samples. This won't stop duplicates
        # getting in since we can't exclude a point till it has been evaluated from all models
        if evaluatedPoints != []:
            all_test = np.zeros_like(constr_check)
            for evalPoints in evaluatedPoints:
                res = (x[:, None] == evalPoints).all(-1).any(-1)
                all_test += res
            all_test[np.where(all_test<len(evaluatedPoints))] = 0
            
            constr_check += all_test
            
        index = np.where(constr_check == 0)[0]
        
        # If it is chosen to optimize the sample size, the loop is continued to 
        # ensure that as close to the required number of samples are acquired
        if opt_sample_size:
            if index.shape[0] >= samples:
                x = x[index[0:samples],:]
                sampleSelection = False
                if np.sum(constraints) != 0:
                    const_satisfied = False
                else:
                    const_satisfied = True
            else:
                if len(index) > largest_set:
                    largest_set = len(index)
                    x_largest = x[index,:]
                if lhs_samples/samples < ndim*2000:
                    lhs_samples += samples*100
                else:
                    x = x_largest
                    sampleSelection = False
                    const_satisfied = False
        # if the choice is to not optimize, the samples that pass all constraints
        # will be returned. This can lead to less samples than specified.
        else:
            x = x[index,:]
            sampleSelection = False
            const_satisfied = True
    return x, const_satisfied

def calculate_KG(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        knowledge gradient of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum knowledge gradient value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum knowledge gradient and the maximum of the fused model

    """
    (finish, model_temp, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = param
    # Initialize the output       
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Use the fused model to obtain the mean and variance at all test points
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    try:
        index_max = index_max_[0]
    except IndexError:
        index_max = index_max_
    # Add the maximum of the fused model to the output    
    output[0] = np.max(fused_mean)
    # Calculate the knowledge gradient for all test point
    nu_star, x_star, NU = knowledge_gradient(true_sample_count, 
                                              0.1, 
                                              fused_mean, 
                                              fused_var)
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # Add the actual input values for the maximum of the fused model
    if len(x_test.shape) > 1:
        for ii in range(x_test.shape[1]):
            output.append(x_test[index_max,ii])
    else:
        output.append(x_test[index_max])
    # Add the input values for the maximum knowledge gradient value
    for i in range(x_test.shape[1]):
        output.append(x_test[x_star,i])
    # Return the results
    return param, output

def calculate_EI(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        expected improvement of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum expected improvement and the maximum of the fused model

    """
    (finish, model_temp, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = param
    # Initialize the output       
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Use the fused model to obtain the mean and variance at all test points
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    fused_var = np.diag(fused_var)
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    try:
        index_max = index_max_[0]
    except IndexError:
        index_max = index_max_
    # Add the maximum of the fused model to the output  
    output[0] = np.max(fused_mean)
    # Calculate the expected improvement for all test point
    nu_star, x_star, NU = expected_improvement(curr_max, 
                                               0.01, 
                                               fused_mean, 
                                               fused_var)
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # Add the actual input values for the maximum of the fused model
    if len(x_test.shape) > 1:
        for ii in range(x_test.shape[1]):
            output.append(x_test[index_max,ii])
    else:
        output.append(x_test[index_max])
    # Add the input values for the maximum knowledge gradient value
    for i in range(x_test.shape[1]):
        output.append(x_test[x_star,i])
    # Return the results
    return param, output   



def calculate_TS(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        Thompson Sampling of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum Thompson Sampling Result and the maximum of the fused model

    """
    (finish, model_temp, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = param
    # Initialize the output       
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Use the fused model to obtain the mean and variance at all test points
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    fused_var = np.diag(fused_var)
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    try:
        index_max = index_max_[0]
    except IndexError:
        index_max = index_max_
    # Add the maximum of the fused model to the output  
    output[0] = np.max(fused_mean)
    # Calculate the expected improvement for all test point
    nu_star, x_star, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # Add the actual input values for the maximum of the fused model
    if len(x_test.shape) > 1:
        for ii in range(x_test.shape[1]):
            output.append(x_test[index_max,ii])
    else:
        output.append(x_test[index_max])
    # Add the input values for the maximum knowledge gradient value
    for i in range(x_test.shape[1]):
        output.append(x_test[x_star,i])
    # Return the results
    return param, output   


def calculate_PI(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        Probability of Improvement of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum Probability of Improvement and the maximum of the fused model

    """
    (finish, model_temp, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = param
    # Initialize the output       
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Use the fused model to obtain the mean and variance at all test points
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    fused_var = np.diag(fused_var)
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    try:
        index_max = index_max_[0]
    except IndexError:
        index_max = index_max_
    # Add the maximum of the fused model to the output  
    output[0] = np.max(fused_mean)
    # Calculate the expected improvement for all test point
    nu_star, x_star, NU = probability_improvement(curr_max, 0.01, fused_mean, np.sqrt(fused_var))
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # Add the actual input values for the maximum of the fused model
    if len(x_test.shape) > 1:
        for ii in range(x_test.shape[1]):
            output.append(x_test[index_max,ii])
    else:
        output.append(x_test[index_max])
    # Add the input values for the maximum knowledge gradient value
    for i in range(x_test.shape[1]):
        output.append(x_test[x_star,i])
    # Return the results
    return param, output   



def calculate_UCB(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        Upper Confidence Bound of a fused model constructed out of a reification 
        model object.

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to both 
        the maximum Upper Confidence Bound and the maximum of the fused model

    """
    (iteration, model_temp, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = param
    # Initialize the output       
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Use the fused model to obtain the mean and variance at all test points
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    fused_var = np.diag(fused_var)
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    try:
        index_max = index_max_[0]
    except IndexError:
        index_max = index_max_
    # Add the maximum of the fused model to the output  
    output[0] = np.max(fused_mean)
    # Calculate the expected improvement for all test point
    
    beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
    
    kt = np.sqrt(0.2 * beta)
    
    nu_star, x_star, NU = upper_conf_bound(kt, fused_mean, np.sqrt(fused_var))
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = nu_star/cost[jj]
    output[2] = x_star
    # Add the actual input values for the maximum of the fused model
    if len(x_test.shape) > 1:
        for ii in range(x_test.shape[1]):
            output.append(x_test[index_max,ii])
    else:
        output.append(x_test[index_max])
    # Add the input values for the maximum knowledge gradient value
    for i in range(x_test.shape[1]):
        output.append(x_test[x_star,i])
    # Return the results
    return param, output   








def fused_calculate(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        maximum of a fused model generated from a reification object.

    Returns
    -------
    results : list
        The output from the module contains the maximum of the fused model as 
        well as the index of the test point that corresponds with that value.

    """
    (finish, model_temp, x_fused, fused_model_HP, \
         kernel, x_test, curr_max, xi, sampleOpt) = param
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    if sampleOpt == "TS":
        """
        Thompson sampling approach
        This approach uses the uncertainty, but is quite significantly slower
        """
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
        output = [nu_star, x_star]
    elif sampleOpt == "EI":
        """
        Expected Improvement approach
        """
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = expected_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output = [nu_star, x_star]
    elif sampleOpt == "PI":
        """
        Probability of Improvement approach
        """
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = probability_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output = [nu_star, x_star]
    elif sampleOpt == "UCB":
        """
        Upper Confidence Bound approach
        """
        beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
        kt = np.sqrt(0.2 * beta)
        fused_var = np.diag(fused_var)
        nu_star, x_star, NU = upper_conf_bound(kt, 
                                                fused_mean, 
                                                fused_var)
        output = [nu_star, x_star]
    elif sampleOpt == "KG":
        """
        Knowledge Gradient approach
        """
        nu_star, x_star, NU = knowledge_gradient(x_test.shape[0], 
                                                  0.1, 
                                                  fused_mean, 
                                                  fused_var)
        output = [nu_star, x_star]
    elif sampleOpt == "Hedge":
        output = []
        nu, x, NU = knowledge_gradient(x_test.shape[0], 
                                                  0.1, 
                                                  fused_mean, 
                                                  fused_var)
        output.append([nu, x])
        fused_var = np.diag(fused_var)
        nu, x, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
        output.append([nu, x])
        nu, x, NU = expected_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output.append([nu, x])
        nu = np.max(fused_mean)
        try:
            x = int(np.nonzero(fused_mean == nu)[0])
        except TypeError:
            x = int(np.nonzero(fused_mean == nu)[0][0])
        output.append([nu, x])
        nu, x, NU = probability_improvement(curr_max, 
                                                    xi, 
                                                    fused_mean, 
                                                    fused_var)
        output.append([nu, x])
        beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
        kt = np.sqrt(0.2 * beta)
        nu, x, NU = upper_conf_bound(kt, 
                                    fused_mean, 
                                    fused_var)
        output.append([nu, x])
    else:
        """
        Greedy Sampling Approach
        """
        # Find the maximum of the fused model
        nu_star = np.max(fused_mean)
        try:
            x_star = int(np.nonzero(fused_mean == nu_star)[0])
        except TypeError:
            x_star = int(np.nonzero(fused_mean == nu_star)[0][0])
        output = [nu_star, x_star]
        
    # return the maximum value and the index of the test point that corresponds
    # with the maximum value
    return param, (output, x_test.shape[0])

def calculate_GPHedge(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        values from all acquisition functions for use in the GP Hedge portfolio
        optimization appraoch.

    Returns
    -------
    results : list
        The output from the module contains the maximum of all acquisition functions
        and the x values associated with these points.

    """
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (iteration, model_data, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = data[param[0]]
    # Initialize the output  
    output = [[0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm],
              [0,[],[],jj,kk,mm]]
    # Create the fused model
    model_temp.update_GP(*model_data)
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Use the fused model to obtain the mean and variance at all test points
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    try:
        index_max = index_max_[0]
    except IndexError:
        index_max = index_max_
    # Add the maximum of the fused model to the output  
    output[0][0] = np.max(fused_mean)
    output[1][0] = np.max(fused_mean)
    output[2][0] = np.max(fused_mean)
    output[3][0] = np.max(fused_mean)
    output[4][0] = np.max(fused_mean)
    output[5][0] = np.max(fused_mean)

    nu_star = []
    x_star = []
    
    
    #################
    ################
    # Need to convert this next section to run in parallel to reduce the time
    
    """
    Knowledge Gradient approach
    """
    nu_star, x_star, NU = knowledge_gradient(x_test.shape[0], 
                                              0.1, 
                                              fused_mean, 
                                              fused_var)
    output[0][1] = nu_star/cost[jj]
    output[0][2] = x_star

    """
    Thompson sampling approach
    This approach uses the uncertainty, but is quite significantly slower
    """
    fused_var = np.diag(fused_var)
    nu_star, x_star, NU = thompson_sampling(fused_mean, np.sqrt(fused_var))
    output[1][1] = nu_star/cost[jj]
    output[1][2] = x_star
    
    """
    Expected Improvement approach
    """
    nu_star, x_star, NU = expected_improvement(curr_max, 
                                        0.01, 
                                        fused_mean, 
                                        fused_var)
    output[2][1] = nu_star/cost[jj]
    output[2][2] = x_star
   
    """
    Greedy Sampling Approach
    """
    # Find the maximum of the fused model
    nu_star = np.max(fused_mean)
    try:
        x_star = int(np.nonzero(fused_mean == nu_star)[0])
    except TypeError:
        x_star = int(np.nonzero(fused_mean == nu_star)[0][0])
    output[3][1] = nu_star/cost[jj]
    output[3][2] = x_star
    
    """
    Probability of Improvement Approach
    """
    # Find the maximum of the fused model
    nu_star, x_star, NU = probability_improvement(curr_max, 
                                        0.01, 
                                        fused_mean, 
                                        fused_var)
    output[4][1] = nu_star/cost[jj]
    output[4][2] = x_star
    
    """
    Upper Confidence Bound Approach
    """
    beta = np.abs((2*np.log(x_test.shape[1]*(iteration**2)*(np.pi**2)/(6/0.1)))/5)
    
    kt = np.sqrt(0.2 * beta)
    # Find the maximum of the fused model
    nu_star, x_star, NU = upper_conf_bound(kt, 
                                        fused_mean, 
                                        fused_var)
    output[5][1] = nu_star/cost[jj]
    output[5][2] = x_star
    
    
    
    # Add the actual input values for the maximum of the fused model
    if len(x_test.shape) > 1:
        for ii in range(x_test.shape[1]):
            output[0].append(x_test[index_max,ii])
            output[1].append(x_test[index_max,ii])
            output[2].append(x_test[index_max,ii])
            output[3].append(x_test[index_max,ii])
            output[4].append(x_test[index_max,ii])
            output[5].append(x_test[index_max,ii])
    else:
        output[0].append(x_test[index_max])
        output[1].append(x_test[index_max])
        output[2].append(x_test[index_max])
        output[3].append(x_test[index_max])
        output[4].append(x_test[index_max])
        output[5].append(x_test[index_max])
        
    for i in range(x_test.shape[1]):
        output[0].append(x_test[output[0][2],i])
        output[1].append(x_test[output[1][2],i])
        output[2].append(x_test[output[2][2],i])
        output[3].append(x_test[output[3][2],i])
        output[4].append(x_test[output[4][2],i])
        output[5].append(x_test[output[5][2],i])
        
    return param, output

def calculate_Greedy(param):
    """
    Parameters
    ----------
    param : tuple
        The input is a tuple that contains the data required for calculating the
        Maximum of a fused model constructed out of a reification 
        model object for Greedy optimization

    Returns
    -------
    results : list
        The output from the module contains information on some of the parameters
        used as inputs, as well as the maximum expected improvement value. Included
        in the output are the values for all the inputs that correspond to the maximum of the fused model

    """
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (finish, model_data, x_fused, fused_model_HP, \
     kernel, x_test, jj, kk, mm, true_sample_count, cost, curr_max) = data[param[0]]
    # Initialize the output  
    output = [0,0,0,jj,kk,mm]
    # Create the fused model
    model_temp.update_GP(*model_data)
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    # Use the fused model to obtain the mean and variance at all test points
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    fused_var = np.diag(fused_var)
    # Find the index of the test point that has the maximum of the fused model
    index_max_ = np.nonzero(fused_mean == np.max(fused_mean))
    # if there are more than on maxima, use the first index
    if index_max_[0].shape[0] > 1:
        index_max = int(index_max_[0][0])
    else:
        index_max = int(index_max_[0])
    # try:
    #     index_max = int(index_max_)
    # except TypeError:
    #     try:
    #         index_max = int(index_max_[0])
    #     except TypeError:
    #         index_max = int(index_max_[0][0])
    # Add the maximum of the fused model to the output  
    output[0] = np.max(fused_mean)
    # Add the maximum knowledge gradient and the index of the test point to the
    # output list
    output[1] = np.max(fused_mean)
    output[2] = index_max
    # Add the actual input values for the maximum of the fused model
    for kk in range(2):
        if len(x_test.shape) > 1:
            for ii in range(x_test.shape[1]):
                output.append(x_test[index_max,ii])
        else:
            output.append(x_test[index_max])
    # Return the results
    # print(output)
    return param, output   


def evaluateFusedModel(param):
    # in order to update the gains for the GP Hedge Portfolio optimization scheme
    # it is necessary to query the next best points predicted by all the acquisition
    # functions.
    with open("data/parameterSets/parameterSet{}".format(param[1]), 'rb') as f:
        data = load(f)
    with open("data/reificationObj", 'rb') as f:
        model_temp = load(f)
    (finish, model_data, x_fused, fused_model_HP, \
         kernel, x_test, curr_max, xi, acqIndex) = data[param[0]]
    # Create the fused model
    model_temp.create_fused_GP(x_fused, fused_model_HP[1:], 
                                fused_model_HP[0], 0.1, 
                                kernel)
    fused_mean, fused_var = model_temp.predict_fused_GP(x_test)
    return param, [acqIndex, fused_mean]
