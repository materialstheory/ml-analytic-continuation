"""
To generate spectral functions (both Arsenault and easy): final version
"""

import numpy as np
import random
from scipy.integrate import simps
import time,sys,os
import copy
import shutil
import pickle

empty_dict = {"wr":[], "sigmar":[], "ar":[], "cr":[], "R":[], "R_center":[]}
omegac = 8
Nomega = 800


def A_w(wr, sigmar, ar, cr, omega):
    """
    wr: normalized center of Gaussian peaks
    sigmar: normalized spread of Gaussian peaks
    ar: normalization factor of Gaussian peaks (dependent on sigmar)
    cr: weight of Gaussian peaks
    omega: omega grid ranges (-1, 1)
    return: the spectral function on the omega grid
    """
    omega_ = np.repeat(np.expand_dims(omega, axis = 0), wr.shape[0], axis = 0)
    sigmar_, wr_, ar_ = list(map(lambda arr: np.repeat(np.expand_dims(arr, axis=1), omega.shape[0], axis = 1), (sigmar, wr, ar)))
    
    A_w = np.average(ar_ * np.exp(-0.5*((omega_ - wr_)/sigmar_)**2), axis = 0, weights=cr)
    
    return A_w

def A_w_scaled(wr, sigmar, ar, cr, omega, factor):
    """
    wr: normalized center of Gaussian peaks
    sigmar: ...
    ar:...
    cr:...
    return numerical SF on a scaled grid
    omega: omega grid to be scaled to, non-uniform grid allowed
    factor: such that omega[-1]*factor = 1
    """
    omega_ = np.repeat(np.expand_dims(factor*omega, axis = 0), wr.shape[0], axis = 0) # scale omega grid to -1, 1
    sigmar_, wr_, ar_ = list(map(lambda arr: np.repeat(np.expand_dims(arr, axis=1), omega.shape[0], axis = 1), (sigmar, wr, ar)))
    
    A_w = factor*np.average(ar_ * np.exp(-0.5*((omega_ - wr_)/sigmar_)**2), axis = 0, weights=cr) # normalize spectral functions
    
    return A_w

class Arsenault_spectral_function:
    """
    Generates spectral functions with piecewise probabilities
    each instance is a set of settings
    """
    
    A_central_max = lambda self, wr, sigmar, ar, cr: np.max(A_w(wr, sigmar, ar, cr, np.arange(self.center_lim[0], self.center_lim[1], 0.02)))
    A_offcentral_max = lambda self, wr, sigmar, ar, cr: np.max(A_w(wr, sigmar, ar, cr, 
                                                                np.concatenate((np.arange(-1, self.center_lim[0], 0.01), 
                                                                                np.arange(self.center_lim[1], 1, 0.02)))))
    is_smooth = lambda self, wr, sigmar, ar, cr : self.delta_smooth > simps(np.abs(np.gradient(np.gradient(A_w(wr, sigmar, ar, cr, 
                                                               np.arange(self.center_lim[0], self.center_lim[1], 0.0001))))), 
                                                               np.arange(self.center_lim[0], self.center_lim[1], 0.0001))    
    
    vanish_at = lambda self, pos, wr, sigmar, ar, cr: np.average(ar * np.exp(-1/2 * ((pos - wr)/sigmar)**2), weights = cr) < self.delta
    
    
    def __init__(self, omega, clim, sigma_c, sigma_o, delta, cr, cr_thr, delta_smooth):
        """
        initiate parameters except for # of peaks
        omega: normalized to omega in (-1, 1)
        clim, sigma_c, sigma_o, delta, cr, cr_thr, delta_smooth: all normalized to omega in (-1, 1)   
        """
        self.omega = omega
        self.center_lim = [-clim[0], clim[0]]
        self.center_lim2 = [-clim[1], clim[1]]
        self.center_lim3 = [-clim[2], clim[2]]
        
        self.sigma_min_center, self.sigma_max_center = sigma_c
        
        self.sigma_min_offcenter, self.sigma_max_offcenter = sigma_o
        
        self.delta = delta
        self.cr_thr = cr_thr
        self.cr = cr
        self.delta_smooth = delta_smooth
        self.empty_dict = {"wr":[], "sigmar":[], "ar":[], "cr":[], "R":[], "R_center":[]}
        
        
    def build(self, n_repeat, n_peak, n_center_peak):
        """
        build normalized spectral functions
        n_repeak: # of A(omega) to generate
        n_peak: list/tuple of len 2, total number of peaks
        n_center_peak: list/tuple of len 2, number of peaks within center
        return: parameters determining the spectral function stored in a dict
        """
        
        self.res = copy.deepcopy(self.empty_dict)
        start_time = time.time()

        i = 0
        while(i < n_repeat):
    
            R = np.random.randint(low = n_peak[0], high = n_peak[1]) # 8, 34
            R_center = np.random.randint(low= n_center_peak[0], high = n_center_peak[1])  # 0, 4;  number of peaks
        
            wr_center = np.random.uniform(low=self.center_lim[0], high=self.center_lim[1], size = R_center) #peak center 
    
            wr_offcenter = np.random.uniform(-1, 1, R - R_center)*(self.center_lim3[1] - self.center_lim[1])
            wr_offcenter = np.sign(wr_offcenter)*(np.abs(wr_offcenter) + self.center_lim[1])
        
            sigmar_center = np.random.uniform(low = self.sigma_min_center, high = self.sigma_max_center, size = R_center) # peak width
            sigmar_offcenter = np.random.uniform(low = self.sigma_min_offcenter, high = self.sigma_max_offcenter, size = R - R_center)
        
        
            sigmar_offcenter[np.abs(wr_offcenter)<self.center_lim2[1]] = \
            sigmar_offcenter[np.abs(wr_offcenter)<self.center_lim2[1]]*(self.sigma_max_center - self.sigma_min_center)+self.sigma_min_center
    
            sigmar_offcenter[np.abs(wr_offcenter)>=self.center_lim2[1]] = \
            sigmar_offcenter[np.abs(wr_offcenter)>=self.center_lim2[1]]*(self.sigma_max_offcenter - self.sigma_min_offcenter) + self.sigma_min_offcenter
        
            wr = np.concatenate((wr_center, wr_offcenter))
            sigmar = np.concatenate((sigmar_center, sigmar_offcenter))
        
            ar = 1/np.sqrt(2*np.pi)/sigmar
            cr = np.append(np.random.uniform(low = self.cr[0], high = self.cr[1], size = R_center), 
                       np.random.uniform(low = self.cr[2], high = self.cr[3], size = R-R_center))
            
            # check if the function vanish at both ends, otherwise, reduce weight first, then spread, then remove
            for pos in [-1, 1]:
            
                while not self.vanish_at(pos, wr, sigmar, ar, cr):
                    ind = np.argmax(cr * ar * np.exp(-1/2 * ((pos - wr)/sigmar)**2))
                    if cr[ind]*ar[ind]/np.sum(cr) > self.cr_thr:
                        cr[ind] *= 0.5        
                    elif sigmar[ind]*0.7 > self.sigma_min_offcenter:
                        sigmar[ind] *= 0.7   
                        ar[ind] /= 0.7
                    else:
                        wr, sigmar, ar, cr = list(map(lambda arr: np.delete(arr, ind), (wr, sigmar, ar, cr)))
                        R -= 1
                        if ind < R_center:
                            R_center -= 1
            if len(wr) > 3:
                i += 1    
            
            # ratio condition removed
            # smooth condition removed          
 
                list(map(lambda lis, elem: lis.append(elem), 
                    (self.res["wr"], self.res["sigmar"], self.res["ar"], self.res["cr"], self.res["R"], self.res["R_center"]),
                    (wr, sigmar, ar, cr, R, R_center)))  

        total_time = time.time() - start_time
        print("Arsenault_spectral_function: " + str(len(self.res["wr"])) + "spectral functions are generated in " + str(total_time) + " s.")       
        return copy.deepcopy(self.res)     
    
    def print(self, res, omega, factor = 1):
        """
        return numerical array for spectral functions
        res: param dict
        omega: symmetric omega grid with symmetric range to scale into
        """
        A_omega = np.zeros((len(res["wr"]), omega.shape[0]))
        for j in range(len(res["wr"])):
              
            A_omega[j, :] = \
            A_w_scaled(res["wr"][j], res["sigmar"][j], res["ar"][j], res["cr"][j], omega, 1/omega[-1]*factor)

        return A_omega
    
    def print_theta_omega(self, res, omega, omega_0, factor = 1):
        """
        not used
        """
        A_omega = np.zeros((len(res["wr"]), omega.shape[0]))
        for j in range(len(res["wr"])):
              
            A_omega[j, :] = \
            A_w_scaled(res["wr"][j], res["sigmar"][j], res["ar"][j], res["cr"][j], omega, 1/omega_0*factor)

        return A_omega
    
    def build_var(self, res, n_repeat):
        """
        include small variation of the spectral functions
        return dict containing parameters
        not in use
        """
        self.res_var = copy.deepcopy(self.empty_dict)
        
        for i in range(len(res["wr"])):
            wr, sigmar, ar, cr, R, R_center = list(map(lambda x: res[x][i], tuple(res)))
            
            for j in range(n_repeat):
                wr_ = np.append(wr, np.random.uniform(size = 5)*(self.center_lim3[1] - self.center_lim3[0]) + self.center_lim3[0])
                sigmar_ = np.append(sigmar, np.random.uniform(low=0.1, high = 0.2, size = 5))
                ar_ = 1/np.sqrt(2*np.pi)/sigmar_
                cr_ = cr/np.sum(cr)/(1-0.05)
                cr_ = np.append(cr_, [0.01]*5)
                
                R_ = 5 + R
                R_center_ = np.sum(np.abs(wr_[-6:]) < self.center_lim[1]) + R_center
    
                        
                list(map(lambda lis, elem: lis.append(elem), 
                        list(map(lambda x: self.res_var[x], tuple(self.res_var))),   # one peak
                        (wr_, sigmar_, ar_, cr_, R_, R_center_)))  
            
        return copy.deepcopy(self.res_var)
    
    def build_linear(self):
        """
        build pairs of spectral functions, such that their linear combination is an existing SF in three ways
        """
        self.res_1peak      = copy.deepcopy(self.empty_dict)
        self.res_minus1peak = copy.deepcopy(self.empty_dict)
        self.res_2peak      = copy.deepcopy(self.empty_dict)
        self.res_minus2peak = copy.deepcopy(self.empty_dict)
        self.res_half1      = copy.deepcopy(self.empty_dict)
        self.res_half2      = copy.deepcopy(self.empty_dict)
        
        for i in range(len(self.res["wr"])):
            wr, sigmar, ar, cr, R, R_center = list(map(lambda x: self.res[x][i], tuple(self.res)))
            
            for j in range(100):
                ind1 = np.random.choice(R, size = 1, replace = False)                
                if np.sum(cr[ind1]) > 0.1:
                    break
            
            list(map(lambda lis, elem: lis.append(elem), 
                    list(map(lambda x: self.res_1peak[x], tuple(self.res_1peak))),   # one peak
                    (wr[ind1], sigmar[ind1], ar[ind1], [1], 1, np.sum(ind1<R_center))))    
            
            ind2 = np.delete(np.arange(0, R, 1), ind1)
        
            list(map(lambda lis, elem: lis.append(elem), 
                map(lambda x: self.res_minus1peak[x], tuple(self.res_minus1peak)),  # counterparts of one peak
                (wr[ind2], sigmar[ind2], ar[ind2], cr[ind2]/np.sum(cr[ind2]), R-1, R_center - np.sum(ind1 < R_center))))
            
            for j in range(100):
                ind5 = np.random.choice(R, size = 2, replace = False)
                if np.sum(cr[ind5]) > 0.1:
                    break
                    
            list(map(lambda lis, elem: lis.append(elem), 
                    list(map(lambda x: self.res_2peak[x], tuple(self.res_2peak))),   # two peaks
                    (wr[ind5], sigmar[ind5], ar[ind5], cr[ind5]/np.sum(cr[ind5]), 2, np.sum(ind5<R_center))))   
            
            ind6 = np.delete(np.arange(0, R, 1), ind5)
            
            list(map(lambda lis, elem: lis.append(elem), 
                map(lambda x: self.res_minus2peak[x], tuple(self.res_minus2peak)),  # counterparts of two peaks
                (wr[ind6], sigmar[ind6], ar[ind6], cr[ind6]/np.sum(cr[ind6]), R-2, R_center - np.sum(ind6 < R_center))))            
            
                   
            for j in range(100):
                ind3 = np.random.choice(R, size = int(R/2), replace = False)             
                if np.sum(cr[ind3]) > 0.1:
                    break
            
            ind4 = np.delete(np.arange(0, R, 1), ind3)
              
            list(map(lambda lis, elem: lis.append(elem), 
                map(lambda x: self.res_half1[x], tuple(self.res_half1)),   # half of the peaks
                (wr[ind3], sigmar[ind3], ar[ind3], cr[ind3]/np.sum(cr[ind3]), int(R/2), np.sum(ind3<R_center))))
           
            list(map(lambda lis, elem: lis.append(elem), 
                map(lambda x: self.res_half2[x], tuple(self.res_half2)),   # the other half of the peaks
                (wr[ind4], sigmar[ind4], ar[ind4], cr[ind4]/np.sum(cr[ind4]), R-int(R/2), R_center - np.sum(ind3<R_center)))) 
            
        return copy.deepcopy(self.res_half2), copy.deepcopy(self.res_half1), copy.deepcopy(self.res_minus1peak), copy.deepcopy(self.res_1peak), copy.deepcopy(self.res_minus2peak), copy.deepcopy(self.res_2peak)
        
def load_create(Arsenault_spectral_function, n_repeat, filename):
    """
    Arsenault_spectral_function: Arsenault_spectral_function class
    n_repeat: number of spectral functions to generate
    filename: input parameter files
    return: Arsenault_spectral_function class object
    """
    params = np.loadtxt(filename)
    omega = np.linspace(-1, 1, Nomega)
    return_obj = Arsenault_spectral_function(omega, 
                                             params[0:3], params[3:5], params[5:7], 
                                             params[7], params[8:12], params[12], params[13])
    
    _ = return_obj.build(n_repeat, params[14:16], params[16:18])
    return return_obj


def augmentation(Arsenault):
    """
    Arsenault: Arsenault_spectral_function instance
    return linear and scaling augmentated spectral functions and the original one
    """

    omega = np.linspace(-omegac, omegac, Nomega)

    A = Arsenault.print(Arsenault.res, omega)
    _ = Arsenault.build_linear()
    A_p1 = Arsenault.print(Arsenault.res_1peak, omega)
    A_pm1 = Arsenault.print(Arsenault.res_minus1peak, omega)
    A_half1 = Arsenault.print(Arsenault.res_half1, omega)
    A_half2 = Arsenault.print(Arsenault.res_half2, omega)
    A_12 = Arsenault.print(Arsenault.res, omega, 1.2)
    A_14 = Arsenault.print(Arsenault.res, omega, 1.4)
    A_16 = Arsenault.print(Arsenault.res, omega, 1.6)
    A_18 = Arsenault.print(Arsenault.res, omega, 1.8)
    
    return np.concatenate([A, A_p1, A_pm1, A_half1, A_half2, A_12, A_14, A_16, A_18], axis = 0)

def main():
    """
    generate N spectral functions with summation normalization
    for reproducibility, the following is quite verbose
    """
    start_time = time.time()
    print('Starting generating spectral functions A_omega')

    np.random.seed(42) #123456 for test set
    Arsenault_train = load_create(Arsenault_spectral_function, repetitions, "Arsenault_param.txt")
    Arsenault_val   = load_create(Arsenault_spectral_function, repetitions, "Arsenault_param.txt")
    easy_train      = load_create(Arsenault_spectral_function, repetitions, "Arsenault_easy_param.txt")
    easy_val        = load_create(Arsenault_spectral_function, repetitions, "Arsenault_easy_param.txt")


    omega = np.linspace(-omegac, omegac, Nomega)
    A_omega_train_aug = augmentation(Arsenault_train)
    A_omega_val_aug = augmentation(Arsenault_val)
    A_easy_omega_train_aug = augmentation(easy_train)
    A_easy_omega_val_aug = augmentation(easy_val)
    
    normalize = lambda arr: arr*(2*omegac)/(Nomega-1)

    A_omega_train_aug = normalize(A_omega_train_aug)
    A_omega_val_aug = normalize(A_omega_val_aug)
    A_easy_omega_train_aug = normalize(A_easy_omega_train_aug)
    A_easy_omega_val_aug = normalize(A_easy_omega_val_aug)
    
    np.save("A_omega_train.npy", A_omega_train_aug)
    np.save("A_omega_val.npy", A_omega_val_aug)
    np.save("A_easy_omega_train.npy", A_easy_omega_train_aug)
    np.save("A_easy_omega_val.npy", A_easy_omega_val_aug)
          
    total_time = time.time() - start_time
    print('\nDONE')
    print(('A_of_omega generation took {:.0f} s.'.format(total_time)))
    
    
    
    

if __name__ == "__main__":       
    repetitions= int(sys.argv[1])  
    #augmentation = int(sys.argv[2])
    main()
