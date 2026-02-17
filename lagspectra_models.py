#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 12:54:47 2025

@author: raj1294
"""
import numpy as np
import matplotlib.pyplot as plt
from stingray import AveragedPowerspectrum, EventList
from stingray import AveragedCrossspectrum, Lightcurve
from astropy.io import fits
from stingray.varenergyspectrum import LagSpectrum
import glob
from kapteyn import kmpfit
from matplotlib.ticker import ScalarFormatter
from scipy import integrate, signal
from scipy import optimize
from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve

Msun = 1.99e30
G = 6.67e-11
c = 3e8
sizefact = 1e6

def reverbmod(p,data):
    
    M,dist = p
    ratearr,rateerr = data
    
    #Width of tophat kernel
    Rg = (G*M/c**2)
    tau = (dist)*(sizefact)*(Rg/c)
    
    tophatkern = Box1DKernel(tau)
    ratemod = convolve(ratearr,tophatkern)
    
    return ratemod
    
def residuals_reverbmod(p,data):
    
    M,dist = p
    ratearr,rateerr = data
    resid = (ratearr - reverbmod(p,data))/rateerr
    
    return resid 

def ignore_btis(arrays,tS,tE):
        
    tref = arrays[-1]   
    oldarrays_list,newarrays_list = [[],[]]
                
    for qd in range(len(arrays)):
                                                                                
        #Identify BTIs
        for qd2 in range(len(tS)-1):
            
            btiS = tE[qd2]
            btiE = tS[qd2+1]
                        
            for qd3 in range(len(tref)):
                
                if(tref[qd3]>=btiS and tref[qd3]<=btiE\
                   and qd!=len(arrays)-1):
                    
                    newindarray = np.arange(0,len(arrays),1)
                    for qd4 in range(len(newindarray)):
                        arrays[qd4][qd3] = -1e10
                        
        #Remove BTIs
        if(qd!=len(arrays)-1):
            newarray = arrays[qd][arrays[qd]>-1e9]
            newarrays_list.append(newarray)
        oldarrays_list.append(arrays[qd])
                
    for pind in range(len(newarrays_list)):    
        newarrays_list[pind] = np.array(newarrays_list[pind])
    newarrays_list = np.array(newarrays_list)
    for pind2 in range(len(oldarrays_list)):
        oldarrays_list[pind2] = np.array(oldarrays_list[pind2])
    oldarrays_list = np.array(oldarrays_list)
        
    return oldarrays_list, newarrays_list

def remove_nans(arrnans):
        
    newarrnanslist = []
    for qdnans in range(len(arrnans)):
                                                                              
        isnanarr = np.isnan(arrnans[qdnans])
        for qdnans2 in range(len(isnanarr)):   
            
            if(isnanarr[qdnans2]==True):
                
                newindex = np.arange(0,len(arrnans),1)
                
                for qdnans3 in range(len(newindex)):
                    arrnans[qdnans3][qdnans2] = -1e10
        
        newarrnanslist.append(arrnans[qdnans])

    for pindnan in range(len(newarrnanslist)):    
        newarrnanslist[pindnan] = np.array(newarrnanslist[pindnan])
        newarrnanslist[pindnan] = newarrnanslist[pindnan]\
                                  [newarrnanslist[pindnan]>-1e9]
    newarrnanslist = np.array(newarrnanslist)
    return newarrnanslist


ks = 1000 #ks in seconds
reblog = 0.0 # Geometric binning factor
geombin = 1.0 #Geometric binning factor
Ntrialmcmc = 1 #Number of MCMC simulations
Nenergies = 15 #Number of energy bins
stats = "NP" #Poisson statistics for LC
Plot = "no" #Plot Lag-energy spectrum
Plotmcmc = "no" #Plot MCMC generated LCs
obsid = "0921360201" #Observation visit identifier

#Instruments
instarr = ["epn"]
labinst = ["EPN"]
col = ["bo","go","ro"]

#Frequency bins
fbinmin = 1e-4
fbinmax = 5e-4
fbinmin = np.array(fbinmin)
fbinmax = np.array(fbinmax)

Energy,dEnergy,Fvarcomp,dFvarcomp,Fvarref,dFvarref =\
[[],[],[],[],[],[]]    
    
for qinstr in range(len(instarr)):
    
    instr = instarr[qinstr]
    
    enlag,denlag,mlag,mlagerr,mlagS,mlagerrS,mufakelag,fakelagerr,\
    = [[],[],[],[],[],[],[],[]]
    
    for k2 in range(Nenergies-1):
                    
        evcomp,rcomp,errcomp,rcompbkg,errcompbkg =\
        [[],[],[],[],[]]
        evref,rref,errref,rrefbkg,errrefbkg,ywref =\
        [[],[],[],[],[],[]]
        
        nobskey = instr + "*net*obs" + str(obsid) + "*_1_*en4*ref*.lc"
                    
        for reflcfile in sorted(glob.glob(nobskey)):
                                                                                                                        
            ObsId = reflcfile.split("obs")[1].split("_")[0]
            
            reflcfile = instr + "_net_obs" + ObsId + "_1_en" +\
                         str(k2+1) + "_ref.lc"
                        
            refbkgfile = instr + "_bkg_obs" + ObsId + "_1_en" +\
                         str(k2+1) + "_ref.lc"
            
            refevfile = instr + "_net_obs" + ObsId + "_1_en" +\
                         str(k2+1) + "_ref.fits"
    
            complcfile = instr + "_net_obs" + ObsId + "_1_en" +\
                         str(k2+1) + "_comp.lc"
                                    
            compbkgfile = instr + "_bkg_obs" + ObsId + "_1_en" +\
                          str(k2+1) + "_comp.lc"
            
            compevfile = instr + "_net_obs" + ObsId + "_1_en" +\
                         str(k2+1) + "_comp.fits"
                        
            #Reference band (source LC)
            hdulistref = fits.open(reflcfile)
            dataref = hdulistref[1].data
            timeref = dataref['TIME']  
            rateref = dataref['RATE']
            errorref = dataref['ERROR']
            bsize = hdulistref[1].header['TIMEDEL']
            inst = hdulistref[1].header['INSTRUME']
            telapse = timeref[-1]-timeref[0]
            tstart = timeref[0]
            tend = timeref[-1]
                                                        
            #Reference band (events)
            hdulistrefev = fits.open(refevfile)
            datarefev = hdulistrefev[1].data
            evrefR = datarefev['TIME']
            evrefR = EventList(time=evrefR)
                    
            #Reference band (background LC)
            hdulistref_bkg = fits.open(refbkgfile)
            dataref_bkg = hdulistref_bkg[1].data
            raterefbkg = dataref_bkg['RATE']
            errorrefbkg = dataref_bkg['ERROR']
            
            #Comparison band (source LC)
            hdulistcomp = fits.open(complcfile)
            datacomp = hdulistcomp[1].data
            headercomp = hdulistcomp[1].header
            ratecomp = datacomp['RATE']
            errorcomp = datacomp['ERROR']
                        
            #Comparison band (events)
            hdulistcompev = fits.open(compevfile)
            datacompev = hdulistcompev[1].data
            evcompR = datacompev['TIME']
            evcompR = EventList(time=evcompR)
            
            #Comparison band (background LC)
            hdulistbkg = fits.open(compbkgfile)
            databkg = hdulistbkg[1].data
            ratecompbkg = databkg['RATE']
            errorcompbkg = databkg['ERROR']
            dateref = hdulistbkg[0].header['DATE-OBS']
            
            #GTIs (reference-band)
            tstartR = hdulistref[2].data['START']
            tstopR = hdulistref[2].data['STOP']
            
            #GTIs (comparison-band)
            tstartC = hdulistcomp[2].data['START']
            tstopC = hdulistcomp[2].data['STOP']

            #Remove BTIs and NANs from reference band and comparison band
            arraysR = np.transpose(np.column_stack((rateref,errorref,\
                      raterefbkg,errorrefbkg,ratecomp,errorcomp,\
                      ratecompbkg,errorcompbkg,timeref))) 
                                                            
            if(len(tstartR)>1):
                                
                arraysR, arraysN = ignore_btis(arraysR,tstartR,tstopR)
            
                rateref,errorref,\
                raterefbkg,errorrefbkg,ratecomp,errorcomp,\
                ratecompbkg,errorcompbkg = arraysN
            
            if(len(tstartR)==1):
                arraysN = remove_nans(arraysR)
                rateref,errorref,\
                raterefbkg,errorrefbkg,ratecomp,errorcomp,\
                ratecompbkg,errorcompbkg,timeref = arraysN
            
            for ml in range(len(errorcomp)):
                if(errorcomp[ml]==0):
                    errorcomp[ml] = np.mean(errorcomp)
            
            # zeroleft = np.zeros(200)
            # zeroright = np.zeros(200)
            # rateref = np.hstack((zeroleft,rateref))
            # rateref = np.hstack((rateref,zeroright))
            # ratecomp = np.hstack((zeroleft,ratecomp))
            # ratecomp = np.hstack((ratecomp,zeroright))
            # errorref = np.hstack((zeroleft,errorref))
            # errorref = np.hstack((errorref,zeroright))
            # errorcomp = np.hstack((zeroleft,errorcomp))
            # errorcomp = np.hstack((errorcomp,zeroright))
            timeref = bsize*np.arange(0,len(rateref),1)
            timecomp = bsize*np.arange(0,len(rateref),1)
            
            Minit = 8*Msun
            distinit = 0.1
            paramsarr = [Minit,distinit]
            fitobj = kmpfit.Fitter(residuals=residuals_reverbmod,\
                                   data=(ratecomp,errorcomp))
            fitobj.fit(params0=paramsarr)
            Mbest = fitobj.params[0]
            Mbesterr = fitobj.xerror[0]
            distbest = fitobj.params[1]
            distbesterr = fitobj.xerror[1]
            
            # print(Mbest/Msun,Mbesterr/Msun)
            # print(distbest,distbesterr)

            bestfitpars = [Mbest,distbest]
            ratecompmodbest = reverbmod(paramsarr,(ratecomp,errorcomp))
            
            plt.errorbar(timecomp/ks,ratecomp,yerr=errorcomp,fmt='r.')  
            plt.plot(timecomp/ks,ratecompmodbest,'b-')
            plt.tick_params(axis='both', which='major', labelsize=18)
            plt.legend(loc="best")
            plt.title("XMM-Newton (EPIC-PN) lightcurves: " + str(obsid),\
                      fontsize=18)
            plt.ylabel("Count rate [s$^{-1}$]",fontsize=18)
            plt.xlabel("Time [ks]",fontsize=18)
            plt.show()

            # tophatkern = Box1DKernel(widths[k2])
            
            # rateconv =\
            # 0.5*(convolve(ratecomp+errorcomp,tophatkern) +\
            #      convolve(ratecomp-errorcomp,tophatkern))

            # errorconv =\
            # 0.5*(convolve(ratecomp+errorcomp,tophatkern) -\
            #      convolve(ratecomp-errorcomp,tophatkern))            
        
            # plt.errorbar(timeref/ks,rateref,yerr=errorref,fmt='r.')       
            # plt.errorbar(timeref/ks,rateconv,yerr=errorconv,fmt='k.')
            
            # # plt.errorbar(timecomp/ks,rateconv,yerr=errorconv,fmt='b.')
            # plt.tick_params(axis='both', which='major', labelsize=18)
            # plt.legend(loc="best")
            # plt.title("XMM-Newton (EPIC-PN) lightcurves: " + str(obsid),\
            #           fontsize=18)
            # plt.ylabel("Count rate [s$^{-1}$]",fontsize=18)
            # plt.xlabel("Time [ks]",fontsize=18)
            # plt.show()



