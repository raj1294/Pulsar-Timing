#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 10:36:59 2025

@author: raj1294
"""
import warnings
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from stingray import AveragedPowerspectrum, EventList
from stingray import AveragedCrossspectrum, Lightcurve
from stingray.varenergyspectrum import LagSpectrum
import glob
from astropy.io import fits
from kapteyn import kmpfit
from scipy import integrate, signal    

#Function to average PSD and CPSD over M segments
def Pbin(Msegs,freqsarr,Pxarr,Pyarr,Cxyarr):
        
    favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg = [[],[],[],[],[],[],[]]
            
    #Average CPSD and PSD over M segments
    for iav in range(np.shape(Cxyarr)[1]):
        
        px = 0
        py = 0
        cxy = 0
        freq = freqsarr[0][iav]
        
        for jav in range(np.shape(Cxyarr)[0]):
            
            cxy += Cxyarr[jav][iav]/Msegs
            px += Pxarr[jav][iav]/Msegs
            py += Pyarr[jav][iav]/Msegs
                
        favg.append(freq)
        Cxyavg.append(cxy)
        Pxavg.append(px)
        Pyavg.append(py)
        dPxavg.append(px/np.sqrt(Msegs))
        dPyavg.append(py/np.sqrt(Msegs))
        dCxyavg.append(cxy/np.sqrt(Msegs))
    
    favg = np.array(favg)
    Pxavg = np.array(np.real(Pxavg))
    Pyavg = np.array(np.real(Pyavg))
    dPxavg = np.array(np.real(dPxavg))
    dPyavg = np.array(np.real(dPyavg))
    Cxyavg = np.array(Cxyavg)
    dCxyavg = np.array(dCxyavg)
        
    #Return averaged quantities
    return favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg

#Function to bin PSD and CPSD in frequency space
def fbin(bfact,farr,PXarr,PYarr,CXYarr,dPXarr,dPYarr,dCXYarr):
    
    avgf,avgpx,davgpx,avgpy,davgpy,avgcxy,davgcxy,Karr =\
    [[],[],[],[],[],[],[],[]]
        
    bmin = 0
    bmax = 0
    while(bmax<(len(farr))):
        
        if((bmax-bmin)!=0):
            bmin = bmax
        fmax = bfact*farr[bmax]        
        for index in range(bmin,len(farr)):
            if(farr[index]>=fmax):
                bmax = index
                break
            if(index==len(farr)-1 and farr[index]<=fmax):
                bmax = index + 1
                break
        if((bmax-bmin)==0):
            bfact+=0.1
                
        #Append relevant quantities        
        af = 0
        apx = 0
        apy = 0
        acxy = 0
        errpx = 0
        errpy = 0
        errcxy = 0
                
        for k2bin in range(bmin,bmax):
            af += farr[k2bin]
            apx += PXarr[k2bin]
            apy += PYarr[k2bin]
            acxy += CXYarr[k2bin]
            errpx += dPXarr[k2bin]**2
            errpy += dPYarr[k2bin]**2
            errcxy += dCXYarr[k2bin]**2
        
        af/=(bmax-bmin)
        apx/=(bmax-bmin)
        apy/=(bmax-bmin)
        acxy/=(bmax-bmin)
        errpx = np.sqrt(errpx)/(bmax-bmin)
        errpy = np.sqrt(errpy)/(bmax-bmin)
        errcxy = np.sqrt(errcxy)/(bmax-bmin)
        
        avgf.append(af)
        avgpx.append(apx)
        avgpy.append(apy)
        avgcxy.append(acxy)
        davgpx.append(errpx)
        davgpy.append(errpy)
        davgcxy.append(errcxy)        
        Karr.append(bmax-bmin)
            
    #Return binned quantities
    return avgf,avgpx,avgpy,avgcxy,davgpx,davgpy,davgcxy,Karr

#Estimate time lag in Fourier domain (Uttley et al. 2014)
def time_lag_func(lc,lcerr,reflc,reflcerr,lcbkg,refbkg,ywindow,\
                  Mseg,bfactor,dt,stat):
    
    freqs,Px,Py,Cxy = [[],[],[],[]]
    
    # Compute noise level depending on whether the counting statistics
    # are Poissonian or not 
    fnyq = 0.5*(dt**-1)
    Pnoise = 0
    Prefnoise = 0
    Msegnew = 0

    # Average power spectrum and cross spectrum over M segments
    for k3t in range(Mseg):
        
        # Split LC into M equal segments
        div = int(len(reflc)/Mseg)
        lctemp = lc[k3t*div:(k3t+1)*div]
        lctemperr = lcerr[k3t*div:(k3t+1)*div]
        reflctemp = reflc[k3t*div:(k3t+1)*div]
        reflctemperr = reflcerr[k3t*div:(k3t+1)*div]
        lcbkgtemp = lcbkg[k3t*div:(k3t+1)*div]
        refbkgtemp = refbkg[k3t*div:(k3t+1)*div]
        ywindowtemp = ywindow[k3t*div:(k3t+1)*div]
        
        if(stat=="Poissonian"):
            
            Pnoise += (2*(np.mean(lctemp) +\
                       np.mean(lcbkgtemp))/(np.mean(lctemp))**2)
            Prefnoise += (2*(np.mean(reflctemp) +\
                          np.mean(refbkgtemp))/(np.mean(reflctemp))**2)
        
        if(stat!="Poissonian"):
            
            errsq = np.sum(lctemperr**2)/len(lctemperr)
            errrefsq = np.sum(reflctemperr**2)/len(reflctemperr)
            
            Pnoise += errsq/(fnyq*(np.mean(lctemp))**2)
            Prefnoise += errrefsq/(fnyq*(np.mean(reflctemp))**2) 
                
        if(np.sum(lctemp)>-100):
            
            # FFT of comparison-band LC
            Xn = fft.fft(lctemp) 
            fxn = fft.fftfreq(len(lctemp),d=dt)
            
            # FFT of reference-band LC
            Yn = fft.fft(reflctemp) 
            fyn = fft.fftfreq(len(reflctemp),d=dt)
            
            # FFT of window function
            Wn = fft.fft(ywindowtemp)
            fwn = fft.fftfreq(len(ywindowtemp),d=dt)
            
            # remx,Xn = signal.deconvolve(Xn,Wn)
            # remy,Yn = signal.deconvolve(Yn,Wn)
                        
            Xn = Xn[fxn>0]
            Yn = Yn[fyn>0]
            Wn = Wn[fwn>0]
            fxn = fxn[fxn>0]
            fyn = fyn[fyn>0]
            fwn = fwn[fwn>0]
                                    
            # Compute PSD and CPSD with 
            # rms-squared normalisation for each segment
            normpsdx = (2.0*dt)/((len(lctemp))*(np.mean(lctemp))**2)
            normpsdy = (2.0*dt)/((len(reflctemp))*(np.mean(reflctemp))**2)
            normcross = (2.0*dt)/((len(lctemp))*(np.mean(lctemp))*\
                        (np.mean(reflctemp)))
                    
            Psdx = normpsdx*((np.conj(Xn))*(Xn))
            Psdy = normpsdy*((np.conj(Yn))*(Yn))
            Crossxy = normcross*((np.conj(Xn))*(Yn))
            Npsdx = len(Psdx)
                                                        
            # Append CPSD and PSD for each segment to 
            # pass to functions for averaging and binning
            if(len(Crossxy)>0 and len(Psdx)>0 and len(Psdy)>0):

                freqs.append(fxn)
                Px.append(Psdx)
                Py.append(Psdy)
                Cxy.append(Crossxy)
                Msegnew += 1
                        
            if(len(Crossxy)<=0 or len(Psdx)<=0 or len(Psdy)<=0):
                
                freqs.append(np.zeros(Npsdx))
                Px.append(np.zeros(Npsdx))
                Py.append(np.zeros(Npsdx))
                Cxy.append(np.zeros(Npsdx))
        
    #Average Pnoise and Prefnoise
    Pnoise /= Msegnew
    Prefnoise /= Msegnew
                    
    Mseg = Msegnew
    freqs = np.array(freqs)
    Px = np.array(Px)
    Py = np.array(Py)
    Cxy = np.array(Cxy)
    
    # print(np.shape(Cxy),reflcfile,complcfile)
            
    # Average PSDs and CPSDs over M segements
    favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg =\
    Pbin(Mseg,freqs,Px,Py,Cxy)
    
    fb = favg
    avgPx = Pxavg
    avgPy = Pyavg
    avgCxy = Cxyavg
    avgPxerr = dPxavg
    avgPyerr = dPyavg
    avgCxyerr = dCxyavg
    Karr = np.ones(len(Pxavg))

    # Implement frequency dependent binning of averaged PSDs and CPSDs
    if(bfactor>1):
        fb,avgPx,avgPy,avgCxy,avgPxerr,avgPyerr,avgCxyerr,Karr =\
        fbin(bfactor,favg,Pxavg,Pyavg,Cxyavg,dPxavg,dPyavg,dCxyavg)
    
    avgPx = np.real(avgPx)
    avgPy = np.real(avgPy)
    avgPx = np.array(avgPx)
    avgPxerr = np.array(avgPxerr)
    avgPy = np.array(avgPy)
    avgPyerr = np.array(avgPyerr)
    avgCxy = np.array(avgCxy)
    avgCxyerr = np.array(avgCxyerr)
    Karr = np.array(Karr)
    fb = np.array(fb)
    freqx = fb
    dfreqx = freqx[1]-freqx[0]
                                                    
    # Averaged number of samples
    nsamples = Mseg*Karr
        
    # Noise level of CPSD amplitude
    nbias = ((avgPx-Pnoise)*Prefnoise + (avgPy-Prefnoise)*Pnoise +\
            (Pnoise*Prefnoise))/nsamples
    
    # Complex-valued CPSD amplitude
    Cxyamp = (np.real(avgCxy))**2 + (np.imag(avgCxy))**2 - nbias
    
    avcxyreal = np.real(avgCxy)
    avcxyrealerr = np.real(avgCxyerr)
    avcxyimag = np.imag(avgCxy)
    avcxyimagerr = np.imag(avgCxyerr)
    
    dnbias = ((avgPxerr**2)*(Prefnoise**2) +\
              (avgPyerr**2)*(Pnoise**2))/(nsamples**2)
        
    dCxyamp = 4*((avcxyreal**2)*(avcxyrealerr**2) +\
                 (avcxyimag**2)*(avcxyimagerr**2)) + dnbias
        
    # #Upscale
    # Pdiffmin = abs(np.min(avgPx-Pnoise))
    # Pdiffminref = abs(np.min(avgPy-Prefnoise))
    # Cxyamp += abs(np.min(Cxyamp))
    # avgPx += Pnoise + Pdiffmin
    # avgPy += Prefnoise + Pdiffminref
    
    # #Compute covariance and its error over a desired frequency range
    # rmsx = np.sqrt((avgPx-Pnoise)*(dfreqx)*(np.mean(lc))**2)
    # rmsy = np.sqrt((avgPy-Prefnoise)*(dfreqx)*(np.mean(reflc))**2)
    # rmsx_noise = np.sqrt((Pnoise)*((np.mean(lc))**2)*(dfreqx))
    # rmsy_noise = np.sqrt((Prefnoise)*((np.mean(reflc))**2)*(dfreqx))
    
    # covariance = (np.mean(reflc))*np.sqrt((Cxyamp*dfreqx)/(avgPx-Pnoise))
    # dcovsqterm = ((covariance**2)*(rmsy_noise**2)+\
    #               (rmsy**2)*(rmsx_noise**2)+\
    #               (rmsy_noise**2)*(rmsx_noise**2))/(2*nsamples*rmsy**2)
    # covariance_err = np.sqrt(dcovsqterm)
            
    # #Filter over a frequency band 
    # covariance = covariance[freqx>fbmin]
    # covariance_err = covariance_err[freqx>fbmin]
    # freqxfilt = freqx[freqx>fbmin]
    # covariance = covariance[freqxfilt<fbmax]
    # covariance_err = covariance_err[freqxfilt<fbmax]
    # freqxfilt = freqxfilt[freqxfilt<fbmax]
            
    # Raw Coherence
    coherence = Cxyamp/((avgPx)*(avgPy))
    
    # # Raw uncertainty on coherence
    # dcoherence = (avgPxerr)*((-Cxyamp*avgPx**-2)/(avgPy))**2 +\
    #              (avgPyerr)*((-Cxyamp*avgPy**-2)/(avgPx))**2 +\
    #              (dCxyamp**2)/(avgPx*avgPy)**2
            
    # Statistical uncertainty on raw coherence
    dcoherence = ((2.0/(nsamples))**(0.5))*(1 - coherence**2)/\
                 (abs(coherence))
    
    coherence = np.sqrt(coherence)
    dcoherence = 0.5*(dcoherence)/(coherence)
    
    intcoherence = Cxyamp/((avgPx-Pnoise)*(avgPy-Prefnoise)) #Intrinsic
    
    # Uncertainty in intrinsic coherence (from Vaughan and Nowak 1997)
    intcoherr = np.zeros(len(intcoherence))
    arbfact = 3 
            
    for u in range(len(coherence)):
                    
        # High powers, high measured coherence         
        cond1 = (arbfact*Pnoise)/(np.sqrt(nsamples[u]))
        cond2 = (arbfact*Prefnoise)/(np.sqrt(nsamples[u]))
        cond3 = (arbfact*nbias[u])/((avgPx[u])*(avgPy[u]))
        
        if((avgPx[u]-Pnoise)>cond1 and (avgPy[u]-Prefnoise)>cond2\
            and coherence[u]>cond3):
                            
            intcoherr[u] = ((nsamples[u])**-0.5)*\
                           (np.sqrt((2*nsamples[u]*nbias[u]**2)/\
                           (Cxyamp[u] - nbias[u])**2 +\
                           ((Pnoise)/(avgPx[u]-Pnoise))**2 +\
                           ((Prefnoise)/(avgPy[u]-Prefnoise))**2 +\
                           (nsamples[u]*dcoherence[u]**2)/\
                           (intcoherence[u]**2)))
                                
            intcoherr[u] *= intcoherence[u]
            intcoherr[u] = abs(intcoherr[u])
            if(np.isnan(intcoherr[u])=='True'):
                intcoherr[u] = 0
        
        # High powers, low measured coherence 
        else:    
            intcoherr[u] = np.sqrt(Prefnoise**2/(avgPx[u]-Prefnoise)**2/\
            nsamples[u] + Pnoise**2/(avgPy[u]-Pnoise)**2/\
            nsamples[u] + (dcoherence[u]/intcoherence[u])**2)
            intcoherr[u] *= intcoherence[u]
            intcoherr[u] = abs(intcoherr[u])
            if(np.isnan(intcoherr[u])=='True'):
                intcoherr[u] = 0
            
    # Compute phase lag as a function of frequency between the 
    # two energy bands        
    Cxyimag = np.imag(avgCxy)
    Cxyreal = np.real(avgCxy)
    phaselag = np.zeros(len(Cxyimag))
    dphaselag = np.zeros(len(Cxyimag))
                                            
    for hp3 in range(len(phaselag)):
                    
        #Clockwise
        phaselag[hp3] = np.arctan(Cxyimag[hp3]/Cxyreal[hp3])
        div = phaselag[hp3]/np.pi
        
        #Ensure phase lag is confined to between -pi to pi
        if(div>1):
                        
            divs = str(div)
            divnum = int(divs.split(".")[0])
                                            
            #Should be between 0 to 1
            if((divnum-1)%3==0):
                div -= divnum
            
            #Should be between 0 to 1
            if(divnum%3==0):
                div -= divnum

            #Should be between -1 to 0
            if((divnum+1)%3==0):
                div += (divnum+1)
            
        if(div<-1):
            
            divs = str(div)
            divnum = int(divs.split(".")[0])
            
            #Should be between 0 to 1
            if((divnum-1)%3==0):
                div -= (divnum-1)
            
            #Should be between 0 to 1
            if(divnum%3==0):
                div -= divnum

            #Should be between -1 to 0
            if((divnum+1)%3==0):
                div -= divnum
        
        coherence[hp3] = np.sqrt(coherence[hp3])
        
        phaselag[hp3] = div*np.pi
        dphaselag[hp3] = np.sqrt((1.0-coherence[hp3]**2)/\
                         (2.0*nsamples[hp3]*coherence[hp3]**2))
        
    return freqx,dfreqx,phaselag,dphaselag

day = 86400

#Optical LC
tmidopt,rate_opt = np.loadtxt("continuum_luminosities.dat",\
                            skiprows=0,unpack=True)
#Swift-XRT LC
mjd_xray,rate,rate_err = [[],[],[]]
with open("ngc1313x2_swiftlc.qdp") as fi:
    for line in fi:
        line = line.split()
        if(len(line)>0):
            if(line[0]!='!' and line[0]!='READ' and line[0]!='!MJD'\
                and line[0]!='NO'):
                mjd_xray.append(float(line[0]))
                mjderr = 0.5*(float(line[1])+float(line[2]))
                rate.append(float(line[3]))
                rate_err.append(0.5*(abs(float(line[4]))+abs(float(line[5]))))
mjd_xray,rate_xray,rate_err_xray = zip(*sorted(zip(mjd_xray,rate,rate_err)))
mjd_xray = np.array(mjd_xray)
rate_xray = np.array(rate_xray)
rate_err_xray = np.array(rate_err_xray)

#Filter in time to match with epoch of Gemini data
tmin = 55115
tmax = 55195
rate_xray = rate_xray[mjd_xray>tmin]
rate_err_xray = rate_err_xray[mjd_xray>tmin]
mjd_xray = mjd_xray[mjd_xray>tmin]
rate_xray = rate_xray[mjd_xray<tmax]
rate_err_xray = rate_err_xray[mjd_xray<tmax]
mjd_xray = mjd_xray[mjd_xray<tmax]

#Interpolate
Npts = 500
tdata = np.linspace(tmin,tmax,Npts)
bsize = tdata[1]-tdata[0]
fmin = (tdata[-1]-tdata[0])**-1
fmax = 0.5*(bsize**-1)

ratexraydata = 0.5*(np.interp(tdata,mjd_xray,rate_xray+rate_err_xray)+\
                    np.interp(tdata,mjd_xray,rate_xray-rate_err_xray))
ratexraydataerr = 0.5*(np.interp(tdata,mjd_xray,rate_xray+rate_err_xray)-\
                    np.interp(tdata,mjd_xray,rate_xray-rate_err_xray))
rateoptdata = np.interp(tdata,tmidopt,rate_opt)
rateoptdataerr = np.zeros(len(rateoptdata))

compband = rateoptdata
refband = ratexraydata
compbanderr = rateoptdataerr
refbanderr = ratexraydataerr

#Background (X-ray and optical)
compbandbkg = np.zeros(len(compband))
refbandbkg = np.zeros(len(refband))
#Window
ywindow = np.zeros(len(refband))

Nseg = 1
bfac = 1.0
stats = "NP"

freq,dfreq,phaselag_gem,dphaselag_gem =\
    time_lag_func(compband,compbanderr,refband,refbanderr,compbandbkg,\
                  refbandbkg,ywindow,Nseg,bfac,bsize,stats)
#Phase-offsets
phaselag_gem[0] -= np.pi

isnanarr = np.isnan(phaselag_gem)
phaselag_gem = phaselag_gem[isnanarr==False]
dphaselag_gem = dphaselag_gem[isnanarr==False]
freq = freq[isnanarr==False]

isnanarr = np.isnan(dphaselag_gem)
phaselag_gem = phaselag_gem[isnanarr==False]
dphaselag_gem = dphaselag_gem[isnanarr==False]
freq = freq[isnanarr==False]

isnanarr = np.isnan(freq)
phaselag_gem = phaselag_gem[isnanarr==False]
dphaselag_gem = dphaselag_gem[isnanarr==False]
freq = freq[isnanarr==False]

timelag_gem = phaselag_gem/(2.0*np.pi*freq)
dtimelag_gem = dphaselag_gem/(2.0*np.pi*freq)

# plt.figure()
# plt.subplot(211)
# plt.errorbar(mjd_xray,rate_xray,rate_err_xray,fmt='gD',\
#              label="X-ray LC: Swift-XRT")
# plt.errorbar(tdata,ratexraydata,yerr=ratexraydataerr,fmt='b.',\
#              label="Interpolated X-ray LC: Swift-XRT")
# plt.xlabel("Time [MJD]",fontsize=14)
# plt.xlim(tmin,tmax)
# plt.ylabel("Swift-XRT count rate [s$^{-1}$]",fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=16)
# plt.legend(loc="best")
# plt.xticks([])
# plt.subplots_adjust(hspace=0)
# plt.subplot(212)
# plt.plot(tmidopt,rate_opt/1e37,'mD',label="Optical LC: GMOS-S")
# plt.errorbar(tdata,rateoptdata/1e37,yerr=rateoptdataerr/1e37,fmt='r.',\
#              label="Interpolated Optical LC: GMOS-S")
# plt.xlim(tmin,tmax)
# plt.ylabel("Optical luminosity [10$^{37}$ erg/s]",fontsize=14)
# plt.legend(loc="best")
# plt.xlabel("Time [MJD]",fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.show()

# freq = freq[1:]
# timelag_gem = timelag_gem[1:]
# dtimelag_gem = dtimelag_gem[1:]

plt.figure()
plt.errorbar(freq,timelag_gem,yerr=dtimelag_gem,fmt='k.')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel("Time lag [days]",fontsize=14)
plt.xlabel("Frequency [day$^{-1}$]",fontsize=14)
plt.show()


