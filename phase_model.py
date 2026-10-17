import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from kapteyn import kmpfit

day = 86400 #day to seconds
c = 3e8 #Speed of light
Msun = 1.99e30 #Mass of Sun
ls = c #Light-seconds
nu0 = 2.98148

def phase_mod_keplerian(p,x):
    
    Porb,Ab,ecc,T0,om,alpha,beta,gamma = p
    
    #Usual spin model
    phase_spin = alpha*(x-x[0])*day + 0.5*beta*((x-x[0])**2)*(day**2) +\
                  (1./6.)*gamma*((x-x[0])**3)*(day**3) + phase0

    #Orbital model (Keplerian eccentric)
    phase_orb = np.zeros(len(x))    
    Manom = (2.0*np.pi/Porb)*(x-T0) 
            
    for k in range(len(x)):
        
        def func(w):
            return (w - ecc*np.sin(w) - Manom[k])/(1.0 - ecc*np.cos(w))
    
        G = fsolve(func,0.0)[0]
        eta = (2.0*np.pi)/(Porb*(1 - ecc*np.cos(G)))/day
        Alpha = Ab*np.sin(om)
        Beta = 0
        if(1.0 - ecc**2 >= 0.0):
            Beta = Ab*(np.sqrt(1.0 - ecc**2))*(np.cos(om))
        if(1.0 - ecc**2 < 0):
            Beta = 0
        
        E1 = Alpha*(np.cos(G) - ecc) + Beta*np.sin(G)
        E2 = -Alpha*np.sin(G) + Beta*np.cos(G)
        E3 = -Alpha*np.cos(G) - Beta*np.sin(G)
        
        Becc = E1*(1 - eta*E2 + eta**2*\
        (E2**2 + 0.5*E1*E3 - 0.5*(ecc*E1*E2*np.sin(G))/\
        (1-ecc*np.cos(G))))
        
        phase_orb[k] = Becc/(nu0**-1) 
        
    if(ecc<0 or Porb<0 or Ab<0):
        phase_orb = np.zeros(len(x)) - 1000.0
    
    phase_tot = phase_orb + phase_spin
    
    return phase_tot

def residuals_keplerian(p, data):
    
    x, y, yerr = data
    Porb,Ab,ecc,T0,om,alpha,beta,gamma = p
    resid = (y - phase_mod_keplerian(p,x))/yerr
    
    return resid 

#Phase model for relativistic post-Keplerian orbits (including spin)
def phase_mod_relativistic(p,x):
    
    Porb,Ab,ecc,T0,om,Porbdot,omdot,deltar,deltah,Gamma,rshapiro,sshapiro,\
    alpha,beta,gamma = p
    
    #Usual spin model
    phase_spin = alpha*(x-x[0])*day + 0.5*beta*((x-x[0])**2)*(day**2) +\
                 (1./6.)*gamma*((x-x[0])**3)*(day**3)

    phase_orb = np.zeros(len(x))
        
    Omega0 = 2.0*np.pi/Porb
    Manom = Omega0*(x-T0) -\
            (0.5*Porbdot)/(2.0*np.pi)*((2.0*np.pi/Porb)*(x - T0))**2
            
    for k in range(len(Manom)):
        
        def func(w):
            return (w - ecc*np.sin(w) - Manom[k])/(1 - ecc*np.cos(w))
    
        G = fsolve(func,0.0)[0]
        arg = (np.sqrt((1+ecc)/(1-ecc)))*np.tan(G/2.)
        trueanom = 2.0*np.arctan(arg)
        
        Omega = om + omdot*(trueanom)    
        epsR = ecc*(1 + deltar)
        epstheta = ecc*(1 + deltah)
        
        eta = (2.0*np.pi)/(Porb*(1 - ecc*np.cos(G)))/day
        Alpha = Ab*np.sin(Omega)
        Beta = Ab*(np.sqrt(1-epstheta**2))*(np.cos(Omega)) + Gamma
    
        G1 = Alpha*(np.cos(G) - epsR) + Beta*np.sin(G)
        G2 = -Alpha*np.sin(G) + Beta*np.cos(G)
        G3 = -Alpha*np.cos(G) + Beta*np.sin(G)
        
        quantity = (1 - ecc*np.cos(G) -\
                   sshapiro*(np.sin(Omega))*(np.cos(G) - ecc) +\
                   (np.cos(Omega))*(np.sqrt(1-ecc**2))*(np.sin(G)))
        
        if(quantity>=1):
            G4 = -2.0*rshapiro*(np.log(quantity))
        else:
            G4 = 0
    
        #Barycentric arrival times
        Bgr = G1*(1.0 - eta*G2 + eta**2*(G2**2 + 0.5*G1*G3 -\
                  0.5*(ecc*G1*G2*np.sin(G))/(1 - ecc*np.cos(G)))) + G4
        
        #Phase model
        phase_orb[k] = Bgr/(nu0**-1)
            
    if(ecc<0 or Porb<0 or Ab<0 or omdot<0 or Gamma<0 or rshapiro<0 or\
       sshapiro<0):
        phase_orb = np.zeros(len(x)) - 1000.0

    phase_total = phase_orb + phase_spin
    
    return phase_total

def residuals_relativistic(p, data):
    
    x, y, yerr = data
    Porb,Ab,ecc,T0,om,Porbdot,omdot,deltar,deltah,Gamma,rshapiro,sshapiro,\
    alpha,beta,gamma = p
    resid = (y - phase_mod_relativistic(p,x))/yerr
    
    return resid 

def simple_sho(p,x):
    
    F1,F2,F3,amp1,amp2,P,tau,T0 = p
    
    ymod = F1 + F2*(x-x[0])*day + 0.5*F3*((x-x[0])**2)*(day**2) +\
            amp1*(np.exp(-0.5*(x-x[0])/tau))*\
            np.cos(2.0*np.pi*(x-T0)/P) -\
            (0.5*amp2/tau)*(np.exp(-0.5*(x-x[0])/tau))*\
            (np.sin(2.0*np.pi*(x-T0)/P))
    
    return ymod

def residuals_sho(p, data):
    
    x, y, yerr = data
    F1,F2,F3,amp1,amp2,P,tau,T0 = p
    resid = (y - simple_sho(p,x))/yerr
    
    return resid 

def glitches_mod(p,x):
    
    f0,fd1,glph_1,glf0_1,glf1_1,glph_2,glf0_2,glf1_2,glph_3,glf0_3,glf1_3 = p
    
    #Glitch epochs
    glep_1 = 12.25918961399293
    glep_2 = 35.25918961399293
    glep_3 = 81.25918961399293
    
    ymod = np.zeros(len(x))
    
    for k in range(len(x)):
                
        ymod[k] = f0*((x[k]-x[0])*day) + 0.5*fd1*((x[k]-x[0])*day)**2     

        if(x[k]>=glep_1):
            ymod[k] += \
            glph_1 + glf0_1*(x[k]-glep_1)*day +\
            0.5*glf1_1*((x[k]-glep_1)*day)**2
        
        if(x[k]>=glep_2):
            ymod[k] += \
            glph_2 + glf0_2*(x[k]-glep_2)*day +\
            0.5*glf1_2*((x[k]-glep_2)*day)**2
    
        if(x[k]>=glep_3):
            ymod[k] += \
            glph_3 + glf0_3*(x[k]-glep_3)*day +\
            0.5*glf1_3*((x[k]-glep_3)*day)**2

    return ymod

def residuals_glitch(p, data):
    
    x, y, yerr = data
    f0,fd1,glph_1,glf0_1,glf1_1,glph_2,glf0_2,glf1_2,glph_3,glf0_3,glf1_3 = p
    resid = (y - glitches_mod(p,x))/yerr
    
    return resid 

times,phases,dphases,_ = np.loadtxt("phase_resid.dat",skiprows=0,unpack=True)
dphases*=1.5
t0 = times[0]
phase0 = phases[0]
phases = phases - phase0
times = times - t0

f0init = 1e-7
f1init = 1e-12
glph_1 = 0.16
glf0_1 = 1e-7
glf1_1 = -0.05
glph_2 = 0.0
glf0_2 = 1e-8
glf1_2 = -0.12
glph_3 = 0.0
glf0_3 = -1e-7
glf1_3 = 0.0

paramsarr = [f0init,f1init,glph_1,glf0_1,glf1_1,glph_2,glf0_2,glf1_2,\
             glph_3,glf0_3,glf1_3]
fitobj = kmpfit.Fitter(residuals=residuals_glitch,data=(times,phases,dphases))
fitobj.fit(params0=paramsarr)

f0best = fitobj.params[0]
f0besterr = fitobj.xerror[0]
f1best = fitobj.params[1]
f1besterr = fitobj.xerror[1]
glph_1best = fitobj.params[2]
glph_1besterr = fitobj.xerror[2]
glf0_1best = fitobj.params[3]
glf0_1besterr = fitobj.xerror[3]
glf1_1best = fitobj.params[4]
glf1_1besterr = fitobj.xerror[4]
glph_2best = fitobj.params[5]
glph_2besterr = fitobj.xerror[5]
glf0_2best = fitobj.params[6]
glf0_2besterr = fitobj.xerror[6]
glf1_2best = fitobj.params[7]
glf1_2besterr = fitobj.xerror[7]
glph_3best = fitobj.params[8]
glph_3besterr = fitobj.xerror[8]
glf0_3best = fitobj.params[9]
glf0_3besterr = fitobj.xerror[9]
glf1_3best = fitobj.params[10]
glf1_3besterr = fitobj.xerror[10]

print("F0: ",f0best,f0besterr)
print("F1: ",f1best,f1besterr)
print("GP1: ",glph_1best,glph_1besterr)
print("GF01: ",glf0_1best,glf0_1besterr)
print("GF11: ",glf1_1best,glf1_1besterr)
print("GP2: ",glph_2best,glph_2besterr)
print("GF02: ",glf0_2best,glf0_2besterr)
print("GF12: ",glf1_2best,glf1_2besterr)
print("GP3: ",glph_3best,glph_3besterr)
print("GF03: ",glf0_3best,glf0_3besterr)
print("GF13: ",glf1_3best,glf1_3besterr)

bestfitpars = [f0best,f1best,glph_1best,glf0_1best,glf1_1best,glph_2best,\
                glf0_2best,glf1_2best,glph_3best,glf0_3best,glf1_3best]
resid = fitobj.residuals(p=bestfitpars,data=(times,phases,dphases))
residerr = np.ones(len(resid))
chi2min = fitobj.chi2_min
dof = fitobj.dof
print(chi2min,dof)

Nmod = 1000
tmod = np.linspace(np.min(times),np.max(times),Nmod)
phase_mod = glitches_mod(bestfitpars,tmod)

times += t0
tmod += t0

plt.figure()
plt.subplot(211)
plt.errorbar(times,phases,yerr=dphases,fmt='ko')
plt.plot(tmod,phase_mod,'b-')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel("Phase [cycles]",fontsize=14)
plt.xticks([])
plt.subplot(212)
plt.errorbar(times,resid,yerr=residerr,fmt='ko',label="Post-fit residuals")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplots_adjust(hspace=0)
plt.xlabel("Time [MJD]",fontsize=14)
plt.ylabel(r"Residuals [$\Delta \chi$]",fontsize=14)
plt.show()

#Compute priors for DDGR case
mus = 1e-6
tsol = 4.925490947*mus

Porbinit = 50.0
Abinit = 0.3
eccinit = 0.1
T0init = 15.0
ominit = 10.0
alphainit = 1e-6
betainit = 1e-13
gammainit = -1e-19

#Priors for post-Keplerian parameters (based on binary component masses)
Mp = 1.4
Mc = 3.0

Porbinit*=day
func = (1.0 + (73./24.)*(eccinit**2) + (37./96.)*(eccinit**4))/\
        (1-eccinit**2)**(7./2.)
pbdotinit = -(192.0*np.pi/5.)*(tsol**(5./3.))*\
            ((Porbinit/(2.0*np.pi))**(-5./3.))*(func)*(Mp*Mc)/(Mp+Mc)**(1./3.)
omdotinit = 3*(tsol**(2./3.))*((Porbinit/(2.0*np.pi))**(-2./3.))*\
                ((Mp + Mc)**(2./3.))/(1 - eccinit**2)
Gammainit = (tsol**(2./3.))*((Porbinit/(2.0*np.pi))**(1./3.))*(eccinit)*\
            (Mc*(Mp+2.0*Mc))/(Mp+Mc)**(4./3.)
rshapiroinit = tsol*Mc
sshapiroinit = (tsol**(-1./3.))*((Porbinit/(2.0*np.pi))**(-2./3.))*\
                (Abinit)*((Mp + Mc)**(2./3.))/Mc
deltahinit = (tsol**(2./3.))*((Porbinit/(2.0*np.pi))**(-2./3.))*\
              (7./2.*Mp**2 + 6*Mp*Mc + 2.0*Mc**2)/(Mp + Mc)**(4./3.)
deltarinit = (tsol**(2./3.))*((Porbinit/(2.0*np.pi))**(-2./3.))*\
              (3*Mp**2 + 6*Mp*Mc + 2.0*Mc**2)/(Mp + Mc)**(4./3.)
Porbinit/=day
print(pbdotinit,omdotinit,Gammainit,rshapiroinit)
print(sshapiroinit,deltahinit,deltarinit)
