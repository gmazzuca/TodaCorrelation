
''' Libraries'''

import numpy as np
import sys
from scipy.special import polygamma as Polygamma
from scipy.special import gamma as Gamma

########################################
########################################
########################################


def Energy_Toda_Vectorized(p,r):
    ''' Energy vectorized, we have to give r instead of q '''
    return p**2*0.5 + np.exp(-r)

def Energy_Exp_Toda(p,r):
    ''' Energy for toda exp, prepares the data for Energy_Exp_Toda_Vectorized'''
    return np.sum( Energy_Toda_Vectorized(p,r))

def evolution_periodic(p0,r0,time = [0,10],time_step = 0.1):
    '''Evolution, 
    p0,r0 = initial data
    time = vector of times for which I want to register the data
    time_step = time step for the evolution

    This function compute the solution till the last entries of the vector time, with time step equal to time_step
    the standar algorithm will be a yoshida4 implemented through a leap-frog 
    '''

    #setting the output vectors
    time_snap = len(time)
    particles = len(p0)

    p = np.zeros((particles,time_snap))
    r = np.zeros((particles,time_snap))
    p[:,0] = p0
    r[:,0] = r0

    for j in range(1,time_snap):
        (p[:,j], r[:,j]) = one_step_evo(p[:,j-1],r[:,j-1],time[j] - time[j-1], time_step)

    return (p,r)


def one_step_evo(p,r,time_interval,time_step):
    ''' Integration step
    p,r = initial data
    time_interval = interval of integration
    time_step = integration step
     '''

    t = 0

    while t < time_interval:

        (p,r) = verlet(p,r,time_step)
        t += time_step

    return (p,r) 


def verlet(p,r,dt):
    ''' step of verlet algorith, good reference https://www.unige.ch/~hairer/poly_geoint/week2.pdf 
    Slightly modificed in order to work with the variables p_j,r_j'''

    r_exp = np.exp(-r)
    r_exp_diff = np.ediff1d(r_exp, to_begin = r_exp[0] - r_exp[-1])
    p_mid = p - 0.5*dt*r_exp_diff 

   
    p_diff = np.ediff1d(p_mid, to_end = p_mid[0] - p_mid[-1])
    r_new = r + dt*p_diff

    r_exp = np.exp(-r_new)
    r_exp_diff = np.ediff1d(r_exp, to_begin = r_exp[0] - r_exp[-1])
    p_new = p_mid - 0.5*dt*r_exp_diff 

    return p_new, r_new


def sample_general_gibbs(particles, beta=1, eta = 1):
    '''Sampling variables according to the standard Gibbs measure 

    particles = number of particles
    beta,eta = parameter for the measure 
    '''

    over_beta_sqrt = 1/np.sqrt(beta)
    p = np.random.normal(0,over_beta_sqrt, size = particles)                     
    x = np.sqrt(np.random.chisquare(2*eta, size = particles))
    r = -2*np.log(x*over_beta_sqrt/np.sqrt(2))
    
    return (p,r)


def correlation_toda(p,r,p0,r0):

    n,snap = p.shape
    corr_p_main = np.zeros((n,snap))
    corr_r_main = np.zeros((n,snap))
    corr_e_main = np.zeros((n,snap))
    corr_pr_main = np.zeros((n,snap))
    corr_er_main = np.zeros((n,snap))
    corr_ep_main = np.zeros((n,snap))
    e = Energy_Toda_Vectorized(p,r)
    e0 = e[:,0]

    for j in range(n):
        corr_p_main += p0[j]*np.roll(p,-j,axis = 0)
        corr_r_main += r0[j]*np.roll(r,-j,axis = 0)
        corr_e_main += e0[j]*np.roll(e,-j,axis = 0)

        corr_pr_main += r0[j]*np.roll(p,-j,axis = 0)
        corr_er_main += r0[j]*np.roll(e,-j,axis = 0)
        corr_ep_main += p0[j]*np.roll(e,-j,axis = 0)
    
    return (corr_p_main/n,corr_r_main/n,corr_e_main/n,corr_pr_main/n,corr_er_main/n,corr_ep_main/n)


#################################################
#################################################
################# MAIN ##########################
#################################################
#################################################

if (len(sys.argv) < 8):
    print('error: give n snap tfinal trials beta eta  numbering')
    exit()

# PARAMETERS FOR EVOLUTION FROM COMMAND LINE
n = int(sys.argv[1])
snap = int(sys.argv[2])
tfinal = float(sys.argv[3])
trials = int(sys.argv[4])
beta = float(sys.argv[5])
eta = float(sys.argv[6])
numbering = int(sys.argv[7])

time_snap = np.linspace(0,tfinal,num = snap)

mid_particle = int(n*0.5)


# correlation vectors
corr_p = np.zeros((n,snap)) 
corr_r = np.zeros((n,snap))
corr_e = np.zeros((n,snap))
corr_pr = np.zeros((n,snap)) 
corr_er = np.zeros((n,snap))
corr_ep = np.zeros((n,snap))
corr_p_main = np.zeros((n,snap))
corr_r_main = np.zeros((n,snap))
corr_e_main = np.zeros((n,snap))
corr_pr_main = np.zeros((n,snap))
corr_er_main = np.zeros((n,snap))
corr_ep_main = np.zeros((n,snap))

# mean values, can be computed exactly 

mean = [0,np.log(beta) - Polygamma(0,eta), (1+2*eta)/(2*beta)]

for k in range(trials):
    # Evolution
    p0,r0 = sample_general_gibbs(n,beta,eta)
    p,r = evolution_periodic(p0,r0,time_snap)

    # Correlation
    corr_p_main_tmp, corr_r_main_tmp, corr_e_main_tmp,corr_pr_main_tmp,corr_er_main_tmp,corr_ep_main_tmp = correlation_toda(p,r,p0,r0)
    corr_p_main += corr_p_main_tmp
    corr_r_main += corr_r_main_tmp
    corr_e_main += corr_e_main_tmp
    corr_pr_main += corr_pr_main_tmp
    corr_er_main += corr_er_main_tmp
    corr_ep_main += corr_ep_main_tmp


# actual correlation
corr_p = np.roll(corr_p_main/trials - mean[0]**2,mid_particle,axis = 0)
corr_r = np.roll(corr_r_main/trials - mean[1]**2,mid_particle,axis = 0)
corr_e = np.roll(corr_e_main/trials - mean[2]**2,mid_particle,axis = 0)
corr_pr = np.roll(corr_pr_main/trials - mean[0]*mean[1],mid_particle,axis = 0)
corr_er = np.roll(corr_er_main/trials - mean[2]*mean[1],mid_particle,axis = 0)
corr_ep = np.roll(corr_ep_main/trials - mean[0]*mean[2],mid_particle,axis = 0)


#save file

np.savetxt('Data/Toda_p_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f_%05d.dat' %(n,beta,eta,time_snap[-1],numbering), corr_p)
np.savetxt('Data/Toda_r_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f_%05d.dat' %(n,beta,eta,time_snap[-1],numbering), corr_r)
np.savetxt('Data/Toda_e_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f_%05d.dat' %(n,beta,eta,time_snap[-1],numbering), corr_e)
np.savetxt('Data/Toda_pr_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f_%05d.dat' %(n,beta,eta,time_snap[-1],numbering), corr_pr)
np.savetxt('Data/Toda_er_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f_%05d.dat' %(n,beta,eta,time_snap[-1],numbering), corr_er)
np.savetxt('Data/Toda_ep_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f_%05d.dat' %(n,beta,eta,time_snap[-1],numbering), corr_ep)



if numbering == 0:
    np.savetxt('Data/Toda_timesnap_n_%d_beta_%0.1f_eta_%0.1f_time_%0.1f.dat' %(n,beta,eta,time_snap[-1]), time_snap )



