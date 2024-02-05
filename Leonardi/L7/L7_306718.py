import numpy as np
import matplotlib.pyplot as plt
# from scipy import stats
from collections import namedtuple
from tqdm import tqdm

# constants
REP_NUMB_0 = 4
GAMMA = 1/14
H_PROB = 0.10
IT_PROB = 0.06
FATALITY_PROB = 0.03
MAX_YEARLY_DEATHS = 100_000

Event = namedtuple('Event', ['type','time'])
class FutureEventSet:
    # implementation of the Future Event Set as a priority queue
    def __init__(self):
        self.items = []
    def is_empty(self):
        return len(self.items) == 0
    def put(self, event):
        self.items.append(event)
    def pop(self):
        # pop next event (lowest time) if there is events on the FES
        if not self.is_empty():
            next_event = min(self.items, key=lambda ev:ev.time)
            self.items.remove( next_event )
            return next_event
        print('FutureEventSet is empty.')

def generate_H_IT():
    # return tuple: 
    #   \ (1, 0) with probability H_PROB; 
    #   \ (0, 1) with probability IT_prob; 
    #   \ (0, 0) with probability 1-(H_PROB+IT_prob)
    u = np.random.uniform(0,1)
    if u <= H_PROB:
        return (1, 0)
    if u >= 1-IT_PROB:
        return (0, 1)
    return (0, 0)

def rho_func(occupancy):
    # computes restriction based on hospitals' occupancy
    occupancy = min(occupancy, 1)
    rho = 1 # Level 0
    restriction_time = 0
    if occupancy > 0.5:         # Level 1 of restrictions
        rho -= 0.5
        restriction_time += 21
    if occupancy > 0.7:         # Level 2 of restrictions
        rho -= 0.20
        restriction_time -= 7
    if occupancy > 0.9:         # Level 3 of restrictions
        rho -= 0.20
        restriction_time -= 7
    return rho, restriction_time

def infection(time, sir_state, lam, rho, fes):
    S,I,R = sir_state
    H,IT = generate_H_IT()
    # next infection
    lambda_SI = lam * rho * S * I
    infection_time = time + np.random.exponential( 1/lambda_SI )
    infection_event = Event('infection', infection_time)
    fes.put( infection_event )
    # update SIR state    
    sir_state = (S-1,I+1,R)
    return sir_state, H, IT

def recovery(time, sir_state, fes, occupied_H, occupied_IT):
    S,I,R = sir_state
    # check if infected individual dies (1) or not (0)
    dead = int(np.random.uniform(0,1) <= FATALITY_PROB)
    # check if infected individual was H or IT
    u = np.random.uniform(0,1)
    h_prob = occupied_H / I
    it_prob = occupied_IT / I
    H = - int(u<=h_prob)
    IT = - int(u>=(1-it_prob))
    # next recovery
    gamma_IR = GAMMA * I
    recovery_time = time + np.random.exponential( 1/gamma_IR )
    recovery_event = Event('recovery', recovery_time)
    fes.put( recovery_event )
    # update SIR state    
    sir_state = (S,I-1,R+1)
    return sir_state, H, IT, dead

def update_log(log, S, I, R, H, IT, D, clock=None):
    # append current state to simulation log
    if clock:
        log['time'].append(clock)
    log["S"].append(S)
    log["I"].append(I)
    log["R"].append(R)
    log["H"].append(H)
    log["IT"].append(IT)
    log["D"].append(D)
    return log

def SIR_stochastic(days, N, I_0, max_H, max_IT):
    # stochastic model of SIR epidemic
    # initial state
    S = N - I_0
    I = I_0
    R = 0
    sir_state = (S, I, R)
    lam = REP_NUMB_0 * GAMMA / S
    deaths = 0
    occupied_H = 0
    occupied_IT = 0
    for i in range(I_0):
        # check if first infected individuals occupy H or IT
        H,IT = generate_H_IT()
        occupied_H += H
        occupied_IT += IT
    rho = 1
    restriction_start, restriction_time = 0, 0 
    simulation_log = { "time":[0], "S":[S], "I":[I], "R":[R], "H":[occupied_H], "IT":[occupied_IT], "D":[deaths] }

    fes = FutureEventSet()
    # first infection
    lambda_SI = lam * S * I
    infection_time = 0. + np.random.exponential( 1/lambda_SI )
    infection_event = Event('infection', infection_time)
    fes.put( infection_event )
    # first recovery
    gamma_IR = GAMMA * I
    recovery_time = 0. + np.random.exponential( 1/gamma_IR )
    recovery_event = Event('recovery', recovery_time)
    fes.put( recovery_event )

    event = fes.pop()
    clock = event.time
    # simulation loops
    while clock<=days and I>0:
        # stops if epidemic is erradicated (I=0)
        s = f'\rClock: {clock:.2f} - S={S} - I={I} - R={R} - deaths={deaths} '
        s+= f'- H={occupied_H} - IT={occupied_IT} - rho={rho:.2f}'
        print(s, end='', flush=True)

        if event.type == 'infection':
            sir_state,H,IT = infection(clock, sir_state, lam, rho, fes)
        elif event.type == 'recovery':
            sir_state,H,IT,dead = recovery(clock, sir_state, fes, occupied_H, occupied_IT)
            deaths += dead
        # update occupancy
        occupied_H += H
        occupied_IT += IT
        occupancy = max(occupied_H/max_H, occupied_IT/max_IT)
        # update restriction
        new_rho, new_restriction_time = rho_func( occupancy )
        if new_rho<rho or clock >= restriction_start+restriction_time:
            # mobility restriction increases or restriction time resets
            restriction_start = clock
            restriction_time = new_restriction_time
            rho = new_rho

        S,I,R = sir_state
        simulation_log = update_log(simulation_log, S, I, R, occupied_H, occupied_IT, deaths, clock=clock)
        # next event
        event = fes.pop()
        clock = event.time

    return simulation_log

def SIR_mean_field(days, N, I_0, max_H, max_IT, dt=1):
    # Mean field model of SIR epidemic
    # initial state
    S = N - I_0
    I = I_0
    R = 0
    deaths, deaths_year = 0, 0 
    lam = REP_NUMB_0 * GAMMA / S
    H = I_0 * H_PROB
    IT = I_0 * IT_PROB
    rho = 1
    restriction_start, restriction_time = 0, 0 
    
    times = np.arange(0, days+dt, dt)
    simulation_log = { "time":times, "S":[], "I":[], "R":[], "H":[], "IT":[], "D":[] }

    # Simulation loop
    for t in times:
        if t % 365 == 0:
            deaths += deaths_year
            deaths_year = 0

        # Update simulation log 
        simulation_log = update_log(simulation_log, S, I, R, H, IT, deaths_year)

        # SIR model equations
        dS = -rho * lam * S * I * dt
        dI = (rho * lam * S - GAMMA ) * I * dt
        dI = (rho * lam * S - GAMMA ) * I * dt
        dR = GAMMA * I * dt
        # Update variables
        S += dS
        I += dI
        R += dR
        
        H += dI * H_PROB
        IT += dI * IT_PROB
        deaths_year += dR * FATALITY_PROB
        
        # update occupancy and restriction
        occupancy = max(H/max_H, IT/max_IT)
        new_rho, new_restriction_time = rho_func( occupancy )
        if new_rho<rho or t >= restriction_start+restriction_time:
            # mobility restriction increases and restriction time resets
            restriction_start = t
            restriction_time = new_restriction_time
            rho = new_rho

    # final state
    print(f'Clock: {t:.2f} - S={S:.0f} - I={I:.0f} - R={R:.0f} - ', end='')
    print(f'deaths={deaths:.0f} - H={H:.0f} - IT={IT:.0f} - rho={rho:.2f}')
    return simulation_log

def plot_SIR(simulation_log, title):
    time = np.array(simulation_log['time'])
    S = np.array(simulation_log['S'])
    I = np.array(simulation_log['I'])
    R = np.array(simulation_log['R'])
    H = np.array(simulation_log['H'])
    IT = np.array(simulation_log['IT'])
    D = np.array(simulation_log['D'])

    fig,(ax1,ax2) = plt.subplots(2, sharex=True)
    
    fig.suptitle(title)
    ax1.plot(time, I, label='Infected', color='r',)
    ax1.plot(time, D, label='Dead', color='k')
    ax2.plot(time, H, label='Hospitalized', color='c')
    ax2.plot(time, IT, label='Intensive treatment', color='m')
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0)
    ax1.set_ylabel('Individuals')
    ax2.set_ylabel('Individuals')
    ax2.set_xlabel('Time (days)')
    ax1.grid()
    ax2.grid()
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax2.set_xticks(np.arange(0,time[-1]+1,30))
    return fig

if __name__=='__main__':
    np.random.seed(402892)

    N = 50_000_000
    max_H = 10_000
    max_IT = 5_000
    # N = 10_000
    # max_H = 20
    # max_IT = 10
    I_0 = 10
    days = 365


    print('\n++++ STOCHASTIC SIR ++++')
    simulation_log = SIR_stochastic(days, N, I_0, max_H, max_IT)
    fig1 = plot_SIR(simulation_log, title='Stochastic SIR')
    print('\n++++ MEAN FIELD SIR ++++')
    simulation_log = SIR_mean_field(days, N, I_0, max_H, max_IT)
    fig2 = plot_SIR(simulation_log, title='Mean field SIR')

    plt.show()