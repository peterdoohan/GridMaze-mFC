from pyControl.utility import *
from devices import Grid_maze_7x7, Rsync

maze = Grid_maze_7x7()
sync_output = Rsync(pin=maze.BNC_1, mean_IPI=5000)

# States and events
states = ['cue',        # Period between goal cue and goal nose poke 
          'poked_in',   # Successfull nose poke at goal port
          'reward',     # Delivery of water reward at goal port
          'ITI']        # Inter-trial interval (period between trials)

events = maze.events + [                # Pre-programmed maze events
         'reward_consumption_timer',    # Time mice can be poked out of a reward port before moving into the ITI
         'cue_noise_off_timer',         # How long the cue nose is played
         'reward_duration_timer',       # Timer for the amount of reward delivered
         'session_timer',               # Total session time
         'rsync']                       # Camera sync pulse

# Starting state
initial_state = 'ITI'

# Maze config vector: 100111101110101010110001101011101011101110111110111000010100010111011001001110111111

# Variables
v.goal_set = ['A2','A6','B4',
              'C1','C3','C7','D3',
              'E4','E5',
              'F2','F7','G4']                # Set of active goals on the 7x7 maze
v.goal_sampler = sample_without_replacement(v.goal_set)     # Sample goals without replacement (repeats when empty)    
v.n_trials = 0                                              # Number of trials completed
v.current_goal = None                                       # Current goal from v.goal_set being cued 
v.trial_start_time = 0                                      # Time port is cued at the start of the trial 
v.trial_end_time = 0                                        # Time mouse has finished geting the reward and enters ITI
v.error_poke_list = []                                      # List of incorrect poke locations for each trial

#Parameters
v.session_duration = 40                     # Duration of entire session
v.audio_cue_duration = 0.5 * second         # Duration of audio tone during goal cueing 
v.min_ITI_dur = 4                           # Minimum ITI duration (seconds).
v.max_ITI_dur = 8                           # Maximum ITI duration (seconds).
v.reward_consumption_dur = 0.5 * second     # Timer for drinking breaks during reward state
v.reward_duration = 100                     # Time solenoid allows water flow, duration of reward delivery (ms)
v.max_reward_state_dur = 10 * second        # Timer for max drinking time during reward state

# Define behaviours. 
def run_start():
    maze.audio.set_volume(10)
    set_timer('session_timer', v.session_duration*minute)

# This state sets a randomised time interval (between min_ITI_dur and max_ITI_dur) between successive trials and resets counter variables
def ITI(event):
    if event == 'entry':
        v.error_poke_list = []
        timed_goto_state('cue', randint(v.min_ITI_dur,v.max_ITI_dur)*second)

# This state chooses a random goal port from the goal set (without replacement), turns on the LED at this port and plays whitenose at the goal port
# for a duration set by 'cue_noise_duration'. If mice poke in the correct goal they are sent to the poked_in state, if they poke a non-goal port
# an error is recorded
# It also prints start trial information: T = trial no., S = current goal port, sT = start time 
def cue(event):
    if event == 'entry':
        v.trial_start_time = get_current_time()
        v.current_goal = v.goal_sampler.next()                   
        maze.LED_on(v.current_goal)                             
        maze.speaker_on(v.current_goal)                         
        maze.audio.noise() 
        print('Start trial - T#:{} S#:{} sT#:{}.').format(v.n_trials, v.current_goal, v.trial_start_time)                                     
        set_timer('cue_noise_off_timer', v.audio_cue_duration)  
    elif event == 'exit':
        maze.speaker_off(v.current_goal)                        
        maze.audio.off()
        maze.LED_off(v.current_goal)       
        disarm_timer('cue_noise_off_timer')
    elif event == 'cue_noise_off_timer':                              
        maze.speaker_off(v.current_goal)                            
        maze.audio.off()
    elif event[-3:] == '_in':                                             
        if event[:2] != v.current_goal: 
            if event[:2] not in v.error_poke_list:                                
                v.error_poke_list.append(event[:2])                                     
        elif event[:2] == v.current_goal:                             
            goto_state('poked_in')                                  

# This state counts successful trials and sends mice to the reward state with a fixed delay of 200ms.
def poked_in(event):                        
    if event == 'entry':
        v.n_trials += 1 
        v.trial_end_time = get_current_time() 
        timed_goto_state('reward', 200)    

# This state activates the solenoid at the goal port once mice have poked into the correct port, for a given duration (reward_duration), delivering
# a fixed amount of reward (see solenoid calibration spreadsheet). Mice are allowed to have drinking breaks where they may poke out of the reward port
# for a max duration set by reward_consumption_timer, once this timer has elapsed or mice reach the max_reward_state_dur they move to the ITI state. 
# Trial variables are also printed here. E = no. incorrect ports visted before goal port (doesn't count multiple pokes into the same incorrect poke). eT = end trial time. D = trial duration 
def reward(event):
    if event == 'entry':
        print('End Trial - T#:{} S:{} eT#:{} E#:{} D:{} '.format(v.n_trials,  v.current_goal, v.trial_end_time, len(v.error_poke_list), v.trial_end_time - v.trial_start_time))    
        maze.SOL_on(v.current_goal)                                     
        set_timer('reward_duration_timer', v.reward_duration)           
        timed_goto_state('ITI', v.max_reward_state_dur)                   
    elif event == 'exit':
        disarm_timer('reward_consumption_timer')
    elif event == 'reward_duration_timer':
        maze.SOL_off(v.current_goal) 
    elif event[-4:] == '_out' and event[:2] == v.current_goal:
        set_timer('reward_consumption_timer', v.reward_consumption_dur)              
    elif event[-3:] == '_in' and event[:2] == v.current_goal:           
        disarm_timer('reward_consumption_timer')
    elif event == 'reward_consumption_timer' or event == 'max_reward_state_durr':
        goto_state('ITI')                              

def all_states(event):
    if event == 'session_timer':
        stop_framework()