import numpy as np

def sigm(x, delta=0.1):
    return (1-delta)/(1+np.exp(-x)**0.5) + delta

def np_softmax(matrix):
    return (np.exp(matrix)/np.sum(np.exp(matrix), axis=0))

def generate_matrix(N_tags, N_time_frames, cur_tag, spike_at=30):
    initial_matrix = np.ones((N_tags, N_time_frames))/N_tags
    # change true tag probabilities to sigmoid
    sigma_time = np.arange(N_time_frames)-spike_at # timeline
    initial_matrix[::,::] = 1-sigm(sigma_time, delta=1-1/N_tags) # wrong labels
    initial_matrix[cur_tag] = sigm(sigma_time, delta=1/N_tags) # correct labels
    return np_softmax(np.log(initial_matrix))

def generate_bell(N_tags, N_time_frames, cur_tag, start=30, stop=90):
    initial_matrix = np.ones((N_tags, N_time_frames))/N_tags
    # change true tag probabilities to sigmoid
    sigma_time1 = np.arange(N_time_frames)-start # timeline upwards
    sigma_time2 = np.arange(N_time_frames)-stop # timeline downwards
    
    up_wrong = 1-sigm(sigma_time1, delta=1-1/N_tags) # wrong labels upwards bell
    down_wrong = -1 + 1/N_tags + sigm(sigma_time2, delta=1-1/N_tags) # wrong labels downwards bell
    
    up_right = sigm(sigma_time1, delta=1/N_tags) # correct labels upwards bell
    down_right = 1 + 1/N_tags - sigm(sigma_time2, delta=1/N_tags) # correct labels downwards bell
    mask = np.ones(N_time_frames)
    mask[N_time_frames//2:] = 0
    
    initial_matrix[::,::] = up_wrong*mask + down_wrong*mask[::-1]
    initial_matrix[cur_tag] = up_right*mask + down_right*mask[::-1]
    
    softmax_bell = np_softmax(np.log(initial_matrix))
    return softmax_bell

