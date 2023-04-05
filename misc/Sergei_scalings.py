import numpy as np

# function to online 
def shift_scale_sample_4d(data, copy_data=True, do_flip=True):
    P_SHOULDER_LEFT, P_SHOULDER_RIGHT = 11, 12
    P_HIP_LEFT, P_HIP_RIGHT = 23, 24
    K_HEIGHT_TO_WIDTH = 1
    
    if copy_data:
        data = data.copy()
    
    # 0) flip Y
    if do_flip:
        data[:, :, :, 1] *= -1

    # 1) shifting
    origin_shoulders = 0.5*(data[:, :, P_SHOULDER_LEFT:P_SHOULDER_LEFT+1, :2] + data[:, :, P_SHOULDER_RIGHT:P_SHOULDER_RIGHT+1, :2] )
    # origin_hips = 0.5*(data[:, :, P_HIP_LEFT:P_HIP_LEFT+1, :2] + data[:, :, P_HIP_RIGHT:P_HIP_RIGHT+1, :2] )
    data = data - origin_shoulders

    # 2) scaling
    kx = 1 / data[:, :, P_SHOULDER_LEFT:P_SHOULDER_LEFT+1, 0]
    origin_hips_new = 0.5*(data[:, :, P_HIP_LEFT:P_HIP_LEFT+1, :2] + data[:, :, P_HIP_RIGHT:P_HIP_RIGHT+1, :2] )
    # origin_shoulders_new = 0.5*(data[:, :, P_SHOULDER_LEFT:P_SHOULDER_LEFT+1, :2] + data[:, :, P_SHOULDER_RIGHT:P_SHOULDER_RIGHT+1, :2] )

    ky = K_HEIGHT_TO_WIDTH / origin_hips_new[:, :, :, 1]
    data[:, :, :, 0] *=  kx
    data[:, :, :, 1] *=  -ky
    
    return data

# Tensor start stop alignment
def align_tensor_by_se(tensor_data, se_list, s_e_new=[5, 65], filling='fl'):
    
    good_fillings = ['fl', 'first/last', 'roll']
    assert filling in good_fillings, f"Error! filling={filling} should be in {good_fillings}!"
    
    f_tensor = tensor_data.copy()
    samples, duration, signals = f_tensor.shape
    
    s_e_new = np.asarray(s_e_new).reshape(-1, 2)
    
    if s_e_new.shape[0] == 1:
        s_e_new = np.repeat(s_e_new, samples, axis=0)
    elif s_e_new.shape[0] == samples:
        pass
    else:
        assert 0, f"Error, s_e_new.shape={s_e_new.shape} in np.array from"\
                  f" should have 1 or {samples} size of the first dimension"\
                  f"for given 'tensor_data'[{tensor_data.shape}]"
    
    print(s_e_new.shape)
    se_array = np.asarray(se_list)
    t = np.arange(0, duration)
    t_tensor = np.repeat(t.reshape(1, -1), samples, axis=0)
    
    
    k_stretching =  (s_e_new [:,1:2] - s_e_new [:,0:1])/(se_array[:,1:2] - se_array[:,0:1])   
    t_tensor_  = (t_tensor -  se_array[:, 0:1])*k_stretching + s_e_new[:,0:1]
    
    
    # Skipping the null procedures
    mask = (np.round(s_e_new,0) == np.round(se_array,0)).prod(axis=1).astype(bool)
    # print(mask)
    t_tensor_[mask] = t_tensor[mask]
    print(f"{mask.sum()} procedures (from {mask.size}) were skipped!")
    
    t_tensor = t_tensor_
    
    interp_period = None
    if filling in ['fl', 'first/last']:
        interp_period = None
    elif filling in ['roll']:
        interp_period = duration
    else:
        print(f"Warning! activity for 'filling'={filling} is not defined! Default: interp_period=None!!")
        
    
    
    for k in range(signals):
        for i in range(samples):
            f = f_tensor[i, :, k]
            t = t_tensor[i, :]
            f_new = np.interp(np.arange(0, duration),t, f, period=interp_period)
            f_tensor[i, :, k] = f_new
            
    return f_tensor