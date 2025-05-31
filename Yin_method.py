from scipy import signal as sg
import numpy as np

def f(x):
    f_0=1
    envelope=lambda x: np.exp(-x)
    return np.sin(x * 2 * np.pi * f_0) *envelope(x)
def ACF(f,W,t,lag):
   return np.sum(f[t:t+W]*f[lag+t:t+W+lag])
def detection_pitch(f,W,t,sr,bounds):
    ACF_vals=[ACF(f,W,t,i) for i in range(*bounds)]
    sample=np.argmax(ACF_vals)+bounds[0]
    return sr/sample
def DF(f,W,t,lag):
    return ACF(f,W,t,0) + ACF(f,W,t+lag,0) - 2*ACF(f,W,t,lag)
def CMNDF(f,W,t,lag):
    if lag==0:
        return 1
    return lag*DF(f,W,t,lag)/np.sum([DF(f,W,t,i+1) for i in range(lag)])
def detection_pitch_df(f,W,t,sr,bounds,treshold):
    CMNDF_vals=[CMNDF(f,W,t,i) for i in range(*bounds)]
    sample=None
    for i,val in enumerate(CMNDF_vals):
        if val<treshold:
            sample=i+bounds[0]
    if sample is None:        
        sample=np.argmin(CMNDF_vals)+bounds[0]
    return sr/sample
def main():
    sample_rate=500
    start=0
    end=5
    num_sample =int(sample_rate*(end-start)+1)
    window_size=200
    bounds=[20,num_sample//2]
    x=np.linspace(start,end,num_sample)
    print(detection_pitch_df(f(x),window_size,1,sample_rate,bounds))
    
if __name__ == "__main__":
    main()