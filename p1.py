#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 09:41:40 2020

@author: shiva
"""
import math as mt
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile as wf
from scipy.signal import freqz,lfilter,tf2zpk
from scipy import signal as sig

data = wf.read('s1ofwb.wav')
Fs = data[0]
audio = data[1].copy()/32767
auio = audio

fsize=0.03
fl=round(Fs*fsize)
m=10
N=fl-1
hamming = np.hamming(fl)
def lpc(y, m):
    R = [y.dot(y)] 
    if R[0] == 0:
        return [1] + [0] * (m-2) + [-1]
    else:
        for i in range(1, m + 1):
            r = y[i:].dot(y[:-i])
            R.append(r)
        R = np.array(R)
    
        A = np.array([1, -R[1] / R[0]])
        E = R[0] + R[1] * A[1]
        for k in range(1, m):
            if (E == 0):
                E = 10e-17
            alpha = - A[:k+1].dot(R[k+1:0:-1]) / E
            A = np.hstack([A,0])
            A = A + alpha * A[::-1]
            E *= (1 - alpha**2)
        return A

def inverse_filter(window,  coefficients):
    
    denominator = np.asarray([1])
    numerator = np.insert(coefficients, 0, 1)
    return sig.lfilter(numerator, denominator, window)
c=[]
errors=[]
for i in range(0,len(auio)-fl,fl):
        y1=auio[i:i+fl]
        y1=y1*hamming
        y1 = np.append(y1[0], y1[1:] - 0.9378 * y1[:-1])
        y=y1
        c.append(lpc(y,m))

c=np.array(c)            
for i in range(0,len(auio)-fl,fl):
    y1=auio[i:i+fl]
    y = np.append(y1[0], y1[1:] - 0.9378 * y1[:-1])
    errors.append(inverse_filter(y, c))
errors=np.array(errors).reshape(46560,1)
def msf(y):
    b, a = sig.butter(9, .33, 'low')
    y1=lfilter(b,a,y)
    msf=sum(abs(y1))
    return msf
def zc(y):
    zcr=0
    for i in range(0,len(y)):
        if i+1==len(y):
            break
        zcr=zcr+np.multiply((1/2),abs(np.sign(y[i+1])-np.sign(y[i])))
    return zcr

def findex(corr1,rmax):
    rmin=0
    for i in range(1,len(corr1)):
        if(corr1[i]==rmax):
            rmin =i
        else:
            pass
    return rmin


def pitchper(y,Fs):
    pmin=round(Fs*0.002)
    pmax=round(Fs*0.02)
    autocorrelation = sig.fftconvolve(y, y[::-1])
    corr1=autocorrelation
    rmax=max(corr1)
    rmin=findex(corr1,rmax)
    pitchperrange=corr1[rmin+pmin:rmin+pmax]
    rmax=max(pitchperrange)
    rmin=findex(pitchperrange,rmax)
    pitchperiod=rmin+pmin
    return pitchperiod
def voiced(x,fs,size):
    msfv=np.zeros(auio.shape[0])
    zcr=np.zeros(auio.shape[0])
    pitch_plot=np.zeros(auio.shape[0])
    for i in range(0,len(auio)-fl,fl):
        y1=auio[i:i+fl]
        y1=y1*hamming
        y1 = np.append(y1[0], y1[1:] - 0.9378 * y1[:-1])
        msfv[i:i+N]=msf(y1)
        zcr[i:i+N]=zc(y1)
        pitch_plot[i:i+N]=pitchper(errors[i:i+N], Fs)
    thresh_msf = (( (np.sum(msfv)/len(msfv)) - min(msfv)) * (0.67) ) + min(msfv)
    voiced_msf =  msfv > thresh_msf
    thresh_zc = (( ( np.sum(zcr)/len(zcr)) - min(zcr) ) *  (1.5) ) + min(zcr)
    voiced_zc = zcr < thresh_zc
    thresh_pitch = (( (sum(pitch_plot)/len(pitch_plot)) - min(pitch_plot)) * (0.5) ) + min(pitch_plot)
    voiced_pitch= pitch_plot>thresh_pitch            
    voiced=np.zeros(len(x))
    for i in range(0,len(x)-fl):
        if  voiced_msf[i]  *voiced_zc[i] == 1 :
            voiced[i] = 1
        else:
            voiced[i] = 0
    return voiced,pitch_plot


voice,pitch_plot=voiced(auio,Fs,fsize)

def gainer(auio,fl,error,voice,pitch_plot):
    gain=[]
    for i in range(0,len(auio)-fl,fl):
        #y1=auio[i:i+fl]
        ed=error[i:i+fl]
        if voice[i]==0:
            denom=ed.shape[0]
            power=sum(np.multiply(ed[0:denom],ed[0:denom]))/denom
            gain.append(mt.sqrt(power))
        else:
            denom=int((mt.floor(ed.shape[0]/pitch_plot[i]))*pitch_plot[i])
            power=np.sum(np.multiply(ed[0:denom],ed[0:denom]))/denom
            gain.append(mt.sqrt(power*pitch_plot[i]))
    return np.array(gain)
gain=gainer(auio,fl,errors,voice,pitch_plot)
k=[]
def synv(c,gain,pitch_plot,fl,i):
    ptrain=[]
    for f in range(0,fl):
        if f==0:
            ptrain.append(0)
        elif f/pitch_plot[i]== mt.floor(f/pitch_plot[i]):
            
            ptrain.append(1)
        else:
            ptrain.append(0)
    ptrain=np.array(ptrain)
    k.append(ptrain)
    B = np.append([1],c[int(i/fl)][1:11])
    A = np.array([1])
    syn2=sig.lfilter(A,B,ptrain)
    return syn2*gain[int(i/fl)]
def synuv(c,gain,pitch_plot,fl,i):
    wn=np.random.uniform(0,1,size=fl)
    B = np.append([1],c[int(i/fl)][1:11])
    A = np.array([1])
    syn2=sig.lfilter(A,B,wn)
    return syn2*gain[int(i/fl)]        

synthspeech=np.zeros(auio.shape[0]).reshape(auio.shape[0],1)
for i in range(0,len(auio)-fl,fl):
    if voice[i]==1 :
      syny1=synv(c,gain,pitch_plot,fl,i)
      synthspeech=np.append(synthspeech,syny1)
    else:
          syny1=synuv(c,gain,pitch_plot,fl,i)
          synthspeech=np.append(synthspeech,syny1)
synthspeech=synthspeech[auio.shape[0]:(2*auio.shape[0])]  
l=[]
for i in range(0,len(synthspeech)-fl,fl):
    y=synthspeech[i:i+fl]
    b = np.array([1])
    a = np.append([1],[0.9875])
    l.append(sig.lfilter(b,a,y))
l=np.array(l).reshape((len(l)*fl,1))
plt.plot(l)
plt.title('Synthesized from LPC')
plt.show()
plt.title('Original one')
plt.plot(auio)
wf.write("Resynthesized.wav",Fs,l)
plt.show()

#measure
def signaltonoise(a, axis=0, ddof=0):
    mx = np.amax(a)
    a = np.divide(a,mx)
    a = np.square(a)
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
u=signaltonoise(auio,0,0)
o=signaltonoise(l,0,0)


#plots



















center=auio[0:fl]#unvoiced frame
#center=auio[4320:4800]#voiced frame
windowed_signal = center*hamming
windowed1 = windowed_signal.copy()
for i in range(1,len(center)):
    windowed_signal[i] = windowed_signal[i]-(0.937*windowed_signal[i-1])
    
#plt.plot(windowed_signal,label='After Preemphasis')
plt.plot(windowed1,label='Before Preemphasis')
plt.title('Signal before applying pre-emphasis',size='x-large')
plt.ylabel('Amplitude',size='x-large')
plt.xlabel('--->time',size='x-large')
plt.legend(loc='lower right',fontsize='xx-large')
plt.show()    



fft_center = np.fft.fft(windowed1,n=2048)
freq = np.fft.fftfreq(fft_center.shape[0],1/Fs)
plt.plot(freq[:len(freq)//2],20*np.log10(abs(fft_center[:len(freq)//2])),label='DFT before preemphasis')

#fft_center = np.fft.fft(windowed_signal,n=2048)

#plt.plot(freq[:len(freq)//2],20*np.log10(abs(fft_center[:len(freq)//2])),label='DFT after preemphasis',color='r')

#plt.title('DFT using Hamming window near center',size='x-large')
plt.xlabel('frequency',size='x-large')
plt.ylabel('Magnitude(db)',size='x-large')
plt.legend(loc='best',fontsize='xx-large')
plt.show()

g=lpc(windowed_signal,10)
e=inverse_filter(windowed_signal, g)

plt.plot(e)
plt.title('Residue signal')
plt.grid()
plt.show()

corr1 = np.correlate(windowed_signal,windowed_signal,mode='full')
corr1 = corr1[corr1.size//2:]

corr = corr1



plt.plot(corr,'r')
plt.title('Original ACF',size='xx-large')
plt.show()

corres1 = np.correlate(e,e,mode='full')
corres1 = corres1[corres1.size//2:]
corres1 = corres1/max(corres1)
plt.plot(corres1,'g')
plt.title('Residual ACF',size='xx-large')
plt.show()


synthevoic=synv(c, gain, pitch_plot, fl, 4320)
#syntheunvoic=synuv(c, gain, pitch_plot, fl,0)
plt.plot(synthevoic,'r')
plt.title('Synthesized  from LP coefficients',size='xx-large')
plt.show()

fft_center = np.fft.fft(synthevoic,n=2048)
freq = np.fft.fftfreq(fft_center.shape[0],1/Fs)
plt.plot(freq[:len(freq)//2],20*np.log10(abs(fft_center[:len(freq)//2])),label='DFT before preemphasis')

#fft_center = np.fft.fft(windowed_signal,n=2048)

#plt.plot(freq[:len(freq)//2],20*np.log10(abs(fft_center[:len(freq)//2])),label='DFT after preemphasis',color='r')

#plt.title('DFT using Hamming window near center',size='x-large')
plt.xlabel('frequency',size='x-large')
plt.ylabel('Magnitude(db)',size='x-large')
plt.legend(loc='best',fontsize='xx-large')
plt.show()