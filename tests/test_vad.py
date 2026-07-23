import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from vad import EnergyVAD, SpectralVAD
from vad.utils import merge_segments, segments_to_mask, mask_to_segments, remove_short, ensure_sr

def _synth(dur=3.0, sr=16000):
    n = int(dur * sr)
    t = np.linspace(0, dur, n)
    a = np.random.randn(n) * 0.005
    s, e = int(1.0*sr), int(2.5*sr)
    a[s:e] += 0.3*np.sin(2*np.pi*200*t[s:e])
    a[s:e] += 0.15*np.sin(2*np.pi*1200*t[s:e])
    return a/(np.max(np.abs(a))+1e-10)

def test_energy_vad():
    assert len(EnergyVAD()(_synth())) >= 1
def test_energy_empty():
    assert EnergyVAD()(np.array([],dtype=np.float32)) == []
def test_spectral_vad():
    assert len(SpectralVAD()(_synth())) >= 1
def test_spectral_empty():
    assert SpectralVAD()(np.array([],dtype=np.float32)) == []
def test_merge():
    assert merge_segments([]) == []
    m = merge_segments([(0,1),(1.3,2)], max_silence=0.5)
    assert len(m) == 1
def test_short():
    r = remove_short(np.array([1,1,0,1,1],bool), 3, False)
    assert r[2] == True
def test_sr():
    assert len(ensure_sr(np.array([1,2,3]),16000,16000)) == 3
def test_mask():
    m = segments_to_mask([(1,2)], 3, 160, 16000)
    assert m.sum() > 0
def test_roundtrip():
    segs = [(1,2),(3,4.5)]
    m = segments_to_mask(segs, 5, 160, 16000)
    b = mask_to_segments(m, 160, 16000)
    for (s1,e1),(s2,e2) in zip(segs,b):
        assert abs(s1-s2) < 0.02

def test_sr_mismatch():
    assert isinstance(EnergyVAD(sr=16000)(_synth(sr=8000), sr=8000), list)