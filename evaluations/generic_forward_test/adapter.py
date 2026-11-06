"""Native scalar-wave inversion adapter. No wavefield is ever a torch tensor.

objective_gradient exposes the explicit receiver-leaf boundary.
differentiable_data exposes identical native backward to PyTorch callers.
Only first-order derivatives are supported (no double backward).
"""
from pathlib import Path
import ctypes as ct
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
P = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
I = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')


def coefficients(model, coordinate):
    if coordinate == 'c2': return model
    if coordinate == 'velocity': return model.square()
    if coordinate == 'log_velocity': return torch.exp(2*model)
    raise ValueError('supported coordinates: c2, velocity, log_velocity')


class NativeWave:
    def __init__(self, nx, nt, dt, dx, damping, source, source_index, receivers):
        self.nx, self.nt, self.dt, self.dx = int(nx), int(nt), float(dt), float(dx)
        self.damping=np.ascontiguousarray(damping,dtype=np.float64)
        self.source=np.ascontiguousarray(source,dtype=np.float64)
        raw=np.asarray(receivers)
        if raw.ndim != 1 or not np.all(raw == raw.astype(np.int64)):
            raise ValueError('receivers must be 1-D integer indices')
        self.receivers=np.ascontiguousarray(raw,dtype=np.int32)
        self.si=int(source_index);self.nr=len(self.receivers)
        if self.nx<5 or self.nt<1 or self.dt<=0 or self.dx<=0: raise ValueError('invalid mesh/time')
        if self.damping.shape!=(self.nx,) or self.source.shape!=(self.nt,): raise ValueError('array shape')
        if not np.isfinite(self.damping).all() or not np.isfinite(self.source).all(): raise ValueError('nonfinite settings')
        if not 1<=self.si<self.nx-1 or not self.nr or np.any(self.receivers<0) or np.any(self.receivers>=self.nx): raise ValueError('geometry')
        ext='dylib' if __import__('sys').platform=='darwin' else 'so'
        self.lib=ct.CDLL(str(ROOT/('libadjoint.'+ext)))
        self.baseline=ct.CDLL(str(ROOT/('libnative.'+ext)))
        ftypes=[ct.c_int,ct.c_int,ct.c_double,ct.c_double,P,P,P,ct.c_int,I,ct.c_int,P,ct.c_void_p]
        for fn in (self.lib.wave_extended_forward,self.baseline.wave_forward):
            fn.argtypes=ftypes;fn.restype=ct.c_int
        self.lib.wave_adjoint.argtypes=[ct.c_int,ct.c_int,ct.c_double,ct.c_double,P,P,I,ct.c_int,P,P,P]
        self.lib.wave_adjoint.restype=None
        self.lib.wave_tangent.argtypes=self.lib.wave_adjoint.argtypes
        self.lib.wave_tangent.restype=None
        self.lib.wave_sample.argtypes=[ct.c_int,ct.c_int,I,ct.c_int,P,P]
        self.lib.wave_sample.restype=None
        self.lib.wave_sample_transpose.argtypes=self.lib.wave_sample.argtypes
        self.lib.wave_sample_transpose.restype=None
        self.reset_counts()

    def reset_counts(self):
        self.counts={'forward':0,'adjoint':0,'replay':0,'tangent_check_only':0,'parameter_fd_production':0,'temporal_ad':False}

    def model_array(self,c2):
        c2=np.ascontiguousarray(c2,dtype=np.float64)
        if c2.shape!=(self.nx,) or not np.isfinite(c2).all() or np.any(c2<=0): raise ValueError('c2 must be positive finite nx array')
        if self.dt*np.sqrt(c2).max()/self.dx>=1: raise ValueError('conservative CFL requires max(v)*dt/dx < 1')
        return c2

    def forward(self,c2,history=True,baseline=False):
        c2=self.model_array(c2)
        d=np.empty((self.nt,self.nr),dtype=np.float64)
        hist=np.empty((self.nt+1,self.nx),dtype=np.float64) if history else None
        fn=self.baseline.wave_forward if baseline else self.lib.wave_extended_forward
        code=fn(self.nx,self.nt,self.dt,self.dx,c2,self.damping,self.source,self.si,self.receivers,self.nr,d,None if hist is None else hist.ctypes.data_as(ct.c_void_p))
        if code: raise RuntimeError(f'native forward status {code}')
        self.counts['forward']+=1
        return d,hist

    def adjoint(self,c2,history,cotangent):
        c2=self.model_array(c2)
        history=np.ascontiguousarray(history,dtype=np.float64)
        cotangent=np.ascontiguousarray(cotangent,dtype=np.float64)
        if history.shape!=(self.nt+1,self.nx) or cotangent.shape!=(self.nt,self.nr): raise ValueError('history/cotangent shape')
        if not np.isfinite(history).all() or not np.isfinite(cotangent).all(): raise ValueError('nonfinite history/cotangent')
        gradient=np.empty(self.nx,dtype=np.float64)
        self.lib.wave_adjoint(self.nx,self.nt,self.dt,self.dx,c2,self.damping,self.receivers,self.nr,history,cotangent,gradient)
        self.counts['adjoint']+=1
        return gradient

    def tangent(self,c2,history,direction):
        """Check-only analytic JVP, kept out of the production API route."""
        c2=self.model_array(c2)
        d=np.empty((self.nt,self.nr),dtype=np.float64)
        self.lib.wave_tangent(self.nx,self.nt,self.dt,self.dx,c2,self.damping,self.receivers,self.nr,np.ascontiguousarray(history),np.ascontiguousarray(direction),d)
        self.counts['tangent_check_only']+=1
        return d

    def sample(self,history):
        out=np.empty((self.nt,self.nr),dtype=np.float64)
        self.lib.wave_sample(self.nx,self.nt,self.receivers,self.nr,np.ascontiguousarray(history,dtype=np.float64),out)
        return out

    def scatter(self,bar):
        out=np.empty((self.nt+1,self.nx),dtype=np.float64)
        self.lib.wave_sample_transpose(self.nx,self.nt,self.receivers,self.nr,np.ascontiguousarray(bar,dtype=np.float64),out)
        return out

    def objective_gradient(self,model,objective,coordinate='c2'):
        """One native forward -> receiver AD -> native adjoint -> static VJP."""
        model=torch.as_tensor(model,dtype=torch.float64).detach().clone().requires_grad_(True)
        if coordinate in ('c2','velocity') and torch.any(model<=0): raise ValueError('positive physical coordinates required')
        c2=coefficients(model,coordinate)
        c2_np=c2.detach().numpy().copy()
        data,history=self.forward(c2_np)
        leaf=torch.from_numpy(data).detach().requires_grad_(True)
        value=objective(leaf)
        if value.ndim or not torch.isfinite(value): raise ValueError('objective must be finite scalar')
        cotangent,=torch.autograd.grad(value,leaf)
        gc2=self.adjoint(c2_np,history,cotangent.detach().numpy())
        gm,=torch.autograd.grad(c2,model,torch.from_numpy(gc2))
        return float(value.detach()),gm.detach().numpy(),{'data':data,'history':history,'data_cotangent':cotangent.detach().numpy(),'gradient_c2':gc2}

    def differentiable_data(self,model,coordinate='c2'):
        if model.device.type!='cpu' or model.dtype!=torch.float64: raise ValueError('CPU float64 required')
        if coordinate in ('c2','velocity') and torch.any(model<=0): raise ValueError('positive physical coordinates required')
        return _NativeC2.apply(coefficients(model,coordinate),self)


class _NativeC2(torch.autograd.Function):
    @staticmethod
    def forward(ctx,c2,solver):
        ctx.solver=solver
        ctx.c2=c2.detach().numpy().copy()
        data,ctx.history=solver.forward(ctx.c2)
        return torch.from_numpy(data)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,bar):
        gc2=ctx.solver.adjoint(ctx.c2,ctx.history,bar.detach().contiguous().numpy())
        return torch.from_numpy(gc2),None


def objectives(observed):
    """Return unweighted discrete L2 and nonlinear processed-data objectives.

    Processed: receiver mixing including duplicates, trace-wise differentiable
    RMS normalization, tanh, fixed time convolution, decimation, real FFT
    smooth magnitude; target uses precisely the same processing.
    """
    obs=torch.as_tensor(observed,dtype=torch.float64)
    nr=obs.shape[1]
    mix=torch.eye(nr,dtype=torch.float64)*0.8+torch.ones((nr,nr),dtype=torch.float64)*0.2/nr
    kernel=torch.tensor([.1,.2,.4,.2,.1],dtype=torch.float64).reshape(1,1,5)
    def process(d):
        z=d@mix
        z=z/torch.sqrt(.05+torch.mean(z.square(),dim=0,keepdim=True))
        z=torch.tanh(.8*z)
        z=torch.nn.functional.conv1d(z.T.unsqueeze(1),kernel,padding=2).squeeze(1).T[::2]
        spectrum=torch.fft.rfft(z,dim=0,norm='ortho')
        mag=torch.sqrt(spectrum.real.square()+spectrum.imag.square()+1e-5)
        return z,mag
    target_z,target_mag=process(obs)
    def l2(d): return .5*torch.sum((d-obs).square())
    def processed(d):
        z,mag=process(d)
        return .5*torch.sum((z-target_z).square())+.07*torch.sum((mag-target_mag).square())
    return {'l2':l2,'processed':processed}
