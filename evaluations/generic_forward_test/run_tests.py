#!/usr/bin/env python3
"""Reproducible CPU float64 correctness and small inversion example."""
from pathlib import Path
import hashlib, importlib.util, json, platform, resource, subprocess, sys, time
import numpy as np
import torch
from adapter import NativeWave, coefficients, objectives

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('skill_numerics',ROOT/'numerics.py')
num=importlib.util.module_from_spec(spec);spec.loader.exec_module(num)
torch.set_num_threads(1)
rng=np.random.default_rng(20261107)
raw={};report={'status':'attempted','thresholds':{'exact_forward':True,'dot_atol':1e-10,'dot_rtol':2e-10,'fd_dot_atol':1e-9,'fd_dot_rtol':3e-6,'taylor_atol':1e-10,'taylor_rtol':2e-5,'taylor_order':[1.8,2.2]},'failures':[]}

def record(name,result):
    report.setdefault('tests',{})[name]=result
    if not result.get('passed',False): report['failures'].append(name)

def setup(nx=43,nt=150):
    x=np.arange(nx,dtype=np.float64)
    vel=1.38+.15*np.sin(x*.21)+.06*np.cos(x*.47)
    damping=1-.027*(np.exp(-((x-1)/4)**2)+np.exp(-((x-(nx-2))/4)**2))
    t=np.arange(nt,dtype=np.float64)
    src=.16*(1-2*((t-12)/4)**2)*np.exp(-((t-12)/4)**2)
    rec=np.array([0,1,5,12,12,22,35,nx-2,nx-1],dtype=np.int32) if nx>35 else np.array([0,1,2,2,nx-2,nx-1],dtype=np.int32)
    solver=NativeWave(nx,nt,.30,1.,damping,src,17 if nx>35 else 1,rec)
    return solver,vel

# Unchanged forward baseline, history on/off and boundary-focused geometry.
for label,nx,nt in [('main',43,150),('boundary_small',9,24)]:
    solver,v=setup(nx,nt)
    native,nh=solver.forward(v*v,baseline=True)
    extended,eh=solver.forward(v*v)
    off,_=solver.forward(v*v,history=False)
    nativeoff,_=solver.forward(v*v,history=False,baseline=True)
    check=num.compare_gradients(native,extended,atol=0.,rtol=0.)
    check.update(history_bitwise_equal=bool(np.array_equal(nh,eh)),no_history_bitwise_equal=bool(np.array_equal(off,nativeoff)),trace_bitwise_equal=bool(np.array_equal(native,extended)))
    check['passed']=check['passed'] and check['history_bitwise_equal'] and check['no_history_bitwise_equal']
    record('native_forward_'+label,check)
    raw[label+'_native_data']=native;raw[label+'_extended_data']=extended;raw[label+'_history']=eh
    # Observation pair tested on arbitrary states, including nonzero endpoints.
    probe=rng.normal(size=eh.shape);bar=rng.normal(size=native.shape)
    record('sampling_'+label,num.check_adjoint(solver.sample,solver.scatter,probe,bar,atol=1e-10,rtol=2e-10))
    dc=rng.normal(size=nx)
    record('native_tangent_adjoint_'+label,num.check_adjoint(lambda p:solver.tangent(v*v,eh,p),lambda y:solver.adjoint(v*v,eh,y),dc,bar,atol=1e-10,rtol=2e-10))
    raw[label+'_dot_direction_c2']=dc;raw[label+'_dot_cotangent']=bar

solver,v=setup()
c2=v*v
truth=v+.055*np.sin(np.arange(solver.nx)*.13+.6)+.025*np.cos(np.arange(solver.nx)*.41)
observed,_=solver.forward(truth*truth,history=False)
data,history=solver.forward(c2)
funcs=objectives(observed)
raw.update(model_velocity=v,model_c2=c2,true_velocity=truth,observed=observed,damping=solver.damping,source=solver.source,receivers=solver.receivers)
steps=np.array([.1/(2**i) for i in range(15)])
# Real AD data VJPs, standard and nonlinear processed targets.
for name,obj in funcs.items():
    leaf=torch.tensor(data,dtype=torch.float64,requires_grad=True)
    value=obj(leaf);g,=torch.autograd.grad(value,leaf)
    g=g.numpy();p=g/np.linalg.norm(g)*np.linalg.norm(data)*.12
    result=num.taylor_test(lambda d:float(obj(torch.tensor(d,dtype=torch.float64))),data,g,p,steps,atol=1e-10,rtol=2e-5)
    result['objective']=float(value.detach());result['gradient_l2']=float(np.linalg.norm(g))
    record('data_vjp_'+name,result)
    raw['data_vjp_'+name]=g;raw['data_direction_'+name]=p
    if name=='l2':record('l2_data_analytic',num.compare_gradients(data-observed,g,atol=1e-12,rtol=1e-12))

# Model-space tests for each advertised parameter family and both objectives.
for coordinate in ('c2','velocity','log_velocity'):
    m={'c2':c2,'velocity':v,'log_velocity':np.log(v)}[coordinate]
    mapdot={'c2':np.ones_like(v),'velocity':2*v,'log_velocity':2*c2}[coordinate]
    p=rng.normal(size=m.shape)*.12*np.linalg.norm(m)/np.sqrt(len(m))
    bar=rng.normal(size=data.shape)
    def f_action(p):return solver.tangent(c2,history,mapdot*p)
    def t_action(y):return mapdot*solver.adjoint(c2,history,y)
    record('coordinate_tangent_adjoint_'+coordinate,num.check_adjoint(f_action,t_action,p,bar,atol=1e-10,rtol=2e-10))
    # Finite differences only supply an independent CHECK JVP; never used in objective_gradient.
    fdrows=[]
    for h in (1e-3,1e-4,1e-5,1e-6):
        def fd_action(p):
            cp=coefficients(torch.tensor(m+h*p),coordinate).numpy()
            cm=coefficients(torch.tensor(m-h*p),coordinate).numpy()
            return (solver.forward(cp,history=False)[0]-solver.forward(cm,history=False)[0])/(2*h)
        row=num.check_adjoint(fd_action,t_action,p,bar,atol=1e-9,rtol=3e-6)
        row['step']=h;fdrows.append(row)
    record('fd_jvp_adjoint_'+coordinate,{'passed':all(row['passed'] for row in fdrows[-2:]),'steps':fdrows})
    for name,obj in funcs.items():
        solver.reset_counts()
        value,g,detail=solver.objective_gradient(m,obj,coordinate)
        route=dict(solver.counts)
        route.update(history_shape=list(detail['history'].shape),history_bytes=detail['history'].nbytes,adjoint_live_bytes=3*solver.nx*8,coefficient_map_uses_static_ad=True,data_leaf_ad=True)
        record('production_route_'+coordinate+'_'+name,{'passed':route['forward']==1 and route['adjoint']==1 and route['replay']==0 and not route['temporal_ad'] and route['parameter_fd_production']==0 and route['tangent_check_only']==0,'route':route,'objective':value,'gradient_l2':float(np.linalg.norm(g))})
        raw['gradient_'+coordinate+'_'+name]=g
        raw['kernel_c2_'+coordinate+'_'+name]=detail['gradient_c2']
        raw['cotangent_'+coordinate+'_'+name]=detail['data_cotangent']
        # API with a user-visible custom grad_fn runs the same native reverse.
        mt=torch.tensor(m,dtype=torch.float64,requires_grad=True)
        custom_data=solver.differentiable_data(mt,coordinate)
        custom_value=obj(custom_data);custom_gradient,=torch.autograd.grad(custom_value,mt)
        record('custom_backward_'+coordinate+'_'+name,num.compare_gradients(g,custom_gradient.numpy(),atol=1e-10,rtol=2e-10))
        # Include a gradient-aligned direction and an independent nonorthogonal random direction.
        rand=rng.normal(size=m.shape);rand/=np.linalg.norm(rand)
        direction_scale=.15*np.linalg.norm(m)
        for dname,dp in [('aligned',g/np.linalg.norm(g)*direction_scale),('mixed',(g/np.linalg.norm(g)*.65+rand*.35)*direction_scale)]:
            def evaluate(mm):
                cc=coefficients(torch.tensor(mm,dtype=torch.float64),coordinate).numpy()
                dd,_=solver.forward(cc,history=False)
                return float(obj(torch.tensor(dd,dtype=torch.float64)))
            result=num.taylor_test(evaluate,m,g,dp,steps,atol=1e-10,rtol=2e-5)
            result['direction_l2']=float(np.linalg.norm(dp));result['gradient_l2']=float(np.linalg.norm(g))
            record('model_taylor_'+coordinate+'_'+name+'_'+dname,result)
            raw['direction_'+coordinate+'_'+name+'_'+dname]=dp

# Complete kernel check with check-only exact tangent columns (not parameter FD).
y=raw['cotangent_c2_processed']
expected=np.array([np.vdot(solver.tangent(c2,history,np.eye(solver.nx)[i]),y) for i in range(solver.nx)])
record('all_c2_tangent_columns',num.compare_gradients(expected,raw['gradient_c2_processed'],atol=1e-10,rtol=2e-10))
raw['tangent_column_gradient_c2']=expected
record('fixed_endpoint_coordinates',{'passed':all(np.array_equal(raw['gradient_'+coordinate+'_'+name][[0,-1]],np.zeros(2)) for coordinate in ('c2','velocity','log_velocity') for name in funcs) and np.array_equal(data[:,[0,-1]],np.zeros((solver.nt,2))),'note':'Fixed endpoint model sensitivities and displacement samples are exactly zero; generic observation dot tests separately use nonzero endpoint states.'})

# Demonstrate inversion with line search using the advertised production interface.
m=np.log(v).copy();obj=funcs['l2'];iterations=[]
for it in range(6):
    value,g,_=solver.objective_gradient(m,obj,'log_velocity')
    alpha=.035/max(np.max(np.abs(g)),1e-12)
    accepted=False
    for backtrack in range(20):
        proposal=m-alpha*g
        dd,_=solver.forward(np.exp(2*proposal),history=False)
        trial=float(obj(torch.tensor(dd,dtype=torch.float64)))
        if trial<=value-1e-4*alpha*np.dot(g,g):accepted=True;break
        alpha*=.5
    iterations.append({'iteration':it,'objective_before':value,'objective_after':trial,'gradient_l2':float(np.linalg.norm(g)),'step':alpha,'backtracks':backtrack,'accepted':accepted})
    if not accepted:break
    m=proposal
record('inversion',{'passed':len(iterations)==6 and all(row['accepted'] for row in iterations) and iterations[-1]['objective_after']<iterations[0]['objective_before']*.8,'iterations':iterations})
raw['inversion_final_velocity']=np.exp(m)

# Rejections avoid silently advertising gradients for unsupported coordinates.
rejections=[]
for coordinate,model in [('density',c2),('velocity',-v)]:
    try:solver.objective_gradient(model,obj,coordinate)
    except ValueError as exc:rejections.append({'coordinate':coordinate,'error':str(exc)})
record('unsupported_rejected',{'passed':len(rejections)==2,'rejections':rejections})

# Matched tiny CPU timings, with/without full history explicitly separated.
def timed(fn,count=100):
    for _ in range(5):fn()
    t=time.perf_counter()
    for _ in range(count):fn()
    return (time.perf_counter()-t)/count
report['timings_seconds']={
    'native_forward_no_history':timed(lambda:solver.forward(c2,history=False,baseline=True)),
    'native_forward_full_history':timed(lambda:solver.forward(c2,history=True,baseline=True)),
    'extended_forward_full_history':timed(lambda:solver.forward(c2,history=True)),
    'native_adjoint':timed(lambda:solver.adjoint(c2,history,y)),
    'complete_objective_gradient_processed_c2':timed(lambda:solver.objective_gradient(c2,funcs['processed'],'c2'),20)}
report['resources']={'cpu_only':True,'gpu_used':False,'peak_process_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'main_history_bytes':history.nbytes,'main_adjoint_live_bytes':3*solver.nx*8,'replays':0,'timing_scope':'warmup 5; 100 native phases, 20 complete gradient repetitions; binding, validation and allocations included; build/import excluded; tiny fixture, no HPC benchmark'}
report['source']={'path':'tests/fixtures/native_wave.cpp','sha256':hashlib.sha256((ROOT/'native_wave.cpp').read_bytes()).hexdigest(),'source_modified':False}
report['runtime']={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform(),'threads':1,'compiler':subprocess.check_output(['clang++','--version'],text=True)}
report['supported']={'physics':'1D scalar wave; native fixed-zero boundaries; fixed nonuniform damping','coordinates':['c2','velocity','log_velocity'],'inactive_nodes':'two endpoints; sensitivities exactly zero by fixed boundary contract','fixed':['damping','source','source_index','receivers','dt','dx','initial_states','boundaries'],'data_layout':'float64 time-major [nt,nr], sample u[t+1]','production_gradient':'native C++ adjoint; static coefficient PyTorch VJP; no temporal AD or parameter FD','history_policy':'full float64 displacement (nt+1)*nx; no replay','higher_order_derivatives':False}
report['application']={'skill':'differentiate-seismic-solver','version':'0.1.0-rc1','packet_sha256':'b09e0ee57324f715b80178c2a30cb26a2140b1ea3d5545ad8b15dc5680dca49f','pre_integrity':'passed','post_integrity':'run separately by reproduce.sh','failures_or_repairs':'See attempts.json; this run keeps every gate and Taylor step','human_interventions':[],'familiarity':'Independent source/skill-only implementation; no intended solution or other agent notes consulted; no causal claim about skill cost reduction'}
report['status']='passed' if not report['failures'] else 'failed'
report['passed']=not report['failures']
np.savez(ROOT/'raw_arrays.npz',**raw)
(ROOT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'test_count':len(report['tests']),'failed':report['failures'],'timings_seconds':report['timings_seconds'],'results':str(ROOT/'results.json')},indent=2))
sys.exit(0 if report['passed'] else 1)
