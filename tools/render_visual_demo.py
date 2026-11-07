#!/usr/bin/env python3
"""Render the portable helper against saved, real compiled-wave fixture outputs."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--alignment-auditor",type=Path)
args=parser.parse_args()
source=root/"evaluations/generic_forward_test/raw_arrays.npz"
report=json.loads((source.parent/"results.json").read_text())
out=root/"docs/visual-demo";out.mkdir(parents=True,exist_ok=True)
with np.load(source,allow_pickle=False) as raw:
    data={k:raw[k] for k in ("model_velocity","true_velocity","main_history","main_native_data","observed","gradient_velocity_l2","gradient_velocity_processed","data_vjp_processed")}
data["x"]=np.arange(data["model_velocity"].size,dtype=float)
data["residual"]=data["main_native_data"]-data["observed"]
for kind in ("l2","processed"):
    rows=report["tests"][f"model_taylor_velocity_{kind}_aligned"]["steps"]
    data[f"h_{kind}"]=np.array([r["step"] for r in rows])
    data[f"e_{kind}"]=np.array([r["first_order_residual"] for r in rows])
np.savez(out/"arrays.npz",**data)
def curve(label,series,ylabel):
    return dict(kind="curves",label=label,series=series,xlabel="Position (grid units)",ylabel=ylabel)
def line(x,y,label):return dict(x=x,y=y,label=label)
model=curve("Actual target and starting model",[line("x","true_velocity","Target"),line("x","model_velocity","Starting")],"Velocity (simulation units)")
wave=dict(kind="map",key="main_history",label="Native displacement history",units="Displacement (simulation units)",extent=[0,42,45,0],xlabel="Position (grid units)",ylabel="Time (simulation units)",aspect="auto",symmetric=True)
gradient=curve("Raw native-adjoint gradient",[line("x","gradient_velocity_l2","L2 objective")],"Derivative with respect to velocity")
caption="Actual arrays from the compiled 1-D scalar-wave development fixture. The starting model is not called blurred: no smoothing was used. The history includes the zero initial state. The raw gradient is data AD followed by the native C++ adjoint and a static material-map VJP; no display smoothing or clipping. This is a workflow example, not a geophysical accuracy or HPC benchmark."
figures=[dict(name="visual-example",title="Model, native state and derivative are visible deliverables",caption=caption,columns=3,panels=[model,wave,gradient])]
panels=[]
for key,label in (("observed","Target data"),("main_native_data","Starting-model prediction"),("residual","Prediction minus target"),("data_vjp_processed","Processed-objective data VJP")):
    p=dict(kind="map",key=key,label=label,units="Discrete data units" if key!="data_vjp_processed" else "Objective derivative / data unit",extent=[0,9,45,.3],xlabel="Receiver index (duplicates included)",ylabel="Time (simulation units)",aspect="auto",symmetric=True)
    if key in ("observed","main_native_data"):p["scale_group"]="receiver_data"
    panels.append(p)
figures.append(dict(name="data-boundary",title="The actual AD cotangent reaches the native boundary",caption="Actual target/prediction, residual and processed-objective receiver derivative. Cotangent and residual have separately labelled scales; they are different quantities.",columns=2,panels=panels))
panels=[]
for kind in ("l2","processed"):
    panels.append(dict(kind="curves",label=f"Velocity Taylor check: {kind}",series=[line(f"h_{kind}",f"e_{kind}","First-order Taylor remainder")],xlabel="Step size",ylabel="Absolute remainder",xscale="log",yscale="log"))
figures.append(dict(name="gradient-verification",title="Raw multi-epsilon checks remain inspectable",caption="Every retained positive first-order remainder in the actual aligned-velocity Taylor sweep is plotted, including the smallest steps. No step selection or fitted rescaling.",columns=2,panels=panels))
config={"figures":figures};(out/"figures.json").write_text(json.dumps(config,indent=2)+"\n")
spec=importlib.util.spec_from_file_location("seismic_visualize",root/"skills/differentiate-seismic-solver/scripts/visualize.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.render(data,config,out,args.alignment_auditor)
result["source"]={"kind":"saved compiled-native fixture outputs","input_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"input":"evaluations/generic_forward_test/raw_arrays.npz","native_source_sha256":"07bf9bfa81994d2f08febcfaf5a586210ddd1d9f01012e38a5b22eab77afe578","new_solver_runs":0}
(out/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
print(out)
