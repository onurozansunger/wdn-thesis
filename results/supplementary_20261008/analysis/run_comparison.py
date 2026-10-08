"""Wait for verified extraction, freeze all calibrations, then evaluate."""
from pathlib import Path
import subprocess,sys,time,json
HERE=Path(__file__).resolve().parent;OUT=HERE/"operating_points";LOG=HERE/"logs"
SCRIPT=HERE/"compare_operating_points.py"
SEEDS=range(701,711);NETS=("modena","ltown")
SOURCES=json.loads((OUT/"protocol.json").read_text())["policy"]["evaluation_sources"]

def run(stage,n=None,s=None,source=None):
 name="compare_"+"_".join(str(x) for x in (stage,n,s,source) if x is not None)
 cmd=[sys.executable,str(SCRIPT),"--stage",stage]
 if n:cmd += ["--network",n,"--seed",str(s)]
 if source is not None:cmd += ["--source",str(source)]
 with (LOG/(name+".log")).open("a") as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps({"stage":stage,"network":n,"seed":s,"source":source,"exit_code":r.returncode}),flush=True)
 if r.returncode:raise SystemExit("Comparison halted; inspect "+name)

def extraction_errors():
 for phase in ("calibration","evaluation"):
  p=LOG/(phase+"_completion.json")
  if p.exists() and any(r["exit_code"] for r in json.loads(p.read_text())):
   raise SystemExit("Extraction failed; comparison halted.")

pending={(n,s) for n in NETS for s in SEEDS}
while pending:
 ready=sorted((n,s) for n,s in pending if (HERE/"hybrid_scores"/n/f"seed{s}"/"calibration_validation.json").exists())
 for n,s in ready:run("calibrate",n,s);pending.remove((n,s))
 if pending:extraction_errors();time.sleep(20)
print("ALL20 CALIBRATION SELECTIONS FROZEN BEFORE NEW EVALUATION ANALYSIS",flush=True)
# The protocol and all selections exist before loading new evaluation outcomes.
pending={(n,s,d) for n in NETS for s in SEEDS for d in SOURCES[n]}
while pending:
 ready=sorted((n,s,d) for n,s,d in pending if (HERE/"hybrid_scores"/n/f"seed{s}"/f"evaluation_source{d}.json").exists())
 for n,s,d in ready:run("evaluate",n,s,d);pending.remove((n,s,d))
 if pending:extraction_errors();time.sleep(20)
run("aggregate")
print("COMPLETE COMPARISON",flush=True)
