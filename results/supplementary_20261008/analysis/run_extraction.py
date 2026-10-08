"""Resume isolated inference with a bounded worker pool; never train models."""
from pathlib import Path
import concurrent.futures as cf
import subprocess,sys,json,time,os
HERE=Path(__file__).resolve().parent
LOG=HERE/"logs";LOG.mkdir(exist_ok=True)
WORKERS=3

def job(network,seed,role):
 name=f"{network}_seed{seed}_{role}"
 path=LOG/(name+".log")
 cmd=[sys.executable,str(HERE/"extract_hybrid_scores.py"),"--network",network,"--seed",str(seed),"--role",role]
 start=time.monotonic()
 env=os.environ.copy()
 for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):env[key]="4"
 with path.open("a") as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
 record=dict(network=network,seed=seed,role=role,exit_code=result.returncode,elapsed_seconds=time.monotonic()-start,log=str(path))
 print(json.dumps(record),flush=True)
 return record

records=[]
for role in ("calibration","evaluation"):
 print("PHASE "+role,flush=True)
 jobs=[(n,s,role) for s in range(701,711) for n in ("modena","ltown")]
 with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
  futures=[pool.submit(job,*args) for args in jobs]
  for future in cf.as_completed(futures):records.append(future.result())
 (LOG/(role+"_completion.json")).write_text(json.dumps([r for r in records if r["role"]==role],indent=2)+"\n")
 if any(r["exit_code"] for r in records):
  raise SystemExit("Extraction errors; no next phase started. See logs.")
print("ALL EXTRACTIONS COMPLETE",flush=True)
