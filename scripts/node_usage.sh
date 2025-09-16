#!/usr/bin/env bash
set -euo pipefail
JOB="${1:-$SLURM_JOB_ID}"

hosts="$(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)")"
for n in $hosts; do
  scontrol -o show node "$n" | awk -v N="$n" '
    {
      for (i=1;i<=NF;i++){
        split($i,a,"=")
        if(a[1]=="CPULoad")         load=a[2]
        if(a[1]=="ThreadsPerCore")   tpc=a[2]
        if(a[1]=="CoresPerSocket")   cps=a[2]
        if(a[1]=="Sockets")           s=a[2]
        if(a[1]=="CPUTot")           tot=a[2]
      }
    }
    END{
      phys = s * cps             # physical cores
      phys_active = (load > phys) ? phys : load
      smt = (load > phys) ? (load - phys) : 0
      printf "%s: ~%.0f/%d physical cores busy; logical usage %.0f/%d threads (CPULoad=%.1f, tpc=%d, SMT-threads≈%.0f)\n",
             N, phys_active, phys, load, phys*tpc, load, tpc, smt
    }'
done
