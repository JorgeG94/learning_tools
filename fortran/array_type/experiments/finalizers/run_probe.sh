#!/bin/bash
# Build and run probe.F90 with every compiler we care about.
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/build"
mkdir -p "$out"

run() {
   local label="$1" mods="$2" fc="$3" flags="$4"
   (
      module purge >/dev/null 2>&1
      module load $mods >/dev/null 2>&1
      echo "=== $label ($($fc --version 2>&1 | grep -m1 -iE 'version|nvfortran|GNU|ifx'))"
      cd "$out"
      for v in "" "-DDEFINED_ASSIGN"; do
         rm -f *.mod
         if $fc $flags $v "$here/probe.F90" -o "probe_$label" >"build_$label.log" 2>&1; then
            "./probe_$label"
         else
            echo "COMPILE FAILED $v"; tail -5 "build_$label.log"
         fi
      done
   )
}

run nvfortran26.5 "nvidia-hpc-sdk/26.5" nvfortran "-O2 -Mpreprocess"
run nvfortran26.5-acc "nvidia-hpc-sdk/26.5 cuda" nvfortran "-O2 -Mpreprocess -acc -gpu=cc70"
run gfortran12 "gcc/12.2.0" gfortran "-O2 -cpp"
run gfortran13 "gcc/13.2.0" gfortran "-O2 -cpp"
run gfortran14 "gcc/14.2.0" gfortran "-O2 -cpp"
run gfortran15 "gcc/15.1.0" gfortran "-O2 -cpp"
run ifx2025.3 "intel-compiler-llvm/2025.3.2" ifx "-O2 -fpp"
