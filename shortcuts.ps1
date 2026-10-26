# One-word commands for the lattice jobs, loaded from the PowerShell profile (see install.ps1).
# `train 30m -vv` or `train -30m -vv` runs train.py with a 30-minute walltime from any folder;
# everything after the duration passes straight through.
$LatticeDir = $PSScriptRoot

function Invoke-Lattice($job, $rest) {
    $script = Join-Path $LatticeDir "$job.py"
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 $script @rest } else { python $script @rest }
}

function train { Invoke-Lattice train $args }
function simulate { Invoke-Lattice simulate $args }
function solve { Invoke-Lattice solve $args }
function search { Invoke-Lattice search $args }
