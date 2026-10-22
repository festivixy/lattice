# One-word commands for the lattice jobs, loaded from the PowerShell profile.
# `train 5m -vv` runs `python train.py --walltime 5m -vv` from any folder; a leading
# duration (5m, 90s, 1h30m, 90, 00:05:00) becomes --walltime, everything else passes through.
$LatticeDir = $PSScriptRoot

function Invoke-Lattice($job, $rest) {
    $rest = @($rest)
    if ($rest.Count -gt 0 -and "$($rest[0])" -match '^(\d+(\.\d+)?[hms])+$|^\d+$|^\d+(:\d+){1,2}$') {
        $rest = @('--walltime', "$($rest[0])") + @($rest | Select-Object -Skip 1)
    }
    python (Join-Path $LatticeDir "$job.py") @rest
}

function train { Invoke-Lattice train $args }
function simulate { Invoke-Lattice simulate $args }
function solve { Invoke-Lattice solve $args }
function search { Invoke-Lattice search $args }
