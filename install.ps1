# Adds the train / simulate / solve / search commands to PowerShell: .\install.ps1
$shortcuts = Join-Path $PSScriptRoot 'shortcuts.ps1'
$quoted = "'" + $shortcuts.Replace("'", "''") + "'"
$line = "if (Test-Path $quoted) { . $quoted }"
$docs = [Environment]::GetFolderPath('MyDocuments')

# Windows PowerShell always; PowerShell 7 only when it is installed
$folders = @('WindowsPowerShell')
if ((Get-Command pwsh -ErrorAction SilentlyContinue) -or (Test-Path (Join-Path $docs 'PowerShell'))) {
    $folders += 'PowerShell'
}
foreach ($folder in $folders) {
    $profilePath = Join-Path $docs "$folder\Microsoft.PowerShell_profile.ps1"
    $existing = if (Test-Path $profilePath) { Get-Content -Raw $profilePath } else { '' }
    if ("$existing".Contains($shortcuts) -or "$existing".Contains($quoted)) {
        Write-Host "already set up: $profilePath"
        continue
    }
    New-Item -ItemType Directory -Force (Split-Path $profilePath) | Out-Null
    Add-Content -Path $profilePath -Value "`r`n$line" -Encoding UTF8
    Write-Host "added to: $profilePath"
}

# Profiles only load when scripts are allowed; Windows blocks them by default.
$policy = 'Undefined'
foreach ($scope in 'MachinePolicy', 'UserPolicy', 'CurrentUser', 'LocalMachine') {
    $policy = Get-ExecutionPolicy -Scope $scope
    if ($policy -ne 'Undefined') { break }
}
if ($policy -in 'Undefined', 'Restricted', 'AllSigned') {
    Write-Warning ('PowerShell is blocking scripts, so the commands will not load. Run this once: ' +
                   'Set-ExecutionPolicy -Scope CurrentUser RemoteSigned')
}
Write-Host 'done - open a new PowerShell window and run: train 5m'
