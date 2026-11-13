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
    $existing = if (Test-Path $profilePath) { "$(Get-Content -Raw $profilePath)" } else { '' }
    $ignoreCase = [StringComparison]::OrdinalIgnoreCase  # Windows paths: code\ and Code\ are the same folder
    if ($existing.IndexOf($shortcuts, $ignoreCase) -ge 0 -or $existing.IndexOf($quoted, $ignoreCase) -ge 0) {
        Write-Host "already set up: $profilePath"
        continue
    }
    New-Item -ItemType Directory -Force (Split-Path $profilePath) | Out-Null
    Add-Content -Path $profilePath -Value "`r`n$line" -Encoding UTF8
    Write-Host "added to: $profilePath"
}

# The fake-typing VS Code extension (vscode/), when VS Code's `code` command is available.
if (Get-Command code -ErrorAction SilentlyContinue) {
    code --install-extension (Join-Path $PSScriptRoot 'vscode\lattice-typer.vsix') --force | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host 'installed the VS Code extension (Ctrl+Alt+Y in a project file)' }
    else { Write-Warning 'could not install the VS Code extension; see vscode\README.md' }
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
