# Load environment variables from a .env file into the current PowerShell session.
#
# This script MUST be dot-sourced for the variables to persist in the caller:
#
#     . .\scripts\load_env.ps1                  # default: loads .\.env
#     . .\scripts\load_env.ps1 -Path .env.prod  # custom path
#
# Lines starting with '#' are treated as comments. Surrounding quotes around
# values are stripped. Blank lines are ignored.

param(
    [string] $Path = ".env"
)

if (-not (Test-Path $Path)) {
    Write-Warning "No env file at '$Path' - skipping load."
    return
}

$count = 0
Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line)            { return }
    if ($line.StartsWith('#')) { return }

    $idx = $line.IndexOf('=')
    if ($idx -lt 1) { return }

    $name  = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1).Trim()
    if ($value.StartsWith('"') -and $value.EndsWith('"')) {
        $value = $value.Substring(1, $value.Length - 2)
    } elseif ($value.StartsWith("'") -and $value.EndsWith("'")) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    Set-Item -Path "env:$name" -Value $value
    $count++
}

Write-Host "Loaded $count variables from $Path" -ForegroundColor Green
