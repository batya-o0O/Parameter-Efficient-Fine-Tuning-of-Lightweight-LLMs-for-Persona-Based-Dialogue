param(
    [Parameter(Mandatory = $true)][string]$User,
    [Parameter(Mandatory = $true)][string]$Host,
    [int]$Port = 22,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$RemoteDir = "~/persona_thesis_rtx4500_k5"
)

$LocalDir = Split-Path -Parent $MyInvocation.MyCommand.Path
scp -i $KeyPath -P $Port -r $LocalDir "$User@$Host`:$RemoteDir"
