param(
    [Parameter(Mandatory = $true)][string]$User,
    [Parameter(Mandatory = $true)][string]$Host,
    [int]$Port = 22,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
)

ssh -i $KeyPath -p $Port "$User@$Host"
