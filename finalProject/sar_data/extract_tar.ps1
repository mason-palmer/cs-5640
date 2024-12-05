# Run this in the directory where downloaded tar.gz files are located
Get-ChildItem -Filter *.tar.gz | ForEach-Object {
    tar -xzf $_.FullName
    Remove-Item $_.FullName -Force
}