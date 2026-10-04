#!/usr/bin/env bash
# Uretilen iki .docx raporu Windows'taki Google Drive klasorune kopyalar.
#
# Neden boyle: rapor 121 MB (icinde 600 dpi TIFF var) ve GitHub normal push'ta
# 100 MB'lik dosya sinirini asiyor. Google Drive for desktop Windows'ta G:
# surucusu olarak bagli; WSL'de /mnt/g olmadigindan (mount sudo isterdi)
# kopyalamayi Windows PowerShell yapar ve WSL klasorunu \\wsl.localhost
# uzerinden okur.
#
# Kullanim: ./raporlari_drive_a_kopyala.sh
set -euo pipefail

PROJE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEDEF_KLASOR="MOF Raporlari"
PS="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

[ -x "$PS" ] || { echo "HATA: Windows PowerShell bulunamadi ($PS)" >&2; exit 1; }

# WSL yolunu Windows UNC yoluna cevir: /home/x/y -> \\wsl.localhost\<distro>\home\x\y
unc="\\\\wsl.localhost\\${WSL_DISTRO_NAME:-Ubuntu}${PROJE_DIR//\//\\}"

# NOT: betigi dosyadan degil -Command ile veriyoruz; Windows PowerShell 5.1
# BOM'suz UTF-8 .ps1 dosyalarini ANSI sanip Turkce karakterleri bozuyor.
"$PS" -NoProfile -Command "
\$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8
\$ErrorActionPreference = 'Stop'

\$surucu = Get-PSDrive -PSProvider FileSystem | Where-Object { \$_.Description -eq 'Google Drive' }
if (-not \$surucu) { Write-Error 'Google Drive surucusu bulunamadi. Drive for desktop calisiyor mu?'; exit 1 }

\$kok = (Get-ChildItem (\$surucu.Root) -Directory | Select-Object -First 1).FullName
\$hedef = Join-Path \$kok '${HEDEF_KLASOR}'
if (-not [System.IO.Directory]::Exists(\$hedef)) { [void][System.IO.Directory]::CreateDirectory(\$hedef) }
Write-Output \"Hedef: \$hedef\"

\$kaynak = '${unc}'
if (-not (Test-Path -LiteralPath \$kaynak)) { Write-Error \"Kaynak klasor okunamadi: \$kaynak\"; exit 1 }

\$n = 0
foreach (\$d in Get-ChildItem -LiteralPath \$kaynak -Filter *.docx) {
    \$v = Join-Path \$hedef \$d.Name
    [System.IO.File]::Copy(\$d.FullName, \$v, \$true)
    \$y = Get-Item -LiteralPath \$v
    if (\$y.Length -ne \$d.Length) { Write-Error \"Boyut uyusmadi: \$(\$d.Name)\"; exit 1 }
    Write-Output (\"  OK {0,-52} {1,8:N1} MB\" -f \$y.Name, (\$y.Length/1MB))
    \$n++
}
if (\$n -eq 0) { Write-Error 'Kopyalanacak .docx bulunamadi'; exit 1 }
Write-Output \"\$n dosya kopyalandi. Drive arka planda yukluyor - sistem tepsisindeki Drive simgesinden durumu gorebilirsiniz.\"
" 2>&1 | tr -d '\r'
