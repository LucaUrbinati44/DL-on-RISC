@echo off
:: Controlla se siamo in esecuzione come amministratore
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [*] Richiesto avvio come amministratore...
    powershell -Command "Start-Process '%~f0' -Verb runAs"
    exit /b
)

:: Se siamo qui, abbiamo i permessi admin
::powershell -Command "usbipd attach --busid 4-1 --wsl"
echo [*] Collegamento dispositivo USB 4-1 a WSL...
usbipd attach --busid 4-1 --wsl
pause