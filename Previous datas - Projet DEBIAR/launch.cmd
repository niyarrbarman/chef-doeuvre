@echo off
set SERVER=evuichard@194.199.113.233
set REMOTE_PATH=/home/evuichard/launch_jupyter.sh

echo [1/3] Envoi du script vers le serveur...
scp launch_jupyter.sh %SERVER%:%REMOTE_PATH%
if errorlevel 1 (
    echo Erreur lors de l'envoi du script.
    pause
    exit /b 1
)

echo [2/3] Connexion SSH et lancement du script...
ssh %SERVER% "bash %REMOTE_PATH%"
if errorlevel 1 (
    echo Erreur lors de la connexion SSH ou de l'execution.
    pause
    exit /b 1
)

echo Script lancé avec succès.
pause