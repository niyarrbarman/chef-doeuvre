Write-Host "[1/3] Sauvegarde du projet DEBIAR du serveur distant vers le local..."

wsl rsync -avz --progress evuichard@194.199.113.233:/home/evuichard/Projet\ DEBIAR/ /mnt/c/Users/Elouan/Documents/Projet\ Biais\ LLM/gpu-server_backup/Projet\ DEBIAR/

Write-Host "Sauvegarde terminée."
