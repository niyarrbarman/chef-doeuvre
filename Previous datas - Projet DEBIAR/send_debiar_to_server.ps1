Write-Host "[1/1] Envoi du projet DEBIAR de la machine locale vers le serveur distant..."

wsl rsync -avz --progress /mnt/c/Users/Elouan/Documents/Projet\ Biais\ LLM/gpu-server_backup/file\ to\ send/Projet\ DEBIAR/ evuichard@194.199.113.233:/home/evuichard/Projet\ DEBIAR/

Write-Host "✅ Transfert terminé."