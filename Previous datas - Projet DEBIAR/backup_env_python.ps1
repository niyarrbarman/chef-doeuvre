Write-Host "[1/3] Sauvegarde des environnements et builds Python (gpuenv, OpenSSL, Python compilé)..."

# Crée le dossier de destination local
wsl mkdir -p "/mnt/c/Users/Elouan/Documents/Projet Biais LLM/gpu-server_backup/python_env_full"

# Lance la synchronisation
wsl rsync -avz --progress evuichard@194.199.113.233:/home/evuichard/Python-3.10.13/ /mnt/c/Users/Elouan/Documents/Projet\ Biais\ LLM/gpu-server_backup/python_env_full/Python-3.10.13/

wsl rsync -avz --progress evuichard@194.199.113.233:/home/evuichard/python-3.10.13-install/ /mnt/c/Users/Elouan/Documents/Projet\ Biais\ LLM/gpu-server_backup/python_env_full/python-3.10.13-install/

wsl rsync -avz --progress evuichard@194.199.113.233:/home/evuichard/openssl/ /mnt/c/Users/Elouan/Documents/Projet\ Biais\ LLM/gpu-server_backup/python_env_full/openssl/

wsl rsync -avz --progress evuichard@194.199.113.233:/home/evuichard/gpuenv/ /mnt/c/Users/Elouan/Documents/Projet\ Biais\ LLM/gpu-server_backup/python_env_full/gpuenv/

Write-Host "Sauvegarde terminée."
