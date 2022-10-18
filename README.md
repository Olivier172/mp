# mp
Handig om te weten:

---------------------------------------------------------
Om vs code op te starten met een linux terminal
1) start de ubuntu app
2) cd /mnt/e/AAA_MASTERPROEF/github/
3) code . (start vs code in de huidige directory)
https://code.visualstudio.com/docs/remote/wsl#_getting-started 

----------------------------------------------------------

PIP commandos
1) pip list -> alle libraries laten zien
2) pip show library -> info/versie checken
3) pip install library -> nieuwe library installeren. handige zijn [numpy, matplotlib, pandas, jupyter, torch, torchvision, torchaudio] (dit is de pytorch lib)
4) pip freeze > requirements.txt (alle libraries staan dan in requirements.txt)

Virtual envirement
maken: $python3 -m venv venv
activeren: $source venv/bin/activate
deactiveren : $deactivate
https://realpython.com/python-virtual-environments-a-primer/#activate-it 

nvidia driver info (cuda versie checken)
nvidia-smi commando in cmd uitvoeren

-----------------------------------------------------------

handige trouble shooting
https://www.youtube.com/watch?v=myouQBBDhXQ 
https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode
https://code.visualstudio.com/docs/remote/wsl#_getting-started
https://cloudbytes.dev/snippets/upgrade-python-to-latest-version-on-ubuntu-linux
https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with

-----------------------------------------------------------
wsl --list
wsl --install -d Ubuntu
wsl --set-default-version 1
-------------------------------------------------
op ubuntu
First use this command:
sudo apt-get update
Then:
sudo apt-get install python3-pip
