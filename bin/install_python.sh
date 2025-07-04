curl https://pyenv.run | bash
pyenv install 3.13
sudo apt install pipx
pipx ensurepath
pipx install poetry 
poetry config virtualenvs.in-project true
pyenv local 3.13.5
poetry install
### activate environemtn