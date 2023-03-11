rm -rf ~/.local/share/su2
mkdir -p ~/.local/share/su2
wget -O ~/.local/share/su2/su2.zip https://github.com/su2code/SU2/releases/download/v7.3.1/SU2-v7.3.1-linux64.zip
unzip ~/.local/share/su2/su2.zip -d ~/.local/share/su2
echo 'export PATH=~/.local/share/su2/bin:$PATH' >> ~/.bashrc
echo 'export SU2_RUN=~/.local/share/su2/bin' >> ~/.bashrc
echo 'export PATH=$PYTHONPATH:$SU2_RUN' >> ~/.bashrc
. ~/.bashrc
