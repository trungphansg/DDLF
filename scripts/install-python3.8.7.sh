sudo apt-get install build-essential checkinstall <<< 1
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev \
     libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev <<< Y
cd /opt
sudo wget https://www.python.org/ftp/python/3.8.7/Python-3.8.7.tgz
sudo tar xzf Python-3.8.7.tgz
cd Python-3.8.7
sudo ./configure --enable-optimizations
sudo make altinstall
python3.8 -V
cd /opt
sudo rm -f Python-3.8.7.tgz
cd
python3.8 -m pip install --upgrade pip
