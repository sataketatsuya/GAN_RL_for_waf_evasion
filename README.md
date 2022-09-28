# GAN_RL_for_waf_evasion


## How to build

```console
$ git clone https://github.com/sataketatsuya/GAN_RL_for_waf_evasion.git
$ cd GAN_RL_for_waf_evasion/gym_waf/data
$ wget https://www.tic.itefi.csic.es/dataset/normalTrafficTraining.rar
$ unrar x normalTrafficTraining.rar
$ cd ../../
$ git clone https://github.com/sataketatsuya/docker-nginx-modsecurity.git
$ cd docker-nginx-modsecurity
$ docker build -t docker-nginx-modsecurity `pwd`
$ docker run -it --rm -p 80:80 docker-nginx-modsecurity
```
