#!/usr/bin/env bash

# Download and install CARLA
mkdir carla
cd carla
wget -O CARLA_0.9.10.1.tar.gz https://tiny.carla.org/carla-0-9-10-1-linux
wget -O AdditionalMaps_0.9.10.1.tar.gz https://tiny.carla.org/additional-maps-0-9-10-1-linux
tar -xf CARLA_0.9.10.1.tar.gz
tar -xf AdditionalMaps_0.9.10.1.tar.gz
rm CARLA_0.9.10.1.tar.gz
rm AdditionalMaps_0.9.10.1.tar.gz
cd ..