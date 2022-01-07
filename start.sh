#!/bin/bash

rmmod brcmfmac.ko
insmod /home/pi/Desktop/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/brcmfmac_5.10.y-nexmon/brcmfmac.ko
sleep 1

pkill wpa_supplicant
ifconfig wlan0 up
sleep 1

mcpparams="$(/home/pi/Desktop/nexmon/patches/bcm43455c0/7_45_189/nexmon_csi/utils/makecsiparams/makecsiparams -c 157/80 -C 1 -N 1 -m 70:8b:cd:c8:5f:fc -b 0x80)"
nexutil -Iwlan0 -s500 -b -l34 "-v$mcpparams"
nexutil -k

iw phy `iw dev wlan0 info | gawk '/wiphy/ {printf "phy" $2}'` interface add mon0 type monitor
ifconfig mon0 up

echo "Start receiving the packet by : tcpdump -i wlan0 dst port 5500"
