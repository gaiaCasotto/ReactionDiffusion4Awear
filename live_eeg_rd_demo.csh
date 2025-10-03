#!/bin/csh -fvx


python 5_t_rd.py --fs 256 --buffer-s 8 --port 5001 --user 1 --posx 50 --posy 50 >& demo_live.log &

sleep 5


python client_eeg_live.py --host 127.0.0.1 --port 5001 --fs 256  >& demo_client-live.log &

