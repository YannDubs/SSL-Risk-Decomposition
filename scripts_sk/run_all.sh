#!/usr/bin/env bash

add="-a data.subset=0.01"
add="-a data.n_per_class=3,5,30 is_riskdec=False"

scripts_sk/riskdec_large.sh -s john_XL "$add"
sleep 5

scripts_sk/riskdec.sh -s john "$add"
sleep 5

scripts_sk/dissl_large.sh -s john_XL "$add"
sleep 5

scripts_sk/dissl.sh -s john "$add"
sleep 5

scripts_sk/beit.sh -s john "$add"
sleep 5

scripts_sk/beit_large.sh -s john_large "$add"
sleep 5

scripts_sk/byol.sh -s john "$add"
sleep 5

scripts_sk/clip.sh -s john "$add"
sleep 5

scripts_sk/clip_large.sh -s john_large "$add"
sleep 5

scripts_sk/dino.sh -s john "$add"
sleep 5

scripts_sk/dino_large.sh -s john_XL "$add"
sleep 5

scripts_sk/ibot.sh -s john "$add"
sleep 5

scripts_sk/ibot_large.sh -s john_large "$add"
sleep 5

scripts_sk/lossyless.sh -s john "$add"
sleep 5

scripts_sk/mae.sh -s john "$add"
sleep 5

scripts_sk/mmselfsup.sh -s john "$add"
sleep 5

scripts_sk/moco.sh -s john "$add"
sleep 5

scripts_sk/msn.sh -s john "$add"
sleep 5

scripts_sk/mugs.sh -s john "$add"
sleep 5

scripts_sk/mugs_large.sh -s john_large "$add"
sleep 5

scripts_sk/pycontrast.sh -s john "$add"
sleep 5

scripts_sk/simsiam.sh -s john "$add"
sleep 5

scripts_sk/supervised.sh -s john "$add"
sleep 5

scripts_sk/supervised_large.sh -s john_large "$add"
sleep 5

scripts_sk/swav.sh -s john "$add"
sleep 5

scripts_sk/swav_large.sh -s john_large "$add"
sleep 5

scripts_sk/swav_XXL.sh -s john_XXL "$add"
sleep 5

scripts_sk/vicreg.sh -s john "$add"
sleep 5

scripts_sk/vicreg_large.sh -s john_large "$add"
sleep 5

scripts_sk/vissl.sh -s john "$add"
sleep 5

scripts_sk/vissl_large.sh -s john_large "$add"
sleep 5

scripts_sk/vissl_XL.sh -s john_XL "$add"
sleep 5