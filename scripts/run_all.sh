#!/usr/bin/env bash

scripts/riskdec_large.sh -s nlprun_large
sleep 5

scripts/riskdec.sh -s nlprun
sleep 5

scripts/dissl_large.sh -s nlprun_large
sleep 5

scripts/dissl.sh -s nlprun
sleep 5

scripts/beit.sh -s nlprun
sleep 5

scripts/byol.sh -s nlprun
sleep 5

scripts/clip.sh -s nlprun
sleep 5

scripts/dino.sh -s nlprun
sleep 5

scripts/ibot.sh -s nlprun
sleep 5

scripts/lossyless.sh -s nlprun
sleep 5

scripts/mae.sh -s nlprun
sleep 5

scripts/mmselfsup.sh -s nlprun
sleep 5

scripts/moco.sh -s nlprun
sleep 5

scripts/msn.sh -s nlprun
sleep 5

scripts/mugs.sh -s nlprun
sleep 5

scripts/pycontrast.sh -s nlprun
sleep 5

scripts/simsiam.sh -s nlprun
sleep 5

scripts/supervised.sh -s nlprun
sleep 5

scripts/swav.sh -s nlprun
sleep 5

scripts/vicreg.sh -s nlprun
sleep 5

scripts/vissl.sh -s nlprun
sleep 5