#!/bin/bash
rand_threshs=(.1 .2 .3 .4 .5 .6 .7 .8 .9 .95 .96 .97 .98 .99 .991 .992 .993 .994 .995 .996 .997 .998 .999)

for thresh in ${rand_threshs[@]}
do
    python main.py --split-type validation -r ${thresh} --to-pickle "rand-thresh${thresh}.pkl"
done
