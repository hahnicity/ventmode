"""
get_count_pva_artifact
~~~~~~~~~~~~~~~~~~~~~~

I'm so sorry, but this code isn't reproducible. We have published our 2017 paper,
but because the PVA detection algorithms are under patent we haven't released them
publiclly. For now I will just say "trust us we got a result from this private function."
"""
import argparse
from glob import glob
import os

import pandas as pd
from ventmap.raw_utils import extract_raw

from algorithms.tor5 import detectPVI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--solo-file')
    parser.add_argument('-o', '--solo-outfile')
    args = parser.parse_args()

    xfiles = glob('*_cohort/raw_vwd/*.csv')
    yfiles = glob('*_cohort/y_dir/*.csv')
    tor_output = []

    y = None
    for f in yfiles:
        pt = os.path.basename(f).split('-')[0]
        if isinstance(y, type(None)):
            y = pd.read_csv(f)
            y['patient'] = [pt for _ in range(len(y))]
        else:
            tmp = pd.read_csv(f)
            tmp['patient'] = [pt for _ in range(len(tmp))]
            y = y.append(tmp)
    y = y.rename(columns={'vent BN': 'ventBN'})
    vent_bn = y[['ventBN']].iloc[:, 0]
    y = y.drop(['ventBN'], axis=1)
    y['ventBN'] = vent_bn

    if not args.solo_file:
        for f in xfiles:
            pt = os.path.basename(f).split('-')[0]
            solo3, _ = detectPVI(extract_raw(open(f), False), output_subdir='/tmp', write_results=False)
            solo3['patient'] = [pt for _ in range(len(solo3))]
            tor_output.append(solo3)
        solo = pd.concat(tor_output)
        if args.solo_outfile:
            solo.to_pickle(args.solo_outfile)
    else:
        solo = pd.read_pickle(args.solo_file)

    merged = y.merge(solo, how='outer', on=['ventBN', 'BN', 'patient'])
    for cls in ['vc', 'pc', 'ps', 'cpap_sbt', 'pav']:
        cls_df = merged[merged[cls] == 1]
        pvas = cls_df[(cls_df['dbl.4'] > 0) | (cls_df['bs.1or2'] > 0)]
        artifacts = cls_df[(cls_df['cosumtvd'] > 0)]
        co = cls_df[(cls_df['co.noTVi'] > 0)]
        su = cls_df[(cls_df['sumt'] > 0)]
        print('----')
        print(cls)
        print('----')
        print("PVAs: {}".format(len(pvas)))
        print("Artifacts: {}".format(len(artifacts)))
        print("Cough: {}".format(len(co)))
        print("Suction: {}".format(len(su)))


if __name__ == "__main__":
    main()
