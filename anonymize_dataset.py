from glob import glob
import os
from random import randint
import re
import shutil
from warnings import warn

from ventmap.anonymize_datatimes import File, Filename

min_years = 100
max_years = 200
patient_pat = re.compile(r'(\w{32})-')


def main():
    dirs = [
        os.path.join(os.path.dirname(__file__), 'train_data/raw_vwd'),
        os.path.join(os.path.dirname(__file__), 'train_data/y_dir'),
        #os.path.join(os.path.dirname(__file__), 'test_data/raw_vwd'),
        #os.path.join(os.path.dirname(__file__), 'test_data/y_dir'),
    ]
    patient_to_shift_map = {}

    for dir in dirs:
        files = sorted(glob(os.path.join(dir, '*.csv')))
        new_files_to_move = []
        new_dir = dir.replace('train', 'anon_train').replace('test', 'anon_test')
        try:
            os.makedirs(new_dir)
        except OSError:
            pass
        for filename in files:
            patient = patient_pat.search(filename).groups()[0]
            if patient in patient_to_shift_map:
                shift_hours = patient_to_shift_map[patient]
            else:
                shift_hours = randint(min_years*24*365, max_years*24*365)
                patient_to_shift_map[patient] = shift_hours
                print("shifting patient: {} data by hours: {}".format(patient, shift_hours))
            file_obj = File(filename, shift_hours, None, None, True)
            processsed_ok, new_filename = file_obj.process_file()
            if not processsed_ok and 'y_dir' in new_filename:
                filename_obj = Filename(filename, shift_hours, None, None, True)
                new_filename = filename_obj.get_new_filename()
                shutil.copy(filename, new_filename)
                warn('file {} was unable to be processed for datetime purposes. This script '
                     'will continue and shift the datetime in the filename, but please make '
                     'sure this issue is OK'.format(filename))
            elif not processsed_ok:
                raise Exception('Unable to process file: {}. Was there a proper datetime in this file?'.format(filename))
            new_files_to_move.append(new_filename)

        for i, new_filename in enumerate(new_files_to_move):
            new_filepath = os.path.join(new_dir, os.path.basename(new_filename))
            shutil.move(new_filename, new_filepath)


if __name__ == "__main__":
    main()
