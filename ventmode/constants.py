from os.path import dirname, join

CUR_DIR = dirname(__file__)
DERIVATION_COHORT_X_DIR = join(CUR_DIR, "../anon_train_data", "raw_vwd")
DERIVATION_COHORT_Y_DIR = join(CUR_DIR, "../anon_train_data", "y_dir")
VALIDATION_COHORT_X_DIR = join(CUR_DIR, "../anon_test_data", "raw_vwd")
VALIDATION_COHORT_Y_DIR = join(CUR_DIR, "../anon_test_data", "y_dir")
