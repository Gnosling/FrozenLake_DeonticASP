ACTION_SET = {"LEFT", "DOWN", "RIGHT", "UP"}
TEST_SUITE_SIZE = {"A": 25, "B":18}
ANACONDA_PATH = r'C:\Users\Adi\anaconda3\Scripts\activate.bat'
CONDA_ENV_NAME = 'FrozenLake'
DEBUG_MODE = False
# DEBUG_MODE = False

def debug_print(msg):
    if DEBUG_MODE:
        print(msg)