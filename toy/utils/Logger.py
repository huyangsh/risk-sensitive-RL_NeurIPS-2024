from tqdm import tqdm

class Logger:
    def __init__(self, prefix, use_tqdm=False, flush_freq=0):
        self.log_path   = prefix + ".log"
        self.log_file   = open(self.log_path, "w")

        self.use_tqdm   = use_tqdm

        self.flush_freq = flush_freq
        self.flush_cnt  = 0

    def log(self, msg):
        self.flush_cnt += 1
        
        if self.use_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

        self.log_file.write(msg + "\n")
        if self.flush_freq == 0 or self.flush_cnt % self.flush_freq == 0:
            self.log_file.flush()
            self.flush_cnt = 0
    
    def save(self):
        self.log_file.close()
        print(f"[info] Log saved to <{self.log_path}>.")


def print_float_list(lst, fmt=".4g"):
    msg = "["
    for x in lst:
        msg += "{:{fmt}}".format(x, fmt=fmt) + ", "
    msg = msg[:-2] + "]"
    return msg

def print_float_matrix(mat, fmt=".4g"):
    x, y = mat.shape

    msg = ""
    for i in range(x):
        for j in range(y):
            msg += "{:{fmt}}".format(mat[i,j], fmt=fmt).rjust(10)
        msg += "\n"
    msg = msg[:-1]
    return msg

def print_episodic_matrix(mat, fmt=".4g"):
    H, x, y = mat.shape

    msg = ""
    for h in range(H):
        msg += f"{h}:"
        for i in range(x):
            for j in range(y):
                msg += "{:{fmt}}".format(mat[h,i,j], fmt=fmt).rjust(10)
            msg += "\n"
        msg += "\n"
    msg = msg[:-1]
    return msg