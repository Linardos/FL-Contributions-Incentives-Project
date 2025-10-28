import os, shutil, glob, re
BRATS_DIR = "/mnt/d/Datasets/FETS_data/MICCAI_FeTS2022_TrainingData"
OUT = "/mnt/d/Datasets/FETS_data/FeTS_ready"
os.makedirs(OUT, exist_ok=True)

# each subject is a folder under BRATS_DIR (adjust the glob if your layout differs)
subs = sorted([d for d in glob.glob(os.path.join(BRATS_DIR, "*")) if os.path.isdir(d)])

for s in subs:
    # subject name becomes the folder name in the input
    subj = os.path.basename(s.rstrip("/"))
    dst = os.path.join(OUT, subj)
    os.makedirs(dst, exist_ok=True)

    def pick(pat):
        cand = glob.glob(os.path.join(s, f"*{pat}.nii.gz"))
        if not cand:
            raise FileNotFoundError(f"Missing {pat} for {subj}")
        return cand[0]

    mapping = {
        "-t1n.nii.gz": pick("t1"),     # native T1
        "-t1c.nii.gz": pick("t1ce"),   # contrast
        "-t2w.nii.gz": pick("t2"),
        "-t2f.nii.gz": pick("flair"),
    }
    for suffix, src in mapping.items():
        dstfile = os.path.join(dst, subj + suffix)
        if not os.path.exists(dstfile):
            shutil.copy(src, dstfile)

print("Prepared:", OUT)
