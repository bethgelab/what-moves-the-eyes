import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pysaliency
from PIL import Image
from pysaliency.external_datasets.utils import create_stimuli
from pysaliency.utils import check_file_hash, download_file
from tqdm import tqdm

output_directory = Path("output")

tmp_dir = TemporaryDirectory()
tmp_path = Path(tmp_dir.name)

target_directory = Path("output") / "PotsdamCorpus"
target_directory.mkdir(parents=True, exist_ok=True)

print("Downloading dataset...")

source_file = tmp_path / "potsdam_corpus.zip"

# download from https://osf.io/cn5yp/
# MD5sum is different every time, so we don't check it here but instead on the extracted files
download_file(
    url="https://files.de-1.osf.io/v1/resources/n3byq/providers/osfstorage/?zip=",
    target=source_file,
    # md5_hash='b08f846cfa690af33029e1477fc0cff0'
)

print("Extracting zip file")

with zipfile.ZipFile(source_file) as source_archive:
    source_archive.extractall(tmp_path)

check_file_hash(tmp_path / "PotsdamCorpusImages.zip", "6f4dcb9d42468721cc5cc1de54287c80")

check_file_hash(tmp_path / "PotsdamCorpusFixations.dat", "e35ed8ad9c886b97e9d6b8fb1c0f6e7f")

stimuli_source_dir = tmp_path / "stimuli_raw"
stimuli_source_dir.mkdir()

images = []

print("extracting images...")

with zipfile.ZipFile(tmp_path / "PotsdamCorpusImages.zip") as stimuli_archive:
    stimuli_archive.extractall(stimuli_source_dir)
shutil.rmtree(stimuli_source_dir / "__MACOSX", ignore_errors=False)

print("Resizing and padding stimuli...")

for image_file in tqdm(stimuli_source_dir.glob("*.jpg")):
    image = Image.open(image_file)
    image = image.resize((1200, 960), Image.LANCZOS)
    # pad to 1280 x 1024 with gray background
    new_image = Image.new("RGB", (1280, 1024), (128, 128, 128))
    new_image.paste(image, (40, 32))
    new_image.save(image_file, quality=95)
    # image.save(image_file, quality=95)


print("Creating stimuli...")

stimuli_files = sorted(stimuli_source_dir.glob("*.jpg"))
stimuli_files = [f.relative_to(stimuli_source_dir) for f in stimuli_files]
stimuli = create_stimuli(stimuli_source_dir, stimuli_files, target_directory / "stimuli")


print("Building scanpaths...")

fixation_data = pd.read_csv(tmp_path / "PotsdamCorpusFixations.dat", sep=" ")


def load_scanpaths(df, stimuli):
    df = df.copy()

    df["first_fix"] = df["trial"] != df.shift(1)["trial"]
    df["last_fix"] = df["trial"] != df.shift(-1)["trial"]

    train_xs = []
    train_ys = []
    train_ts = []
    train_ns = []
    train_subjects = []
    train_durations = []
    trials = []
    train_forced_fixations = []

    skipped = 0

    # for subject in tqdm(range(1, 251)):
    # df = pd.read_csv(saccade_data.open(f'sac_{subject}.csv'))
    # df['sticky'] = df['sticky'].astype(bool)
    # df['blinkFix'] = df['blinkFix'].astype(bool)
    # df['blinkSac'] = df['blinkSac'].astype(bool)
    current_trial = None
    current_subject = None
    for row_no, row_data in df.iterrows():
        if current_trial != row_data["trial"] or current_subject != row_data["id"]:
            if current_trial is not None:
                train_xs.append(xs)
                train_ys.append(ys)
                train_ts.append(ts)
                train_ns.append(n)
                train_subjects.append(current_subject)
                train_durations.append(durations)
                # train_sticky.append(sticky)
                # train_blinks.append(blinks)
                trials.append(current_trial)
                train_forced_fixations.append(forced_fixations)
            xs = []
            ys = []
            ts = []
            durations = []
            forced_fixations = []
            # n = stimulus_filenames.index(row_data['image'])
            n = row_data["image"] - 1  # the dataset uses 1-based indexing for images
            current_trial = row_data["trial"]
            current_subject = row_data["id"]

            # width = stimuli.sizes[n][1]
            # x_px_per_dva = width / range_data.iloc[1]['xrange']
            # x_px_per_dva = width / 31.1
            x_px_per_dva = 1200 / 31.1  # from paper
            # height = stimuli.sizes[n][0]
            # y_px_per_dva = height / 24.9
            # y_px_per_dva = height / range_data.iloc[1]['yrange']
            y_px_per_dva = 960 / 24.9  # from paper

        if np.isnan(row_data["xpos"]):
            if not row_data["last_fix"] and not row_data["first_fix"]:
                # first fix shouldn't happen, but right now sometimes in DAEMONS the forced fixation
                # is split into two fixations, the first of which only has nan data.
                # Lisa suggested that this might be when a short saccade happens
                # in the forced fixation time.
                # In the Postsdam Corpus this seems not to exist
                print("nan data not in first or last saccade!", row_no)

            skipped += 1
            continue

        # x_offset = 40
        # y_offset = 32

        # offset is now part of the images due to the padding
        x_offset = 0
        y_offset = 0

        xs.append(row_data["xpos"] * x_px_per_dva - x_offset)
        ys.append(stimuli.sizes[n][0] - row_data["ypos"] * y_px_per_dva + y_offset)  # y seems to be flipped
        # ts.append(row_data['End']/1000 if not np.isnan(row_data['End']) else 0.0)
        ts.append(len(ts))
        durations.append(row_data["fd"] / 1000)
        # sticky.append(row_data['sticky'])
        # blinks.append(row_data['blinkFix'])
        forced_fixations.append(row_data["first_fix"])

    print(f"Skipped {skipped} fixations due to missing data")

    scanpaths = pysaliency.ScanpathFixations(
        scanpaths=pysaliency.Scanpaths(
            xs=train_xs,
            ys=train_ys,
            ts=train_ts,
            n=train_ns,
            subject=train_subjects,
            fixation_attributes={
                "durations": train_durations,
                #'blinks': train_blinks,
                #'sticky': train_sticky,
                "forced_fixations": train_forced_fixations,
            },
            scanpath_attributes={
                "trial": trials,
            },
            attribute_mapping={
                "durations": "duration",
                #'blinks': 'blink',
            },
        ),
    )

    return scanpaths


scanpath_fixations = load_scanpaths(fixation_data, stimuli)

print("Saving results...")

stimuli.to_hdf5(target_directory / "stimuli.hdf5")
scanpath_fixations.to_hdf5(target_directory / "fixations.hdf5")

print("Done!")
