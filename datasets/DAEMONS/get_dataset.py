import shutil
import struct
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import piexif
import pysaliency
from pysaliency.external_datasets.utils import create_stimuli
from pysaliency.utils import download_and_check, filter_files
from tqdm import tqdm

saccade_data_file = Path("source/final_sac.zip")

output_directory = Path("output")

tmp_dir = TemporaryDirectory()
tmp_path = Path(tmp_dir.name)

source_images_zip = tmp_path / "Images.zip"

target_directory = Path("output") / "DAEMONS"
target_directory.mkdir(parents=True, exist_ok=True)

print("Downloading images...")

# download from https://osf.io/cn5yp/
download_and_check(
    url="https://files.de-1.osf.io/v1/resources/cn5yp/providers/osfstorage/6108fd47e38013008e960bf8/?zip=",
    target=source_images_zip,
    md5_hash="7a232c098411f76a67a527679f790a0f",
)

print("Extracting zip file")

with zipfile.ZipFile(source_images_zip) as source_image_archive:
    source_image_archive.extractall(tmp_path)

stimuli_source_dir = tmp_path / "stimuli"
stimuli_source_dir.mkdir()

images = []

print("extracting images...")

for stimuli_zip_file in [tmp_path / "DAEMONS_flickr_corpus.zip", tmp_path / "DAEMONS_potsdam_corpus.zip"]:
    with zipfile.ZipFile(stimuli_zip_file) as archive:
        namelist = archive.namelist()
        namelist = filter_files(namelist, ["__MACOSX", ".DS_Store", "potsdam_corpus_resz.sh"])
        archive.extractall(stimuli_source_dir, namelist)
        images.extend(namelist)

print("Removing subdirectories...")
for image in tqdm(images):
    image_path = stimuli_source_dir / image
    shutil.move(image_path, stimuli_source_dir / image_path.name)
shutil.rmtree(stimuli_source_dir / "DAEMONS_flickr_corpus")
shutil.rmtree(stimuli_source_dir / "DAEMONS_potsdam_corpus")

target_stimuli_directory = target_directory / "stimuli"

print("Creating stimuli...")

filtered_images = [filename for filename in images if filename.endswith(".jpg")]


moved_images = [Path(image).name for image in filtered_images]
sorted_images = sorted(moved_images)

for image_name in sorted_images:
    image_path = stimuli_source_dir / image_name

    try:
        exif_dict = piexif.load(str(image_path))
        if piexif.ImageIFD.Orientation in exif_dict["0th"] and exif_dict["0th"][piexif.ImageIFD.Orientation] != 1:
            # del exif_dict['0th'][piexif.ImageIFD.Orientation]
            exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(image_path))
    except struct.error as e:
        print(f"Failed to load {image_path}, {e}")


stimuli = create_stimuli(stimuli_source_dir, sorted_images, target_stimuli_directory)

if not set(stimuli.sizes) == {(1080, 1920)}:
    raise ValueError("Unexpected image sizes")

print("Building scanpaths...")

with zipfile.ZipFile(saccade_data_file) as saccade_archive:
    saccade_archive.extractall(tmp_path)

px_per_dva = stimuli.sizes[0][1] / 32
print("Px per dva", px_per_dva)


def load_scanpaths(df, stimuli):
    stimulus_filenames = pysaliency.utils.get_minimal_unique_filenames(stimuli.filenames)

    df = df.copy()

    df["first_fix"] = df["trial"] != df.shift(1)["trial"]
    df["last_fix"] = df["trial"] != df.shift(-1)["trial"]

    train_xs = []
    train_ys = []
    train_ts = []
    train_ns = []
    train_subjects = []
    train_durations = []
    train_sticky = []
    train_blinks = []
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
        if current_trial != row_data["trial"] or current_subject != row_data["VP"]:
            if current_trial is not None:
                train_xs.append(xs)
                train_ys.append(ys)
                train_ts.append(ts)
                train_ns.append(n)
                train_subjects.append(current_subject)
                train_durations.append(durations)
                train_sticky.append(sticky)
                train_blinks.append(blinks)
                trials.append(current_trial)
                train_forced_fixations.append(forced_fixations)
            xs = []
            ys = []
            ts = []
            durations = []
            sticky = []
            blinks = []
            forced_fixations = []
            n = stimulus_filenames.index(row_data["Img"])
            current_trial = row_data["trial"]
            current_subject = row_data["VP"]

        if np.isnan(row_data["x"]):
            if not row_data["last_fix"] and not row_data["first_fix"]:
                # first fix shouldn't happen, but right now sometimes the forced fixation
                # is split into two fixations, the first of which only has nan data.
                # Lisa suggested that this might be when a short saccade happens
                # in the forced fixation time.
                print("nan data not in first or last saccade!", row_no)

            skipped += 1
            continue
        xs.append(row_data["x"] * px_per_dva)
        ys.append(stimuli.sizes[n][0] - row_data["y"] * px_per_dva)  # y seems to be flipped
        ts.append(row_data["End"] / 1000 if not np.isnan(row_data["End"]) else 0.0)
        durations.append(row_data["fixdur"] / 1000)
        sticky.append(row_data["sticky"])
        blinks.append(row_data["blinkFix"])
        forced_fixations.append(row_data["forced_fix"])

    print(f"Skipped {skipped} fixations due to missing data")

    scanpaths = pysaliency.FixationTrains.from_fixation_trains(
        xs=train_xs,
        ys=train_ys,
        ts=train_ts,
        ns=train_ns,
        subjects=train_subjects,
        scanpath_fixation_attributes={
            "durations": train_durations,
            "blinks": train_blinks,
            "sticky": train_sticky,
            "forced_fixations": train_forced_fixations,
        },
        scanpath_attributes={
            "trial": trials,
        },
        scanpath_attribute_mapping={
            "durations": "duration",
            "blinks": "blink",
        },
    )

    return scanpaths


scanpaths_train_full = load_scanpaths(pd.read_csv(tmp_path / "final_sac" / "SAC_train.csv"), stimuli=stimuli)
scanpaths_validation_full = load_scanpaths(pd.read_csv(tmp_path / "final_sac" / "SAC_val.csv"), stimuli=stimuli)
scanpaths_test_full = load_scanpaths(pd.read_csv(tmp_path / "final_sac" / "SAC_test.csv"), stimuli=stimuli)


def reduce_stimuli(stimuli, fixations):
    indices = sorted(set(fixations.n))
    new_stimuli, new_fixations = pysaliency.create_subset(stimuli, fixations, indices)
    return new_stimuli, new_fixations


stimuli_train, scanpaths_train = reduce_stimuli(stimuli, scanpaths_train_full)
stimuli_validation, scanpaths_validation = reduce_stimuli(stimuli, scanpaths_validation_full)
stimuli_test, scanpaths_test = reduce_stimuli(stimuli, scanpaths_test_full)

print("Saving results...")

stimuli_train.to_hdf5(target_directory / "stimuli_train.hdf5")
stimuli_validation.to_hdf5(target_directory / "stimuli_validation.hdf5")
stimuli_test.to_hdf5(target_directory / "stimuli_test.hdf5")

scanpaths_train.to_hdf5(target_directory / "fixations_train.hdf5")
scanpaths_validation.to_hdf5(target_directory / "fixations_validation.hdf5")
scanpaths_test.to_hdf5(target_directory / "fixations_test.hdf5")

print("Done!")
