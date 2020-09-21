import os
from typing import Tuple

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files
)

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
_CHECKSUMS = {
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz":
    "3cd23799cb2bbdec517f1cc028f8d43c",
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz":
    "6b74f3901214cb2c2934e98196829835",
}


def load_speechcommands_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    speaker, _ = os.path.splitext(filename)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, speaker_id, utterance_number


class SPEECHCOMMANDS(Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label, speaker_id, utterance_number
    """

    def __init__(self,
                 root: str,
                 url: str = URL,
                 download: bool = False) -> None:

        if url in ["speech_commands_v0.01", "speech_commands_v0.02" ]:
            base_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        self.root = root
        self.url = url

        self.basename = os.path.basename(url)

        basename = self.basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(FOLDER_IN_ARCHIVE, basename)

        self._path = os.path.join(root, folder_in_archive)
        print(self._path)

        if download:
            self._download()

        walker = walk_files(self._path, suffix=".wav", prefix=True)
        walker = filter(lambda w: HASH_DIVIDER in w and EXCEPT_FOLDER not in w, walker)
        self._walker = list(walker)

    def _download(self) -> None:
        """Download the dataset and extract the archive"""
        archive_path = os.path.join(self.root, self.basename)

        if self._check_integrity(self._path):
            print("Dataset already download and verified")

        else:

            checksum = _CHECKSUMS.get(self.url, None)
            download_url(self.url, self.root, hash_value=checksum, hash_type="md5")
            extract_archive(archive_path, self._path)

    def _check_integrity(self, path, checksum=None) -> bool:
        """Check if the dataset already exist and if yes, if it is not corrupted.

        Returns:
            bool: False if the dataset doesn't exist of it is corrupted.
        """
        if not os.path.isdir(path):
            return False

        # TODO add checksum verification
        return True

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        fileid = self._walker[n]
        return load_speechcommands_item(fileid, self._path)

    def __len__(self) -> int:
        return len(self._walker)