import librosa
import soundfile

from . import params
from .vggish_input import wavfile_to_examples


def get_audio_examples(audio_file_path):
    """
    Get audio data for vggish_input.
    """
    leh = 10
    ueh = params.SAMPLE_RATE // 2
    return wavfile_to_examples(
        audio_file_path, lower_edge_hertz=leh, upper_edge_hertz=ueh)


def preprocess_audio(participant_name, original_dir, clap_dict):
    """
    Resample the audio to 16kHz and remove data before clap.
    """
    raw_fp = original_dir / 'audio' / 'raw' / f'{participant_name}.wav'
    save_fp = original_dir / 'audio' / 'preprocessed' / f'{participant_name}.wav'
    save_fp.parent.mkdir(exist_ok=True, parents=True)

    # load with librosa in order to downsample to params.SAMPLE_RATE (=16000
    # by default)
    fdata, _ = librosa.load(raw_fp, sr=params.SAMPLE_RATE)

    # use the clap dict to find the clap index
    # clap dict is in ms, so first /1000 to convert to secs
    # then x16000 to get the sample id
    clap_index = int(float(clap_dict[participant_name]) * params.SAMPLE_RATE / 1000)

    # now clip the data that is before the clap
    final_data = fdata[clap_index:]

    # times, tasks = get_distributions(data[participant_name])
    soundfile.write(save_fp, final_data, samplerate=params.SAMPLE_RATE)
