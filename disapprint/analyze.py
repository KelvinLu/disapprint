from pydub import AudioSegment
import numpy as np
import scipy.ndimage
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sha



FFT_WINDOW_SIZE_MS = 25
FFT_OVERLAP_RATIO = 0.5

PEAK_NEIGHBORHOOD_SIZE = 20
PEAK_MIN_AMP = 20

FINGERPRINT_PEAK_CONNECTIONS = 15

HASH_TRUNCATE = 20



class Fingerprint(object):
    def __init__(self, hashvalue, timeoffset):
        self.hashvalue = hashvalue
        self.timeoffset = timeoffset

    def __eq__(self, other):
        return self.hashvalue == other.hashvalue

    def __str__(self):
        return "Fingerprint: {0} at offset {1}".format(self.hashvalue, self.timeoffset)



def get_raw(fp, format = None):
    # Read in audio file and use pydub to get raw data
    if format is not None:
        a = AudioSegment.from_file(fp, format)
    else:
        a = AudioSegment.from_file(fp)

    raw = a.export(format = 'wav')
    raw.seek(0)
    raw_buffer = raw.read()

    # Load into numpy array, discard header
    raw_data = np.frombuffer(raw_buffer, np.int16)[24:]

    # Split channels and form average
    channels = []
    for c in xrange(a.channels):
        channels.append(raw_data[c::a.channels] / a.channels)
    shortest_length = min([len(c) for c in channels])
    mono = sum([c[:shortest_length] for c in channels])

    return mono, a.frame_rate



def do_fft(data, frame_rate):
    fft_frame_size = 2 ** int(np.log2(FFT_WINDOW_SIZE_MS / 1000. * frame_rate))
    fft_overlap = int(fft_frame_size * FFT_OVERLAP_RATIO)

    spectrum, freqs, times = mlab.specgram(data, NFFT = fft_frame_size, Fs = frame_rate, window = mlab.window_hanning, noverlap = fft_overlap)
    spectrum = 10 * np.log10(spectrum)
    spectrum[spectrum == -np.inf] = 0

    return spectrum, freqs, times



def find_peaks(spectrum, axis_freqs, axis_times):
    struct = scipy.ndimage.morphology.generate_binary_structure(2, 1)
    neighborhood = scipy.ndimage.morphology.iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    maxima_spectrum = scipy.ndimage.filters.maximum_filter(spectrum, footprint = neighborhood)
    maxima_spectrum = (spectrum == maxima_spectrum)
    maxima_spectrum = np.bitwise_and(maxima_spectrum, (spectrum > PEAK_MIN_AMP))

    # Extract peaks
    freqs, times = np.where(maxima_spectrum)

    # Convert to true frequency and true time
    true_freqs = axis_freqs[freqs]
    true_times = axis_times[times]

    # Form an ndarray of peaks, peak-major and sorted by time
    true_peaks = np.array([true_times, true_freqs]).T
    true_peaks = true_peaks[true_peaks[:,0].argsort()]

    for p in true_peaks:
        print p

    # Scatter of the peaks
    fig, ax = plt.subplots()
    ax.imshow(spectrum)
    ax.scatter(times, freqs)
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title("Spectrogram")
    plt.gca().invert_yaxis()
    plt.show()

    return true_peaks



def calculate_fingerprints(peaks):
    fingerprints = []

    num_peaks = len(peaks)

    for i in xrange(num_peaks - FINGERPRINT_PEAK_CONNECTIONS):
        t_i, f_i = peaks[i]

        for j in xrange(i + 1, i + FINGERPRINT_PEAK_CONNECTIONS):
            t_j, f_j = peaks[j]

            t_delta = round(abs(t_j - t_i), 0)
            f_1 = int(f_i / 10) * 10
            f_2 = int(f_j / 10) * 10

            fingerprint_str = "{0}#{1}#{2}".format(t_delta, f_1, f_2)
            fingerprint_hash = sha.new(fingerprint_str).digest()[:HASH_TRUNCATE]

            fingerprints.append(Fingerprint(fingerprint_hash, round(t_i, 2)))

    return fingerprints



def get_fingerprints(fp, format = None):
    """
    Returns a list of hashes with an audio file's fingerprints
    """

    spectrum, axis_freqs, axis_times = do_fft(*get_raw(fp, format))
    peaks = find_peaks(spectrum, axis_freqs, axis_times)
    fingerprints = calculate_fingerprints(peaks)

    return fingerprints



def find_matches(fingerprints_1, fingerprints_2):
    matches = {}

    for fp_1 in fingerprints_1:
        for fp_2 in fingerprints_2:
            if fp_1 == fp_2:
                matches[fp_1.timeoffset - fp_2.timeoffset] = fp_1.hashvalue

    return matches