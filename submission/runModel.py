from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

from multiprocessing.pool import Pool

import multiprocessing

import os
import numpy as np
import pandas as pd
import ctypes
from ctypes import cdll
import re
import matplotlib.pyplot as plt
import time


os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Model():
    def __init__(self):
        super().__init__()

        dirName = os.path.dirname(__file__)

        geomPath = os.path.join(dirName, 'geometry.bin')
        connPath = os.path.join(dirName, 'connections.bin')
        libPath  = os.path.join(dirName, 'libModel.so')
        geom_string = geomPath.encode('utf-8')
        conn_string = connPath.encode('utf-8')

        p1D_float = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
        p1D_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS')
        p1D_int    = np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='CONTIGUOUS')

        self.tmlLib = cdll.LoadLibrary(libPath)

        self.tmlLib.createModel.restype = ctypes.c_void_p
        self.tmlLib.createModel.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.tmlLib.deleteModel.argtypes = [ctypes.c_void_p]

        self.tmlLib.importHits.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           p1D_int,
                                           p1D_float, p1D_float, p1D_float,
                                           p1D_int, p1D_int, p1D_int]

        self.tmlLib.importCells.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            p1D_int, p1D_int, p1D_int, p1D_float]

        self.tmlLib.findTracks.argtypes = [ctypes.c_void_p, p1D_int]

        self.tmlModel = self.tmlLib.createModel(geom_string, len(geom_string), conn_string, len(conn_string))

    def __del__(self) :
        self.tmlLib.deleteModel(self.tmlModel)

    def predict_one_event(self, event_id, hits, cells):

        hit_id= hits.hit_id.values
        hit_x = hits.x.values
        hit_y = hits.y.values
        hit_z = hits.z.values
        vol_id= hits.volume_id.values
        lay_id= hits.layer_id.values
        mod_id= hits.module_id.values

        cells_hit_id = cells.hit_id.values
        cells_ch0    = cells.ch0.values
        cells_ch1    = cells.ch1.values
        cells_value  = cells.value.values

        nHits = hit_x.shape[0]
        nCells = cells.hit_id.shape[0]

        #hit_id,ch0,ch1,value
        #1,259,732,0.297276
        #2,306,1097,0.297281
        #3,268,995,0.00778383
        #3,267,995,0.118674
        #3,267,996,0.194891
        print(hit_id)
        self.tmlLib.importHits(self.tmlModel, ctypes.c_int(hits.x.values.shape[0]), hit_id.astype('int32'), hit_x.astype('float32'), hit_y.astype('float32'), hit_z.astype('float32'), vol_id.astype('int32'), lay_id.astype('int32'), mod_id.astype('int32'))

        self.tmlLib.importCells(self.tmlModel, ctypes.c_int(cells.hit_id.shape[0]), cells_hit_id.astype('int32'), cells_ch0.astype('int32'), cells_ch1.astype('int32'), cells_value.astype('float32'))

        labels = np.zeros(shape = hit_id.shape, dtype = np.int32)

        self.tmlLib.findTracks(self.tmlModel, labels)

        sub = pd.DataFrame(data=np.column_stack((hits.hit_id.values, labels)), columns=["hit_id", "track_id"]).astype(int)
        sub['event_id'] = event_id
        return sub


def _analyze_tracks(truth, submission):
    """Compute the majority particle, hit counts, and weight for each track.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    Returns
    -------
    pandas.DataFrame
        Contains track_id, nhits, major_particle_id, major_particle_nhits,
        major_nhits, and major_weight columns.
    """
    # true number of hits for each particle_id
    particles_nhits = truth['particle_id'].value_counts(sort=False)
    total_weight = truth['weight'].sum()
    # combined event with minimal reconstructed and truth information
    event = pd.merge(truth[['hit_id', 'particle_id', 'weight']],
                         submission[['hit_id', 'track_id']],
                         on=['hit_id'], how='left', validate='one_to_one')
    event.drop('hit_id', axis=1, inplace=True)
    event.sort_values(by=['track_id', 'particle_id'], inplace=True)

    # ASSUMPTIONs: 0 <= track_id, 0 <= particle_id

    tracks = []
    # running sum for the reconstructed track we are currently in
    rec_track_id = -1
    rec_nhits = 0
    # running sum for the particle we are currently in (in this track_id)
    cur_particle_id = -1
    cur_nhits = 0
    cur_weight = 0
    # majority particle with most hits up to now (in this track_id)
    maj_particle_id = -1
    maj_nhits = 0
    maj_weight = 0

    for hit in event.itertuples(index=False):
        # we reached the next track so we need to finish the current one
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            # could be that the current particle is the majority one
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # store values for this track
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                particles_nhits[maj_particle_id], maj_nhits,
                maj_weight / total_weight))

        # setup running values for next track (or first)
        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
            maj_particle_id = -1
            maj_nhits = 0
            maj_weights = 0
            continue

        # hit is part of the current reconstructed track
        rec_nhits += 1

        # reached new particle within the same reconstructed track
        if cur_particle_id != hit.particle_id:
            # check if last particle has more hits than the majority one
            # if yes, set the last particle as the new majority particle
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # reset runnig values for current particle
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
        # hit belongs to the same particle within the same reconstructed track
        else:
            cur_nhits += 1
            cur_weight += hit.weight

    # last track is not handled inside the loop
    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits
        maj_weight = cur_weight
    # store values for the last track
    tracks.append((rec_track_id, rec_nhits, maj_particle_id,
        particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))

    cols = ['track_id', 'nhits',
            'major_particle_id', 'major_particle_nhits',
            'major_nhits', 'major_weight']
    return pd.DataFrame.from_records(tracks, columns=cols)

def score_event(truth, submission):
    """Compute the TrackML event score for a single event.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    """
    tracks = _analyze_tracks(truth, submission)
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
    return tracks['major_weight'][good_track].sum()



m = Model()
START_EVENT = 1000
END_EVENT = 1100

print("=======================================")
print("Building file directory system...")

files = os.listdir("../train_100_events")
events_dict = {}
for i in range(START_EVENT, END_EVENT):
    for f in files:
        if str(i) in f:
            if not str(i) in events_dict:
                events_dict[str(i)] = {}
            type = re.findall("-([a-zA-Z]+).csv", f)[0]
            events_dict[str(i)][type] = pd.read_csv("../train_100_events/"+f)

print("Complete!")
print("=======================================")
print("Training and scoring...")
scores = []
times = []
for event_id in events_dict:
    event = events_dict[event_id]

    start = time.process_time()
    s = m.predict_one_event(event_id, event['hits'], event['cells'])
    end = time.process_time()
    t = end-start
    times.append(t)
    scores.append(score_event(event['truth'], s))

print("=======================================")
print("Graphing...")

print("The average time to train was ", np.mean(times), "seconds")
print("The average score was ", np.mean(scores), "seconds")

plt.scatter(range(START_EVENT, END_EVENT), scores)
plt.xlabel("Event")
plt.ylabel("Score")
plt.title("Scores of first 100 Events")
plt.show()

plt.scatter(range(START_EVENT, END_EVENT), times)
plt.xlabel("Event")
plt.ylabel("Time")
plt.title("CPU Timing of first 100 Events")
plt.show()


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


quartile1, medians, quartile3 = np.percentile(scores, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

plt.title("Scoring Distribution of First 100 Events")
plt.show()

plt.violinplot(times)
plt.title("Timing Distribution of First 100 Events")
plt.show()

print("Done!")
print("=======================================")




#cell1000 = pd.read_csv("../train_100_events/event000001000-cells.csv")
#hits1000 = pd.read_csv("../train_100_events/event000001000-hits.csv")
#truth1000 = pd.read_csv("../train_100_events/event000001000-truth.csv")
#event_id = "1000"
#s = m.predict_one_event(event_id, hits1000, cell1000)

#print(s)
#print("The score for event", event_id, "is", score_event(truth1000, s))
#predictions = m.predict_one_event("/train_100_events/")
