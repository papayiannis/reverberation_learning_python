# Copyright 2018 Constantinos Papayiannis
# 
# This file is part of Reverberation Learning Toolbox for Python.
# 
# Reverberation Learning Toolbox for Python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Reverberation Learning Toolbox for Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Reverberation Learning Toolbox for Python.  If not, see <http://www.gnu.org/licenses/>.

"""

Acoustic environment object.

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
C. Papayiannis, C. Evers, and P. A. Naylor, "End-to-end discriminative models for the reverberation effect," (to be submitted), 2019
"""

import numpy as np
from scipy.io import wavfile

from utils_base import getfname, column_vector
from utils_spaudio import my_resample


class AcEnv(object):
    """
    Defines Acoustic Environments by including their impulse response, enclosure and measurement
    information

    Attributes:
        name : A label for the environment
        impulse_response : The impulse response (AIR) measured at a position from a source
        at a different position
        sampling_freq : The sampling frequency at which the AIR was measured in
        room_name : The label of the room
        room_type : The type of the room
        room_dimensions : The dimensions of the room
        rec_position : The receiver's (microphone) position for the measurement of the AIR
        src_position : The source's position for the measurement of the AIR
        is_simulation : Indicating whether this is a product of simulation
        from_database : The database from which the AIR was taken from
        known_room : Indicating whether stored metadata for this AIR have been used
    """

    def __init__(self, name='', filename='', samples=None, sampling_freq=0, keep_channel=None,
                 max_allowed_silence=0.001, is_simulation=None, silent_samples_offset=False,
                 matlab_engine=None):
        """

        Args:
            name: Used as the label for the object
            filename: The filename for the measured/simualted AIR
            samples: The samples of the measured/simulated AIR
            sampling_freq (int): The sampling frequency for the AIR
        """
        self.name = name
        self.sampling_freq = 0
        self.impulse_response = np.array([])
        self.room_name = None
        self.room_type = None
        self.room_dimensions = (None, None, None)
        self.rec_position = [(None, None, None)]
        self.src_position = (None, None, None)
        self.is_simulation = is_simulation
        self.from_database = None
        self.receiver_name = None
        self.receiver_config = None
        self.known_room = False
        self.filename = filename
        self.nchannels = 0
        max_allowed_silence_samples = int(np.ceil(max_allowed_silence * self.sampling_freq))

        if (len(filename) == 0) & (samples is None):
            raise NameError(getfname() + ':FilenameOrSamplesRequired')

        if samples is not None:
            self.impulse_response = samples
            if sampling_freq <= 0:
                raise AssertionError('SamplingFreqCannotBe0orNegative')
            self.sampling_freq = sampling_freq
            if keep_channel is not None:
                self.impulse_response = self.get_channel(keep_channel)
        else:
            self.sampling_freq, self.impulse_response = wavfile.read(self.filename)
            if keep_channel is not None:
                self.impulse_response = self.get_channel(keep_channel)
            self.impulse_response = self.impulse_response.astype(float) / np.max(
                np.abs(self.impulse_response))
            if sampling_freq > 0:
                self.impulse_response = np.array(my_resample(np.array(
                    self.impulse_response), self.sampling_freq, sampling_freq,
                    matlab_eng=matlab_engine))
                self.sampling_freq = sampling_freq
        try:
            if self.impulse_response.ndim == 1:
                self.impulse_response = column_vector(self.impulse_response)
        except AttributeError:
            pass
        if len(filename) > 0:
            self.add_room_info()
            self.add_receiver_info()
        if self.impulse_response.ndim < 2:
            self.nchannels = 1
        else:
            self.nchannels = self.impulse_response.shape[1]

        scale_by = float(abs(self.impulse_response).max())
        if scale_by > 0:
            self.impulse_response = self.impulse_response / scale_by

        if self.impulse_response is not None:
            start_sample = self.impulse_response.shape[0]
            for i in range(self.impulse_response.shape[1]):
                start_sample = min(start_sample,
                                   max(0, np.nonzero(self.impulse_response[:, i])[0][0] -
                                       max_allowed_silence_samples))
            if start_sample > 0 and silent_samples_offset:
                self.impulse_response = self.impulse_response[start_sample:, :]
                print('Offsetted AIR by ' + str(start_sample) + ' samples')

    def get_channel(self, idx):
        if self.impulse_response is None:
            print('No AIR samples saved')
            return None
        return self.impulse_response[:, idx]

    def add_receiver_info(self):
        filename_lookup_string = ['Chromebook', 'Crucif', 'EM32', 'Lin8Ch', 'Mobile', 'Single']

        matches = [i for i in range(len(filename_lookup_string)) if filename_lookup_string[i] in
                   self.filename]
        if len(matches) > 0:
            matches = matches[0]
            self.receiver_name = filename_lookup_string[matches]

        filename_lookup_string = ['_1_RIR.wav', '_2_RIR.wav']
        config_string = [1, 2]

        matches = [i for i in range(len(filename_lookup_string)) if filename_lookup_string[i] in
                   self.filename]
        if len(matches) > 0:
            matches = matches[0]
            self.receiver_name = config_string[matches]

    def add_room_info(self):
        """
        Adds room information to Acoustic environment if the filename matches a known database

        Returns: Nothing

        """
        filename_lookup_string = ['car h',
                                  '502', '803', '503', '611', '508', '403', 'EE',
                                  'booth', 'office', 'meeting', 'lecture', 'aula carolina',
                                  'stairway', 'bathroom', 'corridor', 'kitchen', 'lecture1',
                                  'stairway', 'stairway1', 'stairway2',
                                  'R408', 'R503', 'L8Stair', 'L6Cafe', 'R803 Office', 'R403',
                                  'L8Hall',
                                  'cafeteria', 'courtyard', 'office II', 'office I',
                                  ]
        from_DB = ['Harman/Becker',
                   'ACE', 'ACE', 'ACE', 'ACE', 'ACE', 'ACE', 'ACE',
                   'Aachen', 'Aachen', 'Aachen', 'Aachen', 'Aachen', 'Aachen', 'Aachen', 'Aachen',
                   'Aachen', 'Aachen', 'Aachen', 'Aachen', 'Aachen',
                   'SAP', 'SAP', 'SAP', 'SAP', 'SAP', 'SAP', 'SAP',
                   'Oldenburg', 'Oldenburg', 'Oldenburg', 'Oldenburg']

        room_names = ['Car',
                      'IC_EE_502', 'IC_EE_803', 'IC_EE_503', 'IC_EE_611', 'IC_EE_508', 'IC_EE_403',
                      'IC_EE_EE',
                      'Booth', 'Office', 'Meeting', 'Lecture', 'Aula_Carolina', 'Stairway',
                      'Bathroom', 'Corridor', 'Kitchen', 'Lecture1', 'Stairway', 'Stairway1',
                      'Stairway2',
                      'IC_EE_408', 'IC_EE_503', 'IC_EE_L8Stair', 'IC_EE_L6Cafe', 'IC_EE_803',
                      'IC_EE_403', 'IC_EE_L8Hall',
                      'Cafeteria', 'Courtyard', 'Office_I', 'Office_II']

        room_types = ['Car',
                      'Office', 'Office', 'Meeting', 'Meeting', 'Lecture', 'Lecture',
                      'Corridor',
                      'Booth', 'Office', 'Meeting', 'Lecture', 'Lecture', 'Corridor',
                      'Bathroom', 'Corridor', 'Kitchen', 'Lecture', 'Corridor',
                      'Corridor', 'Corridor',
                      'Lecture', 'Meeting', 'Stairway', 'Cafeteria', 'Office', 'Lecture',
                      'Hallway',
                      'Cafeteria', 'Courtyard', 'Office', 'Office',
                      ]

        matches = [i for i in range(len(filename_lookup_string)) if filename_lookup_string[i] in
                   self.filename]
        if len(matches) > 0:
            matches = matches[0]
            self.room_name = room_names[matches]
            self.room_type = room_types[matches]
            self.from_database = from_DB[matches]
            if self.is_simulation is None:
                self.is_simulation = False
            self.known_room = True
        else:
            if self.is_simulation is None:
                self.is_simulation = True
            self.known_room = False
