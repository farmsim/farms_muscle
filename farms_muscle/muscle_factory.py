"""Factory class for generating the muscle model."""

from farms_muscle.geyer_muscle import GeyerMuscle
from farms_muscle.millard_rigid_tendon_muscle import MillarRigidTendonMuscle

class MuscleFactory(object):
    """Implementation of Factory Muscle class.
    """

    def __init__(self):
        """Factory initialization."""
        super(MuscleFactory, self).__init__()
        self._muscles = {'geyer': GeyerMuscle,
                         'millard_rt': MillarRigidTendonMuscle}

    def register_muscle(self, muscle_type, muscle_instance):
        """
        Register a new type of muscle that is a child class of Neuron.
        Parameters
        ----------
        self: type
            description
        muscle_type: <str>
            String to identifier for the muscle.
        muscle_instance: <cls>
            Class of the muscle to register.
        """
        self._muscles[muscle_type] = muscle_instance

    def gen_muscle(self, muscle_type):
        """Generate the necessary type of muscle.
        Parameters
        ----------
        self: type
            description
        muscle_type: <str>
            One of the following list of available muscles.
            1. geyer - Geyer Muscle

        Returns
        -------
        muscle: <cls>
            Appropriate muscle class.
        """
        muscle = self._muscles.get(muscle_type)
        if not muscle:
            raise ValueError(muscle_type)
        return muscle
