"""Factory class for generating the muscle model."""
from farms_container import Container
from farms_muscle.geyer_muscle import GeyerMuscle
from farms_muscle.millard_rigid_tendon_muscle import MillardRigidTendonMuscle
from farms_muscle.degroote_muscle import DeGrooteMuscle

class MuscleFactory(object):
    """Implementation of Factory Muscle class.
    """

    def __init__(self):
        """Factory initialization."""
        super(MuscleFactory, self).__init__()

        container = Container.get_instance()
        #: Create muscles namespace in the container
        container.add_namespace('muscles')
        
        #: Attributes
        #: ODE States
        self.states = container.muscles.add_table('states')
        self.dstates = container.muscles.add_table('dstates')
        #: Muscle parameters
        self.constants = container.muscles.add_table(
            'constants', TABLE_TYPE='CONSTANT')
        self.parameters = container.muscles.add_table('parameters')
        #: Input to each muscle
        self.activations = container.muscles.add_table('activations')
        #: Output of each muscle
        self.forces= container.muscles.add_table('forces')
        #: Secondary outputs 
        self.outputs = container.muscles.add_table('outputs')
        #: Sensors
        self.Ia = container.muscles.add_table('Ia')
        self.II = container.muscles.add_table('II')
        self.Ib = container.muscles.add_table('Ib')
        
        self._muscles = {'geyer': GeyerMuscle,
                         'millard_rt': MillardRigidTendonMuscle,
                         'degroote': DeGrooteMuscle}

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
        muscle = self._muscles.get(muscle_type.lower())
        if not muscle:
            raise ValueError(muscle_type)
        return muscle
