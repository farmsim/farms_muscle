from farms_dae.dae_generator import DaeGenerator
from farms_muscle.millard_rigid_tendon_muscle import MillardRigidTendonMuscle
from farms_muscle.musculo_skeletal_parameters import MuscleParameters

mp = MuscleParameters()

dae = DaeGenerator()

m1 = MillardRigidTendonMuscle(dae, mp, physics_engine='NONE')
