from farms_muscle.musculo_skeletal_parameters import MuscleParameters
from farms_muscle.muscle_factory import MuscleFactory
import matplotlib.pyplot as plt
import numpy as np
from farms_container import Container

container = Container(MAX_ITERATIONS=50)

mp = MuscleParameters()

factory = MuscleFactory()

degroote = factory.gen_muscle("degroote")

m1 = degroote(mp, physics_engine="NONE")

container.initialize()

fl = np.zeros((50,))
fp = np.zeros((50,))
fv = np.zeros((50,))
ft = np.zeros((50,))
lce = np.linspace(0.15, 1.75, 50)
vce = np.linspace(-1.0, 1.0, 50)
lt = np.linspace(0.95, 1.05, 50)
for j, (l, v) in enumerate(zip(lce, vce)):
    fl[j] = m1._py_force_length(l*0.11)
    fp[j] = m1._py_parallel_star_force(l*0.11)
    fv[j] = m1._py_force_velocity(v*10.0)
    ft[j] = m1._py_tendon_force(lt[j]*0.13)
    container.update_log()

plt.figure()
plt.plot(lce, fl)
plt.plot(lce, fp)
plt.grid(True)
plt.figure()
plt.plot(vce, fv)
plt.grid(True)
plt.figure()
plt.plot(lt, ft)
plt.grid(True)
plt.show()
