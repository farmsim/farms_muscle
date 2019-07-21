""" System Parameters """
import uuid


class SystemParameters(object):
    """Parent class providing main attributes for other sub system
    parameters.

    """

    def __init__(self, sys_name='System'):
        super(SystemParameters, self).__init__()
        self.sys_name = sys_name

    def showParameters(self):
        raise NotImplementedError()

    def msg(self, parameters, units, endl="\n" + 4 * " "):
        """ Message """
        to_print = ("{} parameters : ".format(self.sys_name)) + endl
        for param in parameters:
            to_print += ("{} : {} [{}]".format(
                param,
                parameters[param],
                units[param]
            )) + endl
        return to_print


class MuscleParameters(SystemParameters):
    """ Muscle parameters

    with:
        Muscle Parameters:
            - l_slack : Tendon slack length [m]
            - l_opt : Contracticle element optimal fiber length [m]
            - f_max : Maximum force produced by the muscle [N]
            - v_max : Maximum velocity of the contracticle element [m/s]
            - pennation : Fiber pennation angle

            - muscle_type : Type of muscle
                    ['mono' - Mono articular]
                    ['bi' - Bi articular]
            - r_0 : Muscle maximum moment arm across joint 1 [m]
            - joint_attach : Joint to which the muscle attaches <str>
            - theta_max : Joint 1 angle at which maximal torque is
                          generated
            - theta_ref : Jonint 1 angle at which muscle length is at its rest
                          length
            - direction : Direction of torque applied on joint 1
                          ['clockwise/cclockwise']

    Examples:

        >>> muscle_parameters = MuscleParameters(l_slack=0.2, l_opt=0.1)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Muscle Parameters
        - l_slack = 0.13
        - l_opt = 0.11
        - f_max = 1500
        - v_max = 1.2
        - pennation = 1.

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.l_slack = 0.2 # Reassign tendon slack constant

        To assign to another variable from within the class:

        >>> example_l_slack = self.l_slack

    You can display the parameters using:

    >>> muscle_joint_parameters = MuscleParameters()
    >>> print(muscle_parameters,showParameters())
    Muscle parameters :
            f_max : 1500 [N]
            v_max : 1.2 [m/s]
            pennation : 1 []
            l_slack : 0.13 [m]
            l_opt : 0.11 [m]

    Or using print:

    >>> muscle_parameters = MuscleParameters()
    >>> print(muscle_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(MuscleParameters, self).__init__('muscle')
        self.parameters = {}
        self.units = {}

        self.units['model'] = '<str>'
        self.units['l_slack'] = 'm'
        self.units['l_opt'] = 'm'
        self.units['f_max'] = 'N'
        self.units['v_max'] = 'm/s'
        self.units['pennation'] = ''
        self.units['name'] = '<str>'
        self.units['muscle_type'] = '<str>'
        self.units['m_id'] = '<str>'
        self.units['l_ce0'] = 'm'
        self.units['a0'] = '[0-1]'
        self.units['td_to_sc'] = 'ms'
        self.units['td_from_sc'] = 'ms'
        self.units['motiontype'] = '<str>'

        self.parameters['model'] = kwargs.get('model', 'geyer')
        self.parameters['m_id'] = kwargs.get('m_id', uuid.uuid4())
        self.parameters['l_slack'] = kwargs.get('l_slack', 0.13)
        self.parameters['l_opt'] = kwargs.get('l_opt', 0.11)
        self.parameters['f_max'] = kwargs.get('f_max', 1500)
        self.parameters['v_max'] = kwargs.get('v_max', 1.2)
        self.parameters['pennation'] = kwargs.get('pennation', 1)
        self.parameters['name'] = kwargs.get('name', 'muscle')
        self.parameters['muscle_type'] = kwargs.get('muscle_type', 'None')
        self.parameters['l_ce0'] = kwargs.get(
            'l_ce0', self.parameters['l_opt'])
        self.parameters['a0'] = kwargs.get('a0', 0.05)
        self.parameters['td_to_sc'] = kwargs.get('td_to_sc', 5)
        self.parameters['td_from_sc'] = kwargs.get('td_from_sc', 5)
        self.parameters['motiontype'] = kwargs.get('motiontype', 'flexor')

    @property
    def m_id(self):
        """Unique muscle id  """
        return self.parameters['m_id']

    @m_id.setter
    def m_id(self, value):
        """
        Parameters
        ----------
        value : <str>
            Set unique muscle id
        """
        self.parameters['m_id'] = value

    @property
    def l_ce0(self):
        """ Contractile element initial length  """
        return self.parameters['l_ce0']

    @l_ce0.setter
    def l_ce0(self, value):
        """
        Parameters
        ----------
        value : <float>
            Set contractile element initial length
        """
        self.parameters['l_ce0'] = value

    @property
    def td_to_sc(self):
        """ Time delay to spinal cord  """
        return self.parameters['td_to_sc']

    @td_to_sc.setter
    def td_to_sc(self, value):
        """
        Parameters
        ----------
        value : <float>
            Set time delay to spinal cord
        """
        self.parameters['td_to_sc'] = value

    @property
    def td_from_sc(self):
        """ Time delay from spinal cord  """
        return self.parameters['td_from_sc']

    @td_from_sc.setter
    def td_from_sc(self, value):
        """
        Parameters
        ----------
        value : <float>
            Set time delay from spinal cord
        """
        self.parameters['td_from_sc'] = value

    @property
    def a0(self):
        """ Activation initial value  """
        return self.parameters['a0']

    @a0.setter
    def a0(self, value):
        """
        Parameters
        ----------
        value : <float>
            Set activation initial value
        """
        self.parameters['a0'] = value

    @property
    def l_slack(self):
        """ Muscle Tendon Slack length [m]  """
        return self.parameters['l_slack']

    @l_slack.setter
    def l_slack(self, value):
        """ Keyword Arguments:
            value -- Muscle Tendon Slack Length [m]"""
        self.parameters['l_slack'] = value

    @property
    def l_opt(self):
        """ Muscle Optimal Fiber Length [m]  """
        return self.parameters['l_opt']

    @l_opt.setter
    def l_opt(self, value):
        """ Keyword Arguments:
        value -- Muscle Optimal Fiber Length [m]"""
        self.parameters['l_opt'] = value

    @property
    def f_max(self):
        """ Maximum tendon force produced by the muscle [N]  """
        return self.parameters['f_max']

    @f_max.setter
    def f_max(self, value):
        """ Keyword Arguments:
        value -- Maximum tendon force produced by the muscle [N]"""
        self.parameters['f_max'] = value

    @property
    def v_max(self):
        """ Maximum velocity of the contractile element [m/s]  """
        return self.parameters['v_max']

    @v_max.setter
    def v_max(self, value):
        """ Keyword Arguments:
        value -- Maximum velocity of the contractile element [m/s] """
        self.parameters['v_max'] = value

    @property
    def pennation(self):
        """ Muscle fiber pennation angle  """
        return self.parameters['pennation']

    @pennation.setter
    def pennation(self, value):
        """ Keyword Arguments:
            value -- Muscle fiber pennation angle """
        self.parameters['pennation'] = value

    @property
    def motiontype(self):
        """ Muscle motion direction, anatomical term  """
        return self.parameters['motiontype']

    @motiontype.setter
    def motiontype(self, value):
        """ Keyword Arguments:
            value -- Muscle motion direction, anatomical term """
        self.parameters['motiontype'] = value

    @property
    def name(self):
        """ Name of the muscle. """
        return self.parameters['name']

    @name.setter
    def name(self, value):
        """Keyword Arguments:
           value --  Name of the muscle """
        self.parameters['name'] = value

    @property
    def muscle_type(self):
        """ Name of the muscle. """
        return self.parameters['muscle_type']

    @muscle_type.setter
    def muscle_type(self, value):
        """Keyword Arguments:
           value --  Name of the muscle """
        self.parameters['muscle_type'] = value

    @property
    def model(self):
        """ Model type of the muscle. """
        return self.parameters['model']

    @model.setter
    def model(self, value):
        """Keyword Arguments:
           value --  Model type of the muscle """
        self.parameters['model'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


class JointParameters(SystemParameters):
    """ Parameters that define the interface between joint.

    with:
        Joint Parameters:
            - name : Name of the joint <str>
            - theta_max : Maximum allowed joint angle rotation
            - theta_min : Minimum allowed joint angle rotation
            - reference_angle : Joint offset from the simulation anlge
            - joint_type : Moment arm computation type <str> [GEYER/CONSTANT]
    Examples:

        >>> joint_parameters = JointParameters(name='Joint1',
                                                            theta_min=0.0)

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Joint Parameters
        - name = 'joint'
        - theta_max = 1.5707
        - theta_min = -1.5707
        - reference_angle = 0.0
        - joint_type = 'GEYER'

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.name = 'Joint' # Reassign tendon slack constant
        To assign to another variable from within the class:


        >>> example_joint_name = self.name

    You can display the parameters using:

    >>> joint_parameters = JointParameters()
    >>> print(joint_parameters,showParameters())
    Joint parameters :
        joint parameters :
            theta_min : -90 [deg]
            joint_type : GEYER [<str>]
            reference_angle : 0.0 [rad]
            name : joint [<str>]
            theta_max : 90 [deg]


    Or using :

    >>> joint_parameters = JointParameters()
    >>> .info(joint_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(JointParameters, self).__init__('joint')
        self.parameters = {}
        self.units = {}

        self.units['model'] = '<str>'
        self.units['name'] = '<str>'
        self.units['joint_max'] = 'deg'
        self.units['joint_min'] = 'deg'
        self.units['reference_angle'] = 'deg'
        self.units['init_angle'] = 'deg'
        self.units['joint_type'] = '<str>'
        self.units['passive_stiffness'] = 'Nm/rad'

        self.parameters['model'] = kwargs.get('model', 'geyer')
        self.parameters['name'] = kwargs.get('name', 'joint')
        self.parameters['joint_max'] = kwargs.get('joint_max', 90)
        self.parameters['joint_min'] = kwargs.get('joint_min', -90)
        self.parameters['reference_angle'] = kwargs.get('reference_angle', 0.0)
        self.parameters['init_angle'] = kwargs.get('init_angle', 0.0)
        self.parameters['joint_type'] = kwargs.get('joint_type', 'GEYER')
        self.parameters['passive_stiffness'] = kwargs.get(
            'passive_stiffness', 0.0)

    @property
    def name(self):
        """Name of the joint.  """
        return self.parameters['name']

    @name.setter
    def name(self, value):
        """Keyword Arguments:
           value --  Name of the joint <str> """
        self.parameters['name'] = value

    @property
    def joint_max(self):
        """Maximum allowed joint rotation.  """
        return self.parameters['joint_max']

    @joint_max.setter
    def joint_max(self, value):
        """Keyword Arguments:
           value --  Maximum allowed joint rotation [rad] """
        self.parameters['joint_max'] = value

    @property
    def joint_min(self):
        """Minimum allowed joint rotation.  """
        return self.parameters['joint_min']

    @joint_min.setter
    def joint_min(self, value):
        """Keyword Arguments:
           value --  Minimum allowed joint rotation [rad] """
        self.parameters['joint_min'] = value

    @property
    def reference_angle(self):
        """Joint reference/offset in angle measurement.  """
        return self.parameters['reference_angle']

    @reference_angle.setter
    def reference_angle(self, value):
        """Keyword Arguments:
           value --  Joint reference/offset in angle measurement [rad] """
        self.parameters['reference_angle'] = value

    @property
    def init_angle(self):
        """Joint reference/offset in angle measurement.  """
        return self.parameters['init_angle']

    @init_angle.setter
    def init_angle(self, value):
        """Keyword Arguments:
           value --  Joint reference/offset in angle measurement [rad] """
        self.parameters['init_angle'] = value

    @property
    def joint_type(self):
        """Type of joint moment arm computation  """
        return self.parameters['joint_type']

    @joint_type.setter
    def joint_type(self, value):
        """Keyword Arguments:
           value --  Type of joint moment arm computation """
        self.parameters['joint_type'] = value

    @property
    def passive_stiffness(self):
        """ Model type of the muscle. """
        return self.parameters['passive_stiffness']

    @passive_stiffness.setter
    def passive_stiffness(self, value):
        """Keyword Arguments:
           value --  Model type of the muscle """
        self.parameters['passive_stiffness'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)

    @property
    def model(self):
        """ Model type of the muscle. """
        return self.parameters['model']

    @model.setter
    def model(self, value):
        """Keyword Arguments:
           value --  Model type of the muscle """
        self.parameters['model'] = value

    def showParameters(self):
        return self.msg(self.parameters, self.units)


class MuscleJointParameters(SystemParameters):
    """ Parameters that define the interface between joint and muscle.

    with:
        Muscle Joint Parameters:
            - muscle_type : Type of muscle
                    ['mono' - Mono articular]
                    ['bi' - Bi articular]
            - r_0 : Muscle maximum moment arm across joint 1 [m]
            - joint_attach : Joint to which the muscle attaches <str>
            - theta_max : Joint 1 angle at which maximal torque is
                          generated
            - theta_ref : Jonint 1 angle at which muscle length is at its rest
                          length
            - direction : Direction of torque applied on joint 1
                          ['clockwise/cclockwise']
    Examples:

        >>> muscle_joint_parameters = (
        ...     MuscleJointParameters(m_type='mono', r_0 = 0.002)
        ... )

    Note that not giving arguments to instanciate the object will result in the
    following default values:
        # Muscle Joint Parameters
        - muscle_type = 'mono'
        - r_0 = 1
        - joint_attach = 'LH_J_HIP'
        - theta_max = 0.0
        - theta_ref = 0.0
        - direction = 'clockwise'

    These parameter variables can then be called from within the class using
    for example:

        To assign a new value to the object variable from within the class:

        >>> self.m_type = 'bi' # Reassign tendon slack constant
        To assign to another variable from within the class:


        >>> example_muscle_m_type = self.m_type

    You can display the parameters using:

    >>> muscle_joint_parameters = MuscleJointParameters()
    >>> print(muscle_joint_parameters,showParameters())
    Muscle Joint parameters :
        muscle_type = 'mono'
        r_0 = 1
        joint_attach = 'LH_J_HIP'
        theta_max = 0.0
        theta_ref = 0.0
        direction = 'clockwise'

    Or using :

    >>> muscle_joint_parameters = MuscleJointParameters()
    >>> .info(muscle_joint_parameters.showParameters())
    """

    def __init__(self, **kwargs):
        super(MuscleJointParameters, self).__init__('muscle-joint')
        self.parameters = {}
        self.parameters['angle_idx'] = None
        self.parameters['muscle_force_idx'] = None
        self.parameters['r_0'] = None
        self.parameters['joint_type'] = None
        self.parameters['theta_max'] = None
        self.parameters['theta_ref'] = None
        self.parameters['direction'] = None

        self.units = {}
        self.units['angle_idx'] = 'int'
        self.units['muscle_force_idx'] = 'int'
        self.units['muscle_type'] = '<str>'
        self.units['r_0'] = 'm'
        self.units['joint_type'] = '<str>'
        self.units['theta_max'] = 'rad'
        self.units['theta_ref'] = 'rad'

        self.angle_idx = kwargs.pop('angle_idx', 0)
        self.muscle_force_idx = kwargs.pop('muscle_force_idx', 0)
        self.r_0 = kwargs.pop('r_0', '1.')
        self.joint_type = kwargs.pop('joint_type', 0)
        self.theta_max = kwargs.pop('theta_max', 0.0)
        self.theta_ref = kwargs.pop('theta_ref', 0.0)
        self.direction = kwargs.pop('direction', 'clockwise')

    @property
    def angle_idx(self):
        """Set the index of angle in the table  """
        return self.parameters['angle_idx']

    @angle_idx.setter
    def angle_idx(self, value):
        """
        Parameters
        ----------
        value : Index of the angle in the table
        """
        self.parameters['angle_idx'] = value

    @property
    def muscle_force_idx(self):
        """Set the index of muscle force in the table  """
        return self.parameters['muscle_force_idx']

    @muscle_force_idx.setter
    def muscle_force_idx(self, value):
        """
        Parameters
        ----------
        value : Index of the muscle force in the table
        """
        self.parameters['muscle_force_idx'] = value

    @property
    def r_0(self):
        """Maximum muscle moment arm [m]  """
        return self.parameters['r_0']

    @r_0.setter
    def r_0(self, value):
        """ Keyword Arguments:
        value -- Maximum muscle moment arm [m]"""
        if value < 0.0:
            ('Muscle moment arm cannot be negative!')
        else:
            self.parameters['r_0'] = value

    @property
    def joint_type(self):
        """Joint type."""
        return self.parameters['joint_type']

    @joint_type.setter
    def joint_type(self, value):
        """ Keyword Arguments:
        value -- Type of joint"""
        _jtype = 0
        if value == 'CONSTANT':
            _jtype = 1
        elif value == 'GEYER':
            _jtype = 2
        self.parameters['joint_type'] = _jtype

    @property
    def theta_max(self):
        """Joint angle at which maximum muscle torque is applied [rad]  """
        return self.parameters['theta_max']

    @theta_max.setter
    def theta_max(self, value):
        """ Keyword Arguments:
        value -- Joint angle at which maximum muscle torque is applied [rad]"""
        self.parameters['theta_max'] = value

    @property
    def theta_ref(self):
        """Joint angle muscle length is at its rest length [rad]  """
        return self.parameters['theta_ref']

    @theta_ref.setter
    def theta_ref(self, value):
        """ Keyword Arguments:
        value -- Joint angle muscle length is at its rest length [rad]  """
        self.parameters['theta_ref'] = value

    @property
    def direction(self):
        """Direction of muscle torque. <str>
            'clockwise'
            'cclockwise'. """
        return self.parameters['direction']

    @direction.setter
    def direction(self, value):
        """ Keyword Arguments:
        value -- Direction of muscle torque. <str>
            'clockwise'
            'cclockwise'. """
        if value == 'clockwise':
            _dir = -1
        elif value == 'cclockwise':
            _dir = 1
        self.parameters['direction'] = _dir

    def showParameters(self):
        return self.msg(self.parameters, self.units)


if __name__ == '__main__':
    M = MuscleParameters()
    print((M.showParameters()))

    J = JointParameters()
    print((J.showParameters()))
