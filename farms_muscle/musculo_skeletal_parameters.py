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
        self.units['l_ce0'] = 'm'
        self.units['a0'] = '[0-1]'
        self.units['waypoints'] = '<list>'

        self.parameters['model'] = kwargs.get('model', 'geyer')
        self.parameters['l_slack'] = kwargs.get('l_slack', 0.13)
        self.parameters['l_opt'] = kwargs.get('l_opt', 0.11)
        self.parameters['f_max'] = kwargs.get('f_max', 1500)
        self.parameters['v_max'] = kwargs.get('v_max', 1.2)
        self.parameters['pennation'] = kwargs.get('pennation', 1)
        self.parameters['name'] = kwargs.get('name', uuid.uuid4())
        self.parameters['muscle_type'] = kwargs.get('muscle_type', 'None')
        self.parameters['l_ce0'] = kwargs.get(
            'l_ce0', self.parameters['l_opt'])
        self.parameters['a0'] = kwargs.get('a0', 0.05)
        self.parameters['waypoints'] = kwargs.get('waypoints', [])

    @property
    def waypoints(self):
        """ Waypoints describing muscle attachments  """
        return self.parameters['waypoints']
        
    @waypoints.setter
    def waypoints(self, value):
        """
        Parameters
        ----------
        value : <list>
            List of waypoints describing the muscle attachment in local link frame
        """
        self.parameters['waypoints'] = value
        

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


if __name__ == '__main__':
    M = MuscleParameters()
    print((M.showParameters()))
