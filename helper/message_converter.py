import roslib.message
import rospy
import base64
import sys
import copy
import collections

python3 = (sys.hexversion > 0x03000000)

python_list_types = [list, tuple]
python_string_types = [str, unicode]
python_int_types = [int, long]
python_float_types = [float]


ros_to_python_type_map = {
    'bool'    : [bool],
    'float32' : copy.deepcopy(python_float_types + python_int_types),
    'float64' : copy.deepcopy(python_float_types + python_int_types),
    'int8'    : copy.deepcopy(python_int_types),
    'int16'   : copy.deepcopy(python_int_types),
    'int32'   : copy.deepcopy(python_int_types),
    'int64'   : copy.deepcopy(python_int_types),
    'uint8'   : copy.deepcopy(python_int_types),
    'uint16'  : copy.deepcopy(python_int_types),
    'uint32'  : copy.deepcopy(python_int_types),
    'uint64'  : copy.deepcopy(python_int_types),
    'byte'    : copy.deepcopy(python_int_types),
    'char'    : copy.deepcopy(python_int_types),
    'string'  : copy.deepcopy(python_string_types)
}

try:
    import numpy as np
    _ros_to_numpy_type_map = {
        'float32' : [np.float32, np.int8, np.int16, np.uint8, np.uint16],
        # don't include int32, because conversion to float may change value: v = np.iinfo(np.int32).max; np.float32(v) != v
        'float64' : [np.float32, np.float64, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32],
        'int8'    : [np.int8],
        'int16'   : [np.int8, np.int16, np.uint8],
        'int32'   : [np.int8, np.int16, np.int32, np.uint8, np.uint16],
        'int64'   : [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32],
        'uint8'   : [np.uint8],
        'uint16'  : [np.uint8, np.uint16],
        'uint32'  : [np.uint8, np.uint16, np.uint32],
        'uint64'  : [np.uint8, np.uint16, np.uint32, np.uint64],
        'byte'    : [np.int8],
        'char'    : [np.uint8],
    }

    # merge type_maps
    merged = collections.defaultdict(list, ros_to_python_type_map)
    for k, v in _ros_to_numpy_type_map.items():
        merged[k].extend(v)
    ros_to_python_type_map = dict(merged)
except ImportError:
    pass

ros_time_types = ['time', 'duration']
ros_primitive_types = ['bool', 'byte', 'char', 'int8', 'uint8', 'int16',
                       'uint16', 'int32', 'uint32', 'int64', 'uint64',
                       'float32', 'float64', 'string']
ros_header_types = ['Header', 'std_msgs/Header', 'roslib/Header']


def convert_ros_message_to_dictionary(message, binary_array_as_bytes=True):
  dictionary = {}
  message_fields = _get_message_fields(message)


  for field_name, field_type in message_fields:
      field_value = getattr(message, field_name)
      dictionary[field_name] = _convert_from_ros_type(field_type, field_value, binary_array_as_bytes)

  return dictionary


def _get_message_fields(message):
    return zip(message.__slots__, message._slot_types)


def _convert_from_ros_type(field_type, field_value, binary_array_as_bytes=True):
    if field_type in ros_primitive_types:
        field_value = _convert_from_ros_primitive(field_type, field_value)
    elif field_type in ros_time_types:
        field_value = _convert_from_ros_time(field_type, field_value)
    elif _is_ros_binary_type(field_type):
        if binary_array_as_bytes:
            field_value = _convert_from_ros_binary(field_type, field_value)
        elif type(field_value) == str:
            field_value = [ord(v) for v in field_value]
        else:
            field_value = list(field_value)
    elif _is_field_type_a_primitive_array(field_type):
        field_value = list(field_value)
    elif _is_field_type_an_array(field_type):
        field_value = _convert_from_ros_array(field_type, field_value, binary_array_as_bytes)
    else:
        field_value = convert_ros_message_to_dictionary(field_value, binary_array_as_bytes)

    return field_value



def _is_ros_binary_type(field_type):
    """ Checks if the field is a binary array one, fixed size or not
    >>> _is_ros_binary_type("uint8")
    False
    >>> _is_ros_binary_type("uint8[]")
    True
    >>> _is_ros_binary_type("uint8[3]")
    True
    >>> _is_ros_binary_type("char")
    False
    >>> _is_ros_binary_type("char[]")
    True
    >>> _is_ros_binary_type("char[3]")
    True
    """
    return field_type.startswith('uint8[') or field_type.startswith('char[')

def _convert_from_ros_binary(field_type, field_value):
    field_value = base64.b64encode(field_value).decode('utf-8')
    return field_value

def _convert_from_ros_time(field_type, field_value):
    field_value = {
        'secs'  : field_value.secs,
        'nsecs' : field_value.nsecs
    }
    return field_value

def _convert_from_ros_primitive(field_type, field_value):
    # std_msgs/msg/_String.py always calls decode() on python3, so don't do it here
    if field_type == "string" and not python3:
        field_value = field_value.decode('utf-8')
    return field_value

def _convert_from_ros_array(field_type, field_value, binary_array_as_bytes=True):
    # use index to raise ValueError if '[' not present
    list_type = field_type[:field_type.index('[')]
    return [_convert_from_ros_type(list_type, value, binary_array_as_bytes) for value in field_value]

def _get_message_fields(message):
    return zip(message.__slots__, message._slot_types)

def _is_field_type_an_array(field_type):
    return field_type.find('[') >= 0

def _is_field_type_a_primitive_array(field_type):
    bracket_index = field_type.find('[')
    if bracket_index < 0:
        return False
    else:
        list_type = field_type[:bracket_index]
        return list_type in ros_primitive_types
