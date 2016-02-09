# Copyright 2015-2016 David Hadka
#
# This file is part of Rhodium, a Python module for robust decision making and
# exploratory modeling.
#
# Rhodium is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Rhodium is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Rhodium.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division, print_function, absolute_import

import re
import six
import ctypes
import itertools
from .model import *

TYPE_RE = re.compile(r"([a-zA-Z ]+)\s*(\*?)\s*((?:[0-9]+)?)")
        
class NativeModel(Model):
    
    def __init__(self, library, function, mode=ctypes.CDLL):
        super(NativeModel, self).__init__(self._evaluate)
        self._cdll = mode(library)
        self._func = getattr(self._cdll, function)
        
    def _str_to_type(self, type, is_pointer=False):
        if type == "bool" or type == "_Bool":
            return ctypes.c_bool
        elif type == "char":
            if is_pointer:
                return ctypes.c_char_p
            else:
                return ctypes.c_char
        elif type == "wchar_t":
            if is_pointer:
                return ctypes.c_wchar
            else:
                return ctypes.c_wchar_p
        elif type == "unsigned char":
            return ctypes.c_ubyte
        elif type == "short":
            return ctypes.c_short
        elif type == "unsigned short":
            return ctypes.c_ushort
        elif type == "int":
            return ctypes.c_int
        elif type == "unsigned int":
            return ctypes.c_uint
        elif type == "long":
            return ctypes.c_long
        elif type == "unsigned long":
            return ctypes.c_ulong
        elif type == "__int64" or type == "long long":
            return ctypes.c_longlong
        elif type == "unsigned __int64" or type == "unsigned long long":
            return ctypes.c_ulonglong
        elif type == "float":
            return ctypes.c_float
        elif type == "double":
            return ctypes.c_double
        elif type == "long double":
            return ctypes.c_longdouble
        elif type == "void" and is_pointer:
            return ctypes.c_void_p
        else:
            raise ValueError("unknown type " + str(type))
         
    def _cast(self, argument, value=None, length=None):
        if hasattr(argument, "type"):
            type = getattr(argument, "type")
                
            if isinstance(type, six.string_types):
                match = TYPE_RE.match(type)
                
                if match:
                    base_type = match.group(1)
                    is_pointer = match.group(2) == "*"
                    
                    if length is None:
                        if len(match.group(3)) > 0:
                            length = int(match.group(3))
                        elif value is not None and isinstance(value, (list, tuple)):
                            length = len(value)
                        else:
                            length = 1
                    
                    type = self._str_to_type(base_type, is_pointer)
                    
                    if is_pointer and type != ctypes.c_void_p and type != ctypes.c_char_p and type != ctypes.c_wchar_p:
                        type = type * length
                    
                    if value is None:
                        return type()
                    elif isinstance(value, (list, tuple)):
                        return type(*value)
                    else:
                        return type(value)
                else:
                    raise ValueError("unparseable type attribute")
            else:
                if value is None:
                    return type()
                elif isinstance(value, (list, tuple)):
                    return type(*value)
                else:
                    return type(value)
        else:
            raise ValueError("missing type attribute")
        
    def _evaluate(self, **kwargs):
        nargs = len(self.parameters) + sum([1 if hasattr(r, "asarg") and getattr(r, "asarg") else 0 for r in self.responses])
        args = [None]*nargs
        arg_map = {}
        locals = {}
        length_args = {}
        
        # first pass aggregates the length arguments
        for parameter in self.parameters:
            if hasattr(parameter, "len_arg"):
                if getattr(parameter, "len_arg") in length_args:
                    if length_args[getattr(parameter, "len_arg")] != len(kwargs[parameter.name]):
                        raise ValueError("arguments using same length argument have different lengths")
                else:
                    length_args[getattr(parameter, "len_arg")] = len(kwargs[parameter.name])
                    
        for k, v in six.iteritems(length_args):
            kwargs[k] = v
    
        # second pass places arguments with the order attribute
        for parameter in self.parameters:
            if hasattr(parameter, "order"):
                arg = self._cast(parameter,
                                 value=kwargs[parameter.name],
                                 length=length_args[getattr(parameter, "len_arg")] if hasattr(parameter, "len_arg") else None)
                arg_map[parameter.name] = getattr(parameter, "order")
                args[getattr(parameter, "order")] = arg
            
        for response in self.responses:
            if hasattr(response, "asarg") and getattr(response, "asarg") and hasattr(response, "order"):
                arg = self._cast(response,
                                 length=length_args[getattr(response, "len_arg")] if hasattr(response, "len_arg") else None)
                arg_map[response.name] = getattr(response, "order")
                locals[response.name] = arg
                args[getattr(response, "order")] = ctypes.byref(arg)
                
        # third pass fills in any remaining arguments
        index = 0
        
        for parameter in self.parameters:
            if not hasattr(parameter, "order"):
                while args[index] is not None:
                    index += 1
                    
                arg = self._cast(parameter,
                                 value=kwargs[parameter.name],
                                 length=length_args[getattr(parameter, "len_arg")] if hasattr(parameter, "len_arg") else None)
                arg_map[parameter.name] = index
                args[index] = arg
            
        for response in self.responses:
            if hasattr(response, "asarg") and getattr(response, "asarg") and not hasattr(response, "order"):
                while args[index] is not None:
                    index += 1
                    
                arg = self._cast(response,
                                 length=length_args[getattr(response, "len_arg")] if hasattr(response, "len_arg") else None)
                arg_map[response.name] = index
                locals[response.name] = arg
                args[index] = ctypes.byref(arg)
                
        # determine if the function will be returning a value
        leftover = [r for r in self.responses if not hasattr(r, "asarg") or not getattr(r, "asarg")]
        
        if len(leftover) > 1:
            raise ValueError("native functions can only return one type")
        elif len(leftover) == 1:
            self._func.restype = type(self._cast(leftover[0]))
                
        # call the function
        value = self._func(*args)
        
        # process the results
        result = {}
        
        for response in self.responses:
            if hasattr(response, "asarg") and getattr(response, "asarg"):
                if hasattr(locals[response.name], "__getitem__"):
                    result[response.name] = [x for x in locals[response.name]]
                else:
                    result[response.name] = locals[response.name].value
            else:
                result[response.name] = value
            
        return result