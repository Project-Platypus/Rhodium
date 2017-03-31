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

import time
import six
import win32com.client
from win32com.universal import com_error
from .model import *

class ExcelHelper(object):
    
    def __init__(self, filename, sheet=1, visible=False):
        super(ExcelHelper, self).__init__()
        self.xl = win32com.client.Dispatch("Excel.Application")
        self.wb = self.xl.Workbooks.Open(filename)
        
        # ensure auto-calculations is enabled
        sheets = self.xl.Worksheets
        
        for i in range(sheets.Count):
            sheets.Item(i+1).EnableCalculation = True
        
        # set the default sheet
        self.set_sheet(sheet)
        
        if visible:
            self.show()
        
    def set_sheet(self, sheet):
        try:
            self.sheet = self.wb.Sheets(sheet)
            self.sheet_index = sheet
        except com_error:
            raise ValueError("invalid sheet " + str(sheet))
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def show(self):
        self.xl.Visible = 1
        
    def hide(self):
        self.xl.Visible = 0
        
    def close(self):
        if self.wb:
            self.wb.Close(SaveChanges=0)
            self.wb = None
    
        if self.xl:
            self.xl.DisplayAlerts = False
            self.xl.Quit()
            self.xl = None
            
    def _dereference(self, cell):
        try:
            if isinstance(cell, six.string_types):
                return self.sheet.Range(cell)
            elif isinstance(cell, (list, tuple)) and len(cell) == 2:
                return self.sheet.Cells(cell[0], cell[1])
        except com_error:
            pass
        
        raise ValueError("invalid cell reference " + str(cell))
    
    def __getitem__(self, cell):
        return self._dereference(cell).Value
        
    def __setitem__(self, cell, value):
        # expects single value or 2D list/tuple
        if isinstance(value, (list, tuple)):
            value = [v if isinstance(v, (list, tuple)) else [v] for v in value]
        
        self._dereference(cell).Value = value

class ExcelModel(Model):
    
    def __init__(self, filename, **kwargs):
        super(ExcelModel, self).__init__(self._evaluate)
        self.excel_helper = ExcelHelper(filename, **kwargs)
        
    def _evaluate(self, **kwargs):
        result = {}
        
        for parameter in self.parameters:
            if hasattr(parameter, "cell"):
                key = getattr(parameter, "cell")
            else:
                key = parameter.name
                
            if hasattr(parameter, "sheet") and self.excel_helper.sheet_index != getattr(parameter, "sheet"):
                self.excel_helper.set_sheet(getattr(parameter, "sheet"))
                
            value = kwargs.get(parameter.name, parameter.default_value)

            if value is not None:
                self.excel_helper[key] = value
                  
        for response in self.responses:
            if hasattr(response, "cell"):
                key = getattr(response, "cell")
            else:
                key = response.name
                
            if hasattr(response, "sheet") and self.excel_helper.sheet_index != getattr(response, "sheet"):
                self.excel_helper.set_sheet(getattr(response, "sheet"))
                
            result[response.name] = self.excel_helper[key]
            
        return result
        
    def close(self):
        self.excel_helper.close()
        super(ExcelModel, self).close()
        