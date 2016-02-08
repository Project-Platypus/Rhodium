import six
import win32com.client
from win32com.universal import com_error
from .model import *

class ExcelHelper(object):
    
    def __init__(self, filename, sheet=1, visible=False):
        super(ExcelHelper, self).__init__()
        self.xl = win32com.client.Dispatch("Excel.Application")
        self.wb = self.xl.Workbooks.Open(filename)
        
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
        
    def __exit__(self, type, value, traceback):
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
        current_sheet = self.excel_helper.sheet_index
        
        for parameter in self.parameters:
            if hasattr(parameter, "cell"):
                key = getattr(parameter, "cell")
            else:
                key = parameter.name
                
            if hasattr(parameter, "sheet") and current_sheet != getattr(parameter, "sheet"):
                self.excel_helper.set_sheet(getattr(parameter, "sheet"))
                
            value = kwargs.get(parameter.name, parameter.default_value)

            self.excel_helper[key] = value
            
        for response in self.responses:
            if hasattr(response, "cell"):
                key = getattr(response, "cell")
            else:
                key = response.name
                
            if hasattr(parameter, "sheet") and current_sheet != getattr(parameter, "sheet"):
                self.excel_helper.set_sheet(getattr(parameter, "sheet"))
                
            result[response.name] = self.excel_helper[key]
            
        return result
        
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        self.close()
        
    def close(self):
        self.excel_helper.close()