class Gdp:
    def __init__(self, global_attrs, data, variables_attrs):
        self.global_attrs = global_attrs
        self.data = data
        self.variables_attrs = variables_attrs
        self.gridata = None
        self.time = data.index[0]

    def global_attrs(self):
        return self.global_attrs

    def data(self):
        return self.data

    def variables_attrs(self):
        return self.variables_attrs
    
    def set_gridata(self, gridata):
        self.gridata = gridata
    
    def gridata(self):
        if self.gridata:
            return self.gridata
    
    def time(self):
        return self.time