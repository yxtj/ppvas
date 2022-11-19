
class Activation():
    
    def input_share_server(self, data):
        return data
    
    def input_share_client(self, data):
        return data
    
    def activate(self, data_s, data_c):
        # relu for data_s + data_c
        res = data_s + data_c
        mask = (res <= 0)
        res[mask] = 0
        return res
    
    def output_share_server(self, data):
        return data
    
    def output_share_client(self, data):
        return data
    
    def work(self, data_s, data_c):
        i_s = self.input_share_server(data_s)
        i_c = self.input_share_client(data_c)
        r = self.activate(i_s, i_c)
        o_s = self.output_share_server(r)
        o_c = self.output_share_client(r)
        return o_s, o_c
        
