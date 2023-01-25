from dataclasses import dataclass

@dataclass
class Stat:
    time_offline: float
    byte_offline_send: float
    byte_offline_recv: float
    # time_offline_send: float
    # time_offline_recv: float
    time_online: float
    byte_online_send: float
    byte_online_recv: float
    # time_online_send: float
    # time_online_recv: float
    
    def __add__(self, other):
        return Stat(
            self.time_offline + other.time_offline,
            self.byte_offline_send + other.byte_offline_send,
            self.byte_offline_recv + other.byte_offline_recv,
            self.time_online + other.time_online,
            self.byte_online_send + other.byte_online_send,
            self.byte_online_recv + other.byte_online_recv,
        )

