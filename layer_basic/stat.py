from dataclasses import dataclass

@dataclass
class Stat:
    time_offline: float = 0.0
    byte_offline_send: float = 0.0
    byte_offline_recv: float = 0.0
    # time_offline_send: float = 0.0
    # time_offline_recv: float = 0.0
    time_online: float = 0.0
    byte_online_send: float = 0.0
    byte_online_recv: float = 0.0
    # time_online_send: float = 0.0
    # time_online_recv: float = 0.0
    
    def __add__(self, other):
        return Stat(
            self.time_offline + other.time_offline,
            self.byte_offline_send + other.byte_offline_send,
            self.byte_offline_recv + other.byte_offline_recv,
            self.time_online + other.time_online,
            self.byte_online_send + other.byte_online_send,
            self.byte_online_recv + other.byte_online_recv,
        )

