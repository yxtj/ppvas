import sys
import time
import torch
import socket
from Pyfhel import Pyfhel

from model import minionn
import system
import layer

def show_stat_one(promot:str, s:layer.base.Stat, n:int):
    print("{}: offline time (s): {:.3f}, send (kB): {:.3f}, recv (kB): {:.3f}; "
          "online time (s): {:.3f}, send (kB): {:.3f}, recv (kB): {:.3f}".format(
              promot, s.time_offline, s.byte_offline_send/1000, s.byte_offline_recv/1000,
              s.time_online/n, s.byte_online_send/n/1000, s.byte_online_recv/n/1000,
        ))

def show_stat(layers, n):
    s_total = layer.base.Stat(0, 0, 0, 0, 0, 0)
    s_relu = layer.base.Stat(0, 0, 0, 0, 0, 0)
    s_linear = layer.base.Stat(0, 0, 0, 0, 0, 0)
    s_l_conv = layer.base.Stat(0, 0, 0, 0, 0, 0)
    s_l_fc = layer.base.Stat(0, 0, 0, 0, 0, 0)
    s_pool = layer.base.Stat(0, 0, 0, 0, 0, 0)
    s_sc = layer.base.Stat(0, 0, 0, 0, 0, 0)
    for i, lyr in enumerate(layers):
        print("  Layer {} {}: {}".format(i, lyr.__class__.__name__, lyr.stat))
        s_total += lyr.stat
        if isinstance(lyr, (layer.relu.ReLUClient, layer.relu.ReLUServer)):
            s_relu += lyr.stat
        elif isinstance(lyr, (layer.maxpool.MaxPoolClient, layer.maxpool.MaxPoolServer,
                              layer.avgpool.AvgPoolClient, layer.avgpool.AvgPoolServer)):
            s_pool += lyr.stat
        elif isinstance(lyr, (layer.conv.ConvClient, layer.conv.ConvServer,
                              layer.fc.FcClient, layer.fc.FcServer)):
            s_linear += lyr.stat
            if isinstance(lyr, (layer.conv.ConvClient, layer.conv.ConvServer,)):
                s_l_conv += lyr.stat
            else:
                s_l_fc += lyr.stat
        elif isinstance(lyr, (layer.shortcut.ShortCutClient, layer.shortcut.ShortCutServer)):
            s_sc += lyr.stat
    print()
    show_stat_one("Total", s_total, n)
    show_stat_one("  ReLU", s_relu, n)
    show_stat_one("  Linear", s_linear, n)
    show_stat_one("  Linear-Conv", s_l_conv, n)
    show_stat_one("  Linear-FC", s_l_fc, n)
    show_stat_one("  Shortcut", s_sc, n)
    show_stat_one("  Pool", s_pool, n)


def run_server(host: str, port: int, model: torch.nn.Module, inshape: tuple, n: int):
    # listening on port
    s = socket.create_server((host, port))
    print("Server is running on {}:{}".format(host, port))
    conn, addr = s.accept()
    print("Client connected from: {}".format(addr))
    # initialize server
    t0 = time.time()
    server = system.Server(conn, model, inshape)
    print("Server is ready")
    # offline phase
    server.offline()
    t1 = time.time()
    print("Server offline finished")
    # online phase
    for i in range(n):
        server.online()
    t2 = time.time()
    conn.close()
    # finish
    print("Server online finished")
    print("Quick measure: total offline time: {:.3f}; total online time: {:.3f}, average {}".format(t1 - t0, t2 - t1, (t2 - t1)/n))
    print("Statistics with {} samples: ".format(n))
    show_stat(server.layers, n)


def run_client(host: str, port: int, model: torch.nn.Module, inshape: tuple, he:Pyfhel,
               dataset=None, n: int=1, verify: bool=False):
    assert dataset is None or len(dataset) == n
    # connect to server
    s = socket.create_connection((host, port))
    print("Client is connecting to {}:{}".format(host, port))
    # initialize client
    t0 = time.time()
    client = system.Client(s, model, inshape, he)
    print("Client is ready")
    # offline phase
    client.offline()
    t1 = time.time()
    print("Client offline finished")
    # online phase
    if len(inshape) == 3:
        inshape = (1, *inshape)
    for i in range(n):
        d = torch.rand(inshape) if dataset is None else dataset[i]
        with torch.no_grad():
            res = client.online(d)
        if verify:
            with torch.no_grad():
                res2 = model(d)
            print("Verify {}: mean absolute difference: {:.6g}, mean relative difference: {:.6g}".format(
                i, torch.abs(res - res2).mean(), torch.nanmean(torch.abs(res - res2)/res2)))
    t2 = time.time()
    s.close()
    # finish
    print("Client online finished")
    print("Quick measure: total offline time: {:.3f}; total online time: {:.3f}, average {}".format(t1 - t0, t2 - t1, (t2 - t1)/n))
    print("Statistics with {} samples: ".format(n))
    show_stat(client.layers, n)

