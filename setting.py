import os

__all__ = ['USE_HE', 'PROTOCOL']

# whether to use HE in the offline phase
USE_HE = os.environ.get('FASENET_USE_HE', '0').lower() in ['1', 'true', 'yes']

print("USE_HE:", USE_HE)

# which protocol to use
PROTOCOL = os.environ.get('FASENET_PROTOCOL', 'shuffle')

print("PROTOCOL:", PROTOCOL)
