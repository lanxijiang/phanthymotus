"""Immutable xiaoyou01/V6 identity and hashing; no model/backend imports."""
import hashlib
import json
from pathlib import Path

SOURCE_SHA = '451487c68462f0a5400ea49798625d8bf5ea2def858e0cbff56427d7e9d76a14'
FRONTEND_SHA = '548159e3281ffe9bd2040865d50db83e4bb6f49a057199432ce4dd0468e6b53e'
TERMS_SHA = '3a0f34ad00099a1041a8f9e40e30802c08822049c20d9dfbb74905f8fe91dff4'
PHONE_ABI = 'kokoro-v1.0-misaki-0.9.4-xiaoyou-v6'
VOCAB_SHA = '248ee3192acfc27a9907180354a0efaa63389047a6ff78c3f75e2e19466d0b76'
CONFIG_SHA = '5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def profile_sha256(profiles):
    return hashlib.sha256(json.dumps(profiles, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
