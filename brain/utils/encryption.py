"""Simple AES-256-GCM encryption utilities for telemetry storage."""
from __future__ import annotations

import os
from typing import Tuple

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import constant_time


def generate_key() -> bytes:
    """Generate a 32-byte random AES key."""
    return os.urandom(32)


def derive_key_from_password(password: str, salt: bytes, iterations: int = 100000) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations)
    return kdf.derive(password.encode("utf-8"))


def encrypt_bytes(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt bytes with AES-GCM. Output = nonce(12) + ciphertext + tag(16)."""
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    return nonce + ct


def decrypt_bytes(payload: bytes, key: bytes) -> bytes:
    nonce = payload[:12]
    ct = payload[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, associated_data=None)


def encrypt_file(input_path: str, output_path: str, key: bytes) -> None:
    with open(input_path, "rb") as f:
        data = f.read()
    enc = encrypt_bytes(data, key)
    with open(output_path, "wb") as f:
        f.write(enc)


def decrypt_file(input_path: str, output_path: str, key: bytes) -> None:
    with open(input_path, "rb") as f:
        data = f.read()
    dec = decrypt_bytes(data, key)
    with open(output_path, "wb") as f:
        f.write(dec)
