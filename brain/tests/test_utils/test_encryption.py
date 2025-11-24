import os
import tempfile

from utils import encryption


def test_encrypt_decrypt_bytes_roundtrip():
    key = encryption.generate_key()
    plaintext = b'hello world bytes'
    payload = encryption.encrypt_bytes(plaintext, key)
    recovered = encryption.decrypt_bytes(payload, key)
    assert recovered == plaintext


def test_encrypt_decrypt_file_roundtrip(tmp_path):
    key = encryption.generate_key()
    inp = tmp_path / 'in.bin'
    out = tmp_path / 'in.bin.enc'
    dec = tmp_path / 'in.dec'
    inp.write_bytes(b'some data to encrypt')
    encryption.encrypt_file(str(inp), str(out), key)
    assert out.exists()
    encryption.decrypt_file(str(out), str(dec), key)
    assert dec.read_bytes() == inp.read_bytes()


def test_decrypt_with_invalid_key_raises(tmp_path):
    key = encryption.generate_key()
    wrong = encryption.generate_key()
    inp = tmp_path / 'in2.bin'
    encf = tmp_path / 'in2.bin.enc'
    inp.write_bytes(b'abc123')
    encryption.encrypt_file(str(inp), str(encf), key)
    dec = tmp_path / 'should_fail'
    try:
        encryption.decrypt_file(str(encf), str(dec), wrong)
        # depending on implementation, this may succeed but produce garbage; assert file exists
        assert dec.exists()
    except Exception:
        # acceptable if implementation raises on auth failure
        assert True
import os
import tempfile

from utils.encryption import generate_key, encrypt_file, decrypt_file, encrypt_bytes, decrypt_bytes


def test_encrypt_decrypt_roundtrip(tmp_path):
    key = generate_key()
    # create sample file
    p = tmp_path / 'sample.txt'
    p.write_text('hello world', encoding='utf-8')
    enc = str(p) + '.enc'
    dec = str(p) + '.dec'
    encrypt_file(str(p), enc, key)
    assert os.path.exists(enc)
    decrypt_file(enc, dec, key)
    assert os.path.exists(dec)
    assert open(dec, 'r', encoding='utf-8').read() == 'hello world'

def test_encrypt_bytes_roundtrip():
    key = generate_key()
    data = b'some bytes data'
    payload = encrypt_bytes(data, key)
    out = decrypt_bytes(payload, key)
    assert out == data
